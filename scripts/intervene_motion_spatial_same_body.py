"""Same-prefix/body/coarse-scene interventions at the new motion boundary only."""
import argparse,gc,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.motion_spatial_residual_port import MotionSpatialResidualPort
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read
from flybrain_drive.sustained_turn_diagnostic import paired_sustained_onsets
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.train_qualified_sac_unified import context


def state_digest(state):
    digest=hashlib.sha256()
    def add(value):
        if isinstance(value,torch.Tensor):
            a=value.detach().cpu().numpy();digest.update(str((a.dtype,a.shape)).encode());digest.update(a.tobytes())
        elif isinstance(value,dict):
            for key in sorted(value):digest.update(key.encode());add(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(str(type(value)).encode())
            for x in value:add(x)
        else:digest.update(repr(value).encode())
    add(state);return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True);p.add_argument('--fit-audit',required=True);p.add_argument('--episode',type=int,choices=[0,1,2],required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(a.fit);r=read(root/'result.json');m=read(root/'manifest.json')['config'];audit=Path(a.fit_audit);ar=read(audit/'result.json');am=read(audit/'manifest.json')['config']
    if r['status']!='completed_motion_spatial_port_fit' or r['arm']!='fine_motion' or ar['status']!='completed_independent_motion_spatial_port_audit' or am['fit_result_sha256']!=sha256(root/'result.json') or len(ar['full_runtime_episodes'])!=3:raise ValueError('Full audited fine-motion candidate required')
    (components,receipt,entries,items,extras,native,valid,baseline,pre,features,maps,coarse,reference,scale)=load_motion_experiment(m['plan'],'fine_motion');model,head,dn,optical,body_port,_=components;cp=torch.load(root/'motion-spatial.pt',map_location='cpu',weights_only=True);assert sha256(root/'motion-spatial.pt')==r['checkpoint_sha256'];port=MotionSpatialResidualPort(reference,scale,len(model.motor.backbone.vision_ports),maximum_logit_delta=cp['maximum_logit_delta']);port.load_state_dict(cp['motion_port'],strict=True);port.eval().requires_grad_(False)
    e=a.episode;d,c,_=items[e];windows=paired_sustained_onsets(d,-1)
    if not windows:raise ValueError('No paired opposite sustained corner stimuli; no fabricated pairs')
    windows=[dict(w,end=int(np.searchsorted(d['decision_time'],w['time']+.25,side='right'))) for w in windows];starts={w['start'] for w in windows};states={};holder=dict(i=0,case='normal',window=None);goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool);scene=torch.tensor(d['scene'],dtype=torch.float32)
    def descending(module,args,value):
        i=holder['i'];steer,power=value;return steer+dn(extras[e][i:i+1]),power
    def visual(module,args,value):
        i=holder['i'];case=holder['case'];j=int(maps[e][i]);x=features[e][j:j+1]
        if case=='mean_motion':x=reference[None]
        elif case=='opposite_motion':
            w=holder['window'];stamp=w['donor_time']+float(d['decision_time'][i])-w['time'];k=int(np.searchsorted(d['decision_time'],stamp,side='right')-1)
            if not 0<=k<len(scene) or stamp>d['decision_time'][-1]:raise ValueError('Donor relative clock outside original episode')
            x=features[e][int(maps[e][k]):int(maps[e][k])+1]
        return value+optical(extras[e][i:i+1,64:])+port(x)
    def body(module,args,value):
        i=holder['i'];assert torch.equal(args[0],torch.tensor(d['body'][i:i+1],dtype=torch.float32));return value+body_port(native[e][i:i+1],valid[e][i:i+1])
    hooks=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(visual),model.body_feedback.register_forward_hook(body)];run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),fit_result_sha256=sha256(root/'result.json'),fit_audit_sha256=sha256(audit/'result.json'),entry=entries[e],windows=windows,seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='Fine-motion feature-boundary intervention only. Body/history/native/old scene/static clocks and initial neural prefix state fixed. Opposite donor road pose not matched and no independent counterfactual action truth; not pixel/FlyVis resimulation or physical turn success.'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py');saved={};reports=[];before=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False);start=time.perf_counter()
    def step(i):
        holder['i']=i
        with torch.no_grad():
            _,mu=model.sensory_decision(decision_time=float(d['decision_time'][i]),racing_goals=goals,scene_features=scene[i:i+1],scene_valid=views,body_context=context(d,c,i));return head(mu)[0].numpy().copy()
    def receiver_hashes(begin,end):
        values={name:np.asarray(d[name][begin:end]) for name in ['body','body_valid','view_body','view_body_valid','body_packet','body_receipt','decision_time','render_lowers','camera_receipt','scene']}
        values.update(native19=native[e][begin:end].numpy(),native19_valid=valid[e][begin:end].numpy(),old_static=extras[e][begin:end].numpy())
        return {name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in values.items()}
    try:
        if not torch.set_flush_denormal(True):raise ValueError('Original numerical mode unavailable')
        model.reset_episode();full=[]
        for i in range(len(scene)):
            if i in starts:states[i]=snapshot(model)
            full.append(step(i))
            if i%1000==0:write_json(run/'progress.json',dict(phase='normal_prefix_replay',event=i,total=len(scene),elapsed_seconds=time.perf_counter()-start))
        with np.load(root/'predictions.npz') as z:reference_prediction=z[f'prediction_{e}'].copy()
        full=np.asarray(full);error=float(abs(full-reference_prediction).max());assert error<=2e-6
        for k,w in enumerate(windows):
            begin,end=w['start'],w['end'];mask=pre[e][begin:end];indices=np.arange(begin,end)[mask]
            if not len(indices):raise ValueError('No original pre-corner labels in paired window')
            input_hashes=receiver_hashes(begin,end);branch={};initial=state_digest(states[begin])
            for case in ['normal','mean_motion','opposite_motion']:
                restore(model,states[begin]);assert state_digest(snapshot(model))==initial;holder.update(case=case,window=w);prediction=np.asarray([step(i) for i in range(begin,end)]);saved[f'window_{k}_{case}']=prediction;branch[case]=prediction
            assert float(abs(branch['normal']-reference_prediction[begin:end]).max())<=2e-6
            assert input_hashes==receiver_hashes(begin,end)
            normal=branch['normal'][mask,0];opposite=branch['opposite_motion'][mask,0];mean=branch['mean_motion'][mask,0];target=d['labels'][indices,0];directional=float(w['direction']*(normal-opposite).mean());reports.append(dict(window=k,receiver=w,initial_state_sha256=initial,receiver_input_sha256=input_hashes,pre_indices=indices.tolist(),normal_pre_rmse=float(np.sqrt(np.mean((normal-target)**2))),mean_motion_pre_rmse=float(np.sqrt(np.mean((mean-target)**2))),opposite_motion_pre_rmse=float(np.sqrt(np.mean((opposite-target)**2))),maximum_absolute_opposite_steering_change=float(abs(opposite-normal).max()),hypothesized_directional_ordering_margin=directional,hypothesized_directional_ordering_positive=bool(directional>1e-6)))
            write_json(run/'completed-windows.json',reports)
        result.update(status='completed_motion_spatial_same_body_interventions',model_events=len(full),full_replay_max_difference=error,paired_windows=len(reports),hypothesized_direction_windows=sum(w['hypothesized_directional_ordering_positive'] for w in reports),windows=reports,physical_counterfactual_action_correctness_verified=False,optimizer_updates=0,termination_reason='Original fine candidate replay and restored same-state/body/coarse-scene fine-motion removal/opposite donor branches completed; expected ordering is a diagnostic hypothesis, not independently correct counterfactual driving.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),windows=reports)
    finally:
        for hook in hooks:hook.remove()
        torch.set_flush_denormal(False);gc.enable() if before else gc.disable()
    np.savez_compressed(run/'branches.npz',**saved);result['branches_sha256']=sha256(run/'branches.npz');result['elapsed_seconds']=time.perf_counter()-start;write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
