"""Source-bound internal states through the original fixed flight circuit.

Premotor candidates are defined by one measured/signed graph edge to selected
wing motor neurons. Structural proximity is not physiological qualification.
"""
import argparse,gc,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from scripts.evaluate_controlled_motion_factorial import synthetic_body
from scripts.sac_steering_experiment import snapshot,restore
from scripts.audit_controlled_motion_function import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);plan=read(a.source_plan)
    for path,value in plan['source_sha256'].items():
        if sha256(path)!=value:raise ValueError('Sealed stage source changed '+path)
    assert sha256(plan['candidate_plan'])==plan['candidate_plan_sha256'];components,motion,parent=load_readonly_motion_candidate(read(plan['candidate_plan']));model,head,dn,optical,body_port,_=components
    for module in [*components[:5],motion]:module.eval().requires_grad_(False)
    before_hash=[digest(module.state_dict()) for module in [*components[:5],motion]];b=model.motor.backbone;c=b.circuit;matrix=c.matrix.coalesce();edge=matrix.indices().numpy();active=matrix.values().numpy()!=0;members={name:c.output_ports[getattr(b.readout,name)].numpy() for name in ['b1_L','b1_R','b2_L','b2_R','power_L','power_R']};motor=np.unique(np.concatenate(list(members.values())));external=np.concatenate([b.vision_ports.numpy(),b.body_ports.numpy(),b.steering_ports.numpy(),b.goal_ports.numpy()]);premotor=np.setdiff1d(np.unique(edge[1,active & np.isin(edge[0],motor)]),np.concatenate([external,motor]));groups=dict(visual_state=b.vision_ports.numpy(),descending_state=b.steering_ports.numpy(),presynaptic_candidate_state=premotor,individual_motor_state=motor);columns=np.unique(np.concatenate(list(groups.values())));group_columns={name:np.searchsorted(columns,values).tolist() for name,values in groups.items()};spec=read('configs/flight-circuit-separated-v5-20261029.json');assert sha256('configs/flight-circuit-separated-v5-20261029.json')==b.spec_sha256 and sha256(spec['graph_file'])==spec['graph_sha256'] and sha256(spec['annotations_file'])==spec['annotations_sha256'];annotation={str(v['bodyId']):v for v in read(spec['annotations_file'])}
    with np.load(spec['graph_file']) as z:ids=z['ids'].copy()
    revision=subprocess.run(['git','-c','safe.directory='+str(Path.cwd()).replace('\\','/'),'rev-parse','HEAD'],text=True,capture_output=True);assert revision.returncode==0
    out=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.source_plan),candidate=parent,source_revision_observed_before_execution=revision.stdout.strip(),program_sha256=sha256(__file__),state_columns=columns.tolist(),stage_columns=group_columns,motor_group_state_indices={k:v.tolist() for k,v in members.items()},temperature_c=80.,seed=None,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope=__doc__));shutil.copyfile(__file__,out/'executed-entrypoint.py');write_json(out/'neurons.json',[dict(state_index=int(i),graph_body_id=str(ids[i]),annotation=annotation.get(str(ids[i]))) for i in columns]);holder={};hooks=[];saved={};reports=[];goals=torch.tensor(racing_goal(300,1));views=torch.ones(1,4,dtype=torch.bool);contract=sha256('configs/ac-virtual-navigation-body-v2.json');start=time.perf_counter();enabled=gc.isenabled();gc.disable();result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False)
    def descending(module,args,value):
        k,i=holder['case'],holder['frame'];steer,power=value;return steer+dn(torch.tensor(holder['data']['static'][k,i:i+1])),power
    def visual(module,args,value):
        k,i=holder['case'],holder['frame'];x=holder['data'];pooled=value;fine=motion(torch.tensor(x['motion'][k,i:i+1]));static=optical(torch.tensor(x['static'][k,i:i+1,64:]));holder['visual_logits']=pooled+fine+static;holder['fine_logits']=fine;return holder['visual_logits']
    def body(module,args,value):return value+body_port(holder['native'],holder['native_valid'])
    hooks=[model.motor.command.register_forward_hook(descending),model.visual.register_forward_hook(visual),model.body_feedback.register_forward_hook(body)]
    def step(case,i):
        x=holder['data'];holder.update(case=case,frame=i);clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005;values,mask=holder['body'],holder['mask'];current=NavigationBodyObservation('synthetic-factorial',1000+i,now-.001,now,values.copy(),mask.copy(),contract);history=tuple(NavigationBodyObservation('synthetic-factorial',10*i+j,t-.001,camera,values.copy(),mask.copy(),contract) for j,t in enumerate(clocks));ctx=per_view_temporal_body_context(history,current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now)
        with torch.no_grad():
            _,muscles=model.sensory_decision(decision_time=now,racing_goals=goals,scene_features=torch.tensor(x['scene'][case,i:i+1]),scene_valid=views,body_context=ctx);prediction=head(muscles)[0].numpy().copy();return dict(state=model.state[0,columns].numpy().copy(),visual_current=model.previous_drive[0,b.vision_ports].numpy().copy(),visual_logits=holder['visual_logits'][0].numpy().copy(),fine_logits=holder['fine_logits'][0].numpy().copy(),muscles=muscles[0].numpy().copy(),prediction=prediction)
    try:
        torch.set_flush_denormal(True);write_json(out/'backend-initialization.json',prepare_fixed_cpu_backend(model))
        for source_id,source in enumerate(plan['sources']):
            root=Path(source['stimuli']);ref=Path(source['response']);sr=read(root/'result.json');sm=read(root/'manifest.json')['config'];rr=read(ref/'result.json');rm=read(ref/'manifest.json')['config'];assert sha256(root/'result.json')==source['stimuli_result_sha256'] and sha256(ref/'result.json')==source['response_result_sha256'] and sha256(root/'features.npz')==sr['features_sha256'] and sha256(ref/'responses.npz')==rr['responses_sha256'] and rm['candidate']==parent and rm['features_sha256']==sr['features_sha256']
            with np.load(root/'features.npz') as z:holder['data']={k:z[k].copy() for k in z.files}
            with np.load(ref/'responses.npz') as z:reference={k:z[k].copy() for k in z.files if k.endswith('_fine_prediction')}
            case_outputs=[]
            for pair in range(0,12,2):
                case=sm['cases'][pair];values,mask,native,valid=synthetic_body(case['speed_mps'],80.,body_port.reference.numpy());holder.update(body=values,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]));model.reset_episode();prefix=[step(pair,i) for i in range(sm['plan']['branch_frame'])];state=snapshot(model)
                for case_id in (pair,pair+1):
                    restore(model,state);sequence=prefix+[step(case_id,i) for i in range(sm['plan']['branch_frame'],sm['plan']['frame_count'])];arrays={name:np.asarray([v[name] for v in sequence]) for name in sequence[0]};difference=float(abs(arrays['prediction']-reference[f'group_{3*(pair//2)+1}_fine_prediction'][case_id-pair]).max());assert difference<=2e-6;case_outputs.append(arrays);reports.append(dict(source=source_id,case=case_id,reference_max_difference=difference));write_json(out/'progress.json',dict(cases=len(reports),total=12*len(plan['sources']),elapsed_seconds=time.perf_counter()-start))
            for name in case_outputs[0]:saved[f'source_{source_id}_{name}']=np.asarray([v[name] for v in case_outputs])
        assert before_hash==[digest(module.state_dict()) for module in [*components[:5],motion]];np.savez_compressed(out/'states.npz',**saved);write_json(out/'cases.json',reports);result.update(status='completed_controlled_flight_stage_recording',sources=len(plan['sources']),cases=len(reports),events=sum(len(saved[f'source_{j}_prediction'])*41 for j in range(len(plan['sources']))),baseline_reference_max_difference=max(v['reference_max_difference'] for v in reports),stage_neuron_counts={k:len(v) for k,v in groups.items()},all_component_state_dict_sha256=before_hash,states_sha256=sha256(out/'states.npz'),neurons_sha256=sha256(out/'neurons.json'),cases_sha256=sha256(out/'cases.json'),termination_reason='Fixed source-bound whole synthetic trajectories with identical body/common neural prefixes. Internal indexed states only observed; no changed circuit, teacher actor input or deployed comparison readout.')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error),completed_cases=len(reports))
    finally:
        for hook in hooks:hook.remove()
        torch.set_flush_denormal(False);gc.enable() if enabled else gc.disable()
    result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
