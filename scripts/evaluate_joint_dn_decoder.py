"""Full held-weight replay, seven same-state command probes and reserved vision."""
import argparse,gc,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from scripts.joint_dn_decoder_experiment import JointDNDecoderExperiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.evaluate_controlled_motion_factorial import synthetic_body
from scripts.improve_sac_anticipation import command_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=[1,4],required=True);a=p.parse_args();torch.set_num_threads(1);root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');fit=root/f'rank{a.rank}-fit-01';fr=read(fit/'result.json');cp=fit/'joint-dn-decoder.pt';assert fr['status']=='completed_joint_dn_decoder_fit' and sha256(cp)==fr['checkpoint_sha256'];state=torch.load(cp,map_location='cpu',weights_only=True);assert state['rank']==a.rank and state['actor_qualified'] is False;e=JointDNDecoderExperiment(plan,a.rank,state['readout']);e.calibration.load_state_dict(state['current_calibration'],strict=True);e.head.requires_grad_(False);e.calibration.requires_grad_(False);e.verify_fixed();tasks=read(Path(plan['yaw_probes'])/'result.json')['cases'];needed={(v['task']['episode'],v['task']['begin']) for v in tasks};prefixes={};saved={};metrics=[];probes=[];groups=[];counts={k:0 for k in ['fine','calibration_off_after_prefix','motion_off_after_prefix']};targetpairs=0;out=begin_run(root/f'rank{a.rank}-evaluation-01',dict(plan=plan,checkpoint_sha256=sha256(cp),rank=a.rank,parent_candidate=e.parent,development_used_for_selection=False,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));start=time.perf_counter();enabled=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False);capture=[];hook=e.head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().numpy().copy()))
 try:
  for episode in range(3):
   e.model.reset_episode();e.holder.update(mode='real',episode=episode,calibration_off=False,motion_off=False);d=e.items[episode][0];pred=[];capture.clear()
   for i in range(len(d['scene'])):
    if (episode,i) in needed:prefixes[episode,i]=snapshot(e.model)
    pred.append(e.step(i,False).numpy().copy())
    if i%1000==0:write_json(out/'progress.json',dict(phase='recording',episode=episode,event=i,elapsed_seconds=time.perf_counter()-start))
   prediction=np.asarray(pred);saved[f'prediction_{episode}']=prediction;saved[f'muscles_{episode}']=np.concatenate(capture);baseline=e.anchors[episode] if episode in e.anchors else None
   if baseline is None:
    with np.load(Path(read(plan['paired_data_plan'])['parent_fine_fit'])/'predictions.npz') as z:baseline=z[f'prediction_{episode}'].copy()
   metrics.append(dict(episode=episode,events=len(prediction),metrics={role:dict(parent=command_metrics(baseline,d['labels'],mask),joint=command_metrics(prediction,d['labels'],mask)) for role,mask in [('all',d['fit_mask']),('pre',e.pre[episode])]}))
  hook.remove();torch.save(prefixes,out/'candidate-prefix-states.pt')
  for case in tasks:
   t=case['task'];episode,begin,end=t['episode'],t['begin'],t['end'];e.holder.update(mode='real',episode=episode,calibration_off=False,motion_off=False);row=dict(role=case['role'],case=case['case'],task=t,prefix_sha256=state_digest(prefixes[episode,begin]),models={})
   for field in ['yaw','speed']:
    pair=[]
    for sign in [-1,1]:
     restore(e.model,prefixes[episode,begin]);e.yaw_offset=sign*.01 if field=='yaw' else 0.;e.speed_offset=sign*.01 if field=='speed' else 0.;pair.append(torch.stack([e.step(i,False) for i in range(begin,end)]).numpy())
    e.yaw_offset=e.speed_offset=0.;p=np.asarray(pair);saved[f"{case['role']}_{case['case']}_{field}_prediction"]=p;derivative=(p[1].astype(float)-p[0])/.02;row['models'][field]=dict(vehicle_derivative=derivative.mean(0).tolist(),longitudinal_derivative=float((derivative[:,1]-derivative[:,2]).mean()))
   probes.append(row)
  source=root/'reserved-stimuli-01';sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];sp=sm['plan'];assert sha256(source/'features.npz')==sr['features_sha256'] and sha256(root/'reserved-plan-01.json')==plan['reserved_test_plan_sha256'] and sp['texture_seed']==781183
  with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
  source_item=dict(x=x,metadata=sm,plan=sp)
  for k in range(0,12,2):
   case=sm['cases'][k]
   for temperature in sp['temperature_c']:
    body,mask,native,valid=synthetic_body(case['speed_mps'],temperature,e.body_port.reference.numpy());e.holder.update(mode='synthetic',source=source_item,case=k,body=body,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]),calibration_off=False,motion_off=False);e.model.reset_episode()
    for i in range(sp['branch_frame']):e.step(i,False)
    prefix=snapshot(e.model);row=dict(group=len(groups),pair=k,temperature_c=temperature,prefix_sha256=state_digest(prefix),arms={})
    for arm in counts:
     branch=[]
     for case_id in [k,k+1]:
      restore(e.model,prefix);e.holder.update(case=case_id,calibration_off=arm=='calibration_off_after_prefix',motion_off=arm=='motion_off_after_prefix');branch.append(np.asarray([e.step(i,False).numpy() for i in range(sp['branch_frame'],sp['frame_count'])]))
     prediction=np.asarray(branch);saved[f"group_{row['group']}_{arm}_prediction"]=prediction;late=sp['evaluation_first_frame']-sp['branch_frame'];margin=float(-np.mean(prediction[1,late:,0]-prediction[0,late:,0])/2);counts[arm]+=int(margin>sp['bend_ordering_minimum_margin']);targetpairs+=int(arm=='fine' and margin>=plan['target_margin']);row['arms'][arm]=dict(signed_margin=margin)
    groups.append(row);write_json(out/'progress.json',dict(phase='reserved',groups=len(groups),elapsed_seconds=time.perf_counter()-start))
  e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'groups.json',groups);result.update(status='completed_joint_dn_decoder_evaluation',actual_recordings=metrics,command_probes=probes,yaw_direction_gate=all(v['models']['yaw']['vehicle_derivative'][0]<0 for v in probes),speed_direction_gate=all(v['models']['speed']['longitudinal_derivative']>0 for v in probes),reserved_direction_counts=counts,reserved_target_pairs=targetpairs,predictions_sha256=sha256(out/'predictions.npz'),prefix_states_sha256=sha256(out/'candidate-prefix-states.pt'),groups_sha256=sha256(out/'groups.json'),actual_driving_improvement_verified=False,termination_reason='Full original three recordings, candidate-owned command prefixes and one reserved texture completed. Offline same-course and synthetic evidence only.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 finally:hook.remove();e.yaw_offset=e.speed_offset=0.;e.close();gc.enable() if enabled else gc.disable()
 result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(result['status'])
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
