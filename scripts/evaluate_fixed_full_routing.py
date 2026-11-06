"""Fixed full recurrence, 45 command windows and held-out real/synthetic image exchange."""
import argparse,gc,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from scripts.fixed_full_routing_experiment import FixedFullRoutingExperiment as VisualNativePortExperiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.evaluate_controlled_motion_factorial import synthetic_body
from scripts.improve_sac_anticipation import command_metrics

ROOT=Path('runs/fixed-full-routing-learning-20261104')
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['visual_only','visual_native'],required=True);a=p.parse_args();torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');fit=ROOT/f'{a.arm}-fit-01';assert read(fit/'result.json')['status']=='completed_fixed_full_routing_fit' and sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e=VisualNativePortExperiment(plan,train_body=a.arm=='visual_native');e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.calibration.requires_grad_(False);e.body_port.requires_grad_(False);e.verify_fixed();mc=read('runs/multicorner-command-20261104/protocol-01.json');tasks=mc['tasks'];needed={(v['episode'],v['begin']) for v in tasks};prefixes={};saved={};metrics=[];reports=[];counts={k:dict(total=0,yaw=0,speed=0) for k in ['TRAIN','development']};out=begin_run(ROOT/f'{a.arm}-evaluation-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(ROOT/'plan-01.json'),checkpoint_sha256=sha256(fit/'ports.pt'),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));start=time.perf_counter();enabled=gc.isenabled();gc.disable();result=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
 try:
  original='runs/motion-spatial-learning-20261104/fine_motion-fit-01/predictions.npz'
  for episode in range(3):
   e.model.reset_episode();e.holder.clear();e.holder.update(mode='real',episode=episode);d=e.items[episode][0];pred=[];mu=[]
   for i in range(len(d['scene'])):
    if (episode,i) in needed:prefixes[episode,i]=snapshot(e.model)
    pred.append(e.step(i,False).numpy());mu.append(e.last_muscles[0].numpy())
    if i%1000==0:write_json(out/'progress.json',dict(phase='recording',episode=episode,event=i,elapsed_seconds=time.perf_counter()-start))
   prediction=np.asarray(pred);muscles=np.asarray(mu);saved[f'prediction_{episode}']=prediction;saved[f'muscles_{episode}']=muscles
   with torch.no_grad():saved[f'logit_{episode}']=e.head.steering(e.head.features(torch.tensor(muscles))).numpy()[:,0]
   with np.load(original) as z:baseline=z[f'prediction_{episode}'].copy()
   with np.load('runs/multicorner-command-20261104/affine-evaluation-01/predictions.npz') as z:initial=z[f'prediction_{episode}'].copy()
   metrics.append(dict(episode=episode,events=len(prediction),metrics={role:dict(original=command_metrics(baseline,d['labels'],mask),initial=command_metrics(initial,d['labels'],mask),new=command_metrics(prediction,d['labels'],mask)) for role,mask in [('all',d['fit_mask']),('pre',e.pre[episode])]}))
  torch.save(prefixes,out/'candidate-prefix-states.pt')
  for j,task in enumerate(tasks):
   episode,begin,end=task['episode'],task['begin'],task['end'];e.holder.clear();e.holder.update(mode='real',episode=episode);neutral=saved[f'prediction_{episode}'][begin:end];logit0=saved[f'logit_{episode}'][begin:end];row=dict(task=task,prefix_sha256=state_digest(prefixes[episode,begin]),commands={});role='TRAIN' if episode<2 else 'development';counts[role]['total']+=1
   for field in ['yaw','speed']:
    branches=[];logits=[]
    for sign in [-1,1]:
     restore(e.model,prefixes[episode,begin]);e.yaw_offset=sign*.01 if field=='yaw' else 0.;e.speed_offset=sign*.01 if field=='speed' else 0.;sequence=[];ls=[]
     for i in range(begin,end):
      sequence.append(e.step(i,False).numpy())
      with torch.no_grad():ls.append(float(e.head.steering(e.head.features(e.last_muscles))[0,0]))
     branches.append(sequence);logits.append(ls)
    e.yaw_offset=e.speed_offset=0.;p=np.asarray(branches);saved[f'case{j}_{field}_prediction']=p;value=p[:,:,0] if field=='yaw' else p[:,:,1]-p[:,:,2];base=neutral[:,0] if field=='yaw' else neutral[:,1]-neutral[:,2];sign=-1 if field=='yaw' else 1;minus=value[0]-base;plus=value[1]-base;correct=bool(sign*minus.mean()<0 and sign*plus.mean()>0);counts[role][field]+=int(correct);row['commands'][field]=dict(mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),correct_minus_fraction=float((sign*minus< -1e-7).mean()),correct_plus_fraction=float((sign*plus>1e-7).mean()),mean_both_correct=correct)
    if field=='yaw':row['commands'][field]['mean_logit_minus_plus']=((np.asarray(logits)-logit0).mean(1)/.01).tolist()
   reports.append(row)
  source=ROOT/'reserved-stimuli-01';sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];sp=sm['plan'];assert sha256(source/'features.npz')==sr['features_sha256'] and sp['texture_seed']==plan['reserved_seed']
  with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
  item=dict(x=x,metadata=sm,plan=sp);groups=[];directions={k:0 for k in ['fine','visual_change_off','native_change_off','motion_off']};target=0;native_negative_control=0.
  for k in range(0,12,2):
   case=sm['cases'][k]
   for temperature in sp['temperature_c']:
    body,mask,native,valid=synthetic_body(case['speed_mps'],temperature,e.body_port.reference.numpy());e.holder.clear();e.holder.update(mode='synthetic',source=item,case=k,body=body,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]));e.model.reset_episode()
    for i in range(sp['branch_frame']):e.step(i,False)
    prefix=snapshot(e.model);g=len(groups);row=dict(group=g,pair=k,temperature_c=temperature,prefix_sha256=state_digest(prefix),arms={});values={}
    for arm in directions:
     branches=[]
     for case_id in [k,k+1]:
      restore(e.model,prefix);e.holder.update(case=case_id,visual_change_off=arm=='visual_change_off',native_change_off=arm=='native_change_off',motion_off=arm=='motion_off');branches.append(np.asarray([e.step(i,False).numpy() for i in range(sp['branch_frame'],sp['frame_count'])]))
     p=np.asarray(branches);values[arm]=p;saved[f'group{g}_{arm}_prediction']=p;late=sp['evaluation_first_frame']-sp['branch_frame'];margin=float(-np.mean(p[1,late:,0]-p[0,late:,0])/2);directions[arm]+=int(margin>.0001);target+=int(arm=='fine' and margin>=.002);row['arms'][arm]=dict(signed_margin=margin)
    native_negative_control=max(native_negative_control,float(abs(values['fine']-values['native_change_off']).max()));groups.append(row);write_json(out/'progress.json',dict(phase='reserved',groups=len(groups),elapsed_seconds=time.perf_counter()-start))
  image_groups=[];donors=[v for v in tasks if v['episode']==2 and v['phase']=='early'];lefts=[v for v in donors if v['onset']['direction']<0];rights=[v for v in donors if v['onset']['direction']>0]
  for recipient in donors:
   begin,end=recipient['begin'],recipient['end'];prefix=prefixes[2,begin]
   for left in lefts:
    for right in rights:
     row=dict(group=len(image_groups),recipient=recipient,left_donor=left,right_donor=right,prefix_sha256=state_digest(prefix),arms={})
     for arm in ['fine','visual_change_off','native_change_off']:
      branches=[]
      for donor in [left,right]:
       restore(e.model,prefix);e.holder.clear();e.holder.update(mode='real',episode=2,counterfactual_images=True,visual_change_off=arm=='visual_change_off',native_change_off=arm=='native_change_off');sequence=[]
       for offset,i in enumerate(range(begin,end)):e.holder['donor']=(2,donor['begin']+offset);sequence.append(e.step(i,False).numpy())
       branches.append(sequence)
      p=np.asarray(branches);saved[f'image{row["group"]}_{arm}_prediction']=p;row['arms'][arm]=dict(signed_teacher_bend_margin=float(-np.mean(p[0,:,0]-p[1,:,0])/2))
     image_groups.append(row)
   write_json(out/'progress.json',dict(phase='real_images',groups=len(image_groups),elapsed_seconds=time.perf_counter()-start))
  assert len(image_groups)==30
  if a.arm=='visual_only':assert native_negative_control==0
  e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'reserved-groups.json',groups);write_json(out/'real-image-groups.json',image_groups);result.update(status='completed_fixed_full_routing_evaluation',actual_recordings=metrics,command_reports=reports,command_counts=counts,reserved_direction_counts=directions,reserved_target_pairs=target,native_change_control_max_difference=native_negative_control,real_image_direction_counts={arm:sum(v['arms'][arm]['signed_teacher_bend_margin']>0 for v in image_groups) for arm in ['fine','visual_change_off','native_change_off']},real_image_groups=30,predictions_sha256=sha256(out/'predictions.npz'),prefix_states_sha256=sha256(out/'candidate-prefix-states.pt'),reserved_groups_sha256=sha256(out/'reserved-groups.json'),real_image_groups_sha256=sha256(out/'real-image-groups.json'),actual_driving_improvement_verified=False,termination_reason='All3 actual fixed-weight trajectories, 45 candidate-prefix command windows, one fresh synthetic texture and 30 consumed-development retimed real-image exchanges completed. No selection/control or tyre-adaptation qualification.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 finally:e.yaw_offset=e.speed_offset=0.;e.close();gc.enable() if enabled else gc.disable()
 result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(result['status'])
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
