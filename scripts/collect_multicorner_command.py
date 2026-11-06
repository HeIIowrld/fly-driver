"""Independent fixed-candidate episode replay and identical-prefix command branches."""
import argparse,gc,time
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.virtual_navigation_body import NAVIGATION_FEATURE_NAMES
from flybrain_drive.native_multisensory_extension import EXTENSION_NAMES
from scripts.multicorner_command_data import ROOT,experiment,tasks_for_episode
from scripts.sac_steering_experiment import snapshot,restore
from scripts.intervene_motion_spatial_same_body import state_digest

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--episode',type=int,choices=[0,1,2],required=True);a=parser.parse_args();torch.set_num_threads(1);plan=read(ROOT/'protocol-01.json');assert all(sha256(p)==h for p,h in plan['source_sha256'].items());e=experiment(plan);episode=a.episode;d=e.items[episode][0];tasks=tasks_for_episode(d,episode);assert tasks==[v for v in plan['tasks'] if v['episode']==episode];out=begin_run(ROOT/f'episode{episode}-probes-01',dict(protocol=plan,protocol_sha256=sha256(ROOT/'protocol-01.json'),episode=episode,seed=None,parent_candidate=e.parent,source_entry=e.entries[episode],program_sha256=sha256(__file__),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));states={};needed={v['begin'] for v in tasks};capture=[];hook=e.head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().numpy().copy()));prediction=[];saved={};reports=[];enabled=gc.isenabled();gc.disable();start=time.perf_counter();result=dict(status='failed',episode=episode,actor_qualified=False,vehicle_control_sent=False,optimizer_updates=0)
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=episode,calibration_off=False,motion_off=False)
  for i in range(len(d['scene'])):
   if i in needed:states[i]=snapshot(e.model)
   prediction.append(e.step(i,False).numpy().copy())
   if i%1000==0:write_json(out/'progress.json',dict(phase='neutral',events=i,total=len(d['scene']),elapsed_seconds=time.perf_counter()-start))
  full_prediction=np.asarray(prediction);full_mu=np.concatenate(capture)
  with np.load(__import__('pathlib').Path(plan['parent_evaluation'])/'predictions.npz') as z:parity=float(abs(full_prediction-z[f'prediction_{episode}']).max());mu_parity=float(abs(full_mu-z[f'muscles_{episode}']).max())
  assert parity<2e-6 and mu_parity<2e-6;np.savez_compressed(out/'neutral-full.npz',prediction=full_prediction,muscles=full_mu);torch.save(states,out/'candidate-prefix-states.pt')
  for k,task in enumerate(tasks):
   begin,end=task['begin'],task['end'];row=dict(task=task,prefix_sha256=state_digest(states[begin]),body_speed_range_m_s=[float(d['body'][begin:end,16].min()),float(d['body'][begin:end,16].max())],body_yaw_range_rad_s=[float(d['body'][begin:end,4].min()),float(d['body'][begin:end,4].max())],body_pitch_roll_mean_rad=d['body'][begin:end,:2].mean(0).tolist(),native_valid_fraction=e.valid[episode][begin:end].float().mean(0).tolist(),commands={});saved[f'case{k}_neutral_muscles']=full_mu[begin:end];saved[f'case{k}_body']=d['body'][begin:end];saved[f'case{k}_body_valid']=d['body_valid'][begin:end];saved[f'case{k}_native']=e.native[episode][begin:end].numpy();saved[f'case{k}_native_valid']=e.valid[episode][begin:end].numpy()
   restore(e.model,states[begin]);capture.clear();neutral=np.asarray([e.step(i,False).numpy() for i in range(begin,end)]);neutral_error=float(abs(neutral-full_prediction[begin:end]).max());assert neutral_error<2e-6;row['neutral_branch_parity']=neutral_error
   for field,offset in plan['offsets'].items():
    branches=[];muscles=[]
    for sign in [-1,1]:
     restore(e.model,states[begin]);assert state_digest(snapshot(e.model))==row['prefix_sha256'];e.yaw_offset=sign*offset if field=='yaw' else 0.;e.speed_offset=sign*offset if field=='speed' else 0.;capture.clear();branches.append(np.asarray([e.step(i,False).numpy() for i in range(begin,end)]));muscles.append(np.concatenate(capture))
    e.yaw_offset=e.speed_offset=0.;p=np.asarray(branches);saved[f'case{k}_{field}_muscles']=np.asarray(muscles);saved[f'case{k}_{field}_prediction']=p;value=p[:,:,0] if field=='yaw' else p[:,:,1]-p[:,:,2];base=neutral[:,0] if field=='yaw' else neutral[:,1]-neutral[:,2];sign=-1 if field=='yaw' else 1;minus=value[0]-base;plus=value[1]-base;row['commands'][field]=dict(mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),mean_both_directions_correct=bool(sign*minus.mean()<0 and sign*plus.mean()>0))
   reports.append(row);write_json(out/'progress.json',dict(phase='commands',completed_tasks=k+1,total_tasks=len(tasks),elapsed_seconds=time.perf_counter()-start))
  e.verify_fixed();np.savez_compressed(out/'probes.npz',**saved);result.update(status='completed_multi_corner_command_collection',reports=reports,full_events=len(d['scene']),full_prediction_parity=parity,full_muscle_parity=mu_parity,neutral_full_sha256=sha256(out/'neutral-full.npz'),probes_sha256=sha256(out/'probes.npz'),prefix_states_sha256=sha256(out/'candidate-prefix-states.pt'),body_feature_names=list(NAVIGATION_FEATURE_NAMES),native_feature_names=list(EXTENSION_NAMES),termination_reason='One chronological fixed-weight episode and independent neutral/+/- yaw/speed branches completed. Same source body/images and exact candidate prefix across branches, no optimizer or vehicle control.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error),collected_tasks=len(reports))
 finally:hook.remove();e.yaw_offset=e.speed_offset=0.;e.close();gc.enable() if enabled else gc.disable()
 result['elapsed_seconds']=time.perf_counter()-start;write_json(out/'result.json',result);print(result['status'])
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
