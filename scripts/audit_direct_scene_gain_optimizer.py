"""Replay all Adam proposals and cumulative bounds independently of model training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/direct-scene-gain-learning-20261105')

def parameters(cp):return [cp['current_calibration']['motion_projection.weight'],cp['body_port']['projection.weight'],cp['speed_policy']['projection.weight'],cp['speed_policy']['projection.bias'],cp['direct_gain']['delta']]

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['frozen_direct','learn_direct'],required=True);a=p.parse_args();torch.set_num_threads(1);fit=ROOT/f'{a.arm}-fit-01';r=read(fit/'result.json');plan=read(ROOT/'plan-01.json');assert r['status']=='completed_direct_scene_gain_fit'
 for f,k in [('ports.pt','checkpoint_sha256'),('optimizer-state.pt','optimizer_checkpoint_sha256'),('optimizer-trajectory.npz','optimizer_trajectory_sha256'),('training-records.json','training_records_sha256')]:assert sha256(fit/f)==r[k]
 cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);parent=torch.load(plan['direct_parent_checkpoint'],map_location='cpu',weights_only=True);records=read(fit/'training-records.json');ps=[torch.nn.Parameter(v.clone()) for v in parameters(cp)];rates=[plan[k] for k in ['visual_learning_rate','native_learning_rate','speed_weight_learning_rate','speed_bias_learning_rate','direct_gain_learning_rate']];optimizer=torch.optim.Adam([dict(params=[v],lr=lr) for v,lr in zip(ps[:4],rates[:4])]);optimizer.load_state_dict(torch.load(plan['direct_parent_optimizer'],map_location='cpu',weights_only=True));optimizer.add_param_group(dict(params=[ps[4]],lr=rates[4]));sizes=[v.numel() for v in ps];cuts=np.cumsum([0]+sizes);maximum=0.;max_steps=np.zeros(5);native_origin=torch.load(plan['motion_parent_checkpoint'],map_location='cpu',weights_only=True)['body_port']['projection.weight'].numpy();reference=[parent['current_calibration']['motion_projection.weight'],parent['body_port']['projection.weight'],parent['speed_policy']['projection.weight'],parent['speed_policy']['projection.bias'],torch.zeros_like(ps[4])];out=begin_run(ROOT/f'{a.arm}-optimizer-audit-01',dict(program_sha256=sha256(__file__),checkpoint_sha256=r['checkpoint_sha256'],trajectory_sha256=r['optimizer_trajectory_sha256'],optimizer_updates=0,vehicle_control_sent=False))
 with np.load(fit/'optimizer-trajectory.npz') as z:before=z['before'];proposal=z['proposal'];after=z['after'];gradients=z['gradients']
 np.testing.assert_array_equal(before[0],torch.cat([v.reshape(-1) for v in reference]).numpy());assert len(records)==len(before)==395
 for j,row in enumerate(records):
  for k,v in enumerate(ps):
   with torch.no_grad():v.copy_(torch.from_numpy(before[j,cuts[k]:cuts[k+1]].reshape(v.shape)))
   v.grad=None if row['parameter_grad_is_none'][k] else torch.from_numpy(gradients[j,cuts[k]:cuts[k+1]].reshape(v.shape).copy())
  optimizer.step();actual=torch.cat([v.detach().reshape(-1) for v in ps]).numpy();maximum=max(maximum,float(abs(actual-proposal[j]).max()));steps=[np.linalg.norm(after[j,cuts[k]:cuts[k+1]]-before[j,cuts[k]:cuts[k+1]]) for k in range(5)];max_steps=np.maximum(max_steps,steps)
  w,n,sw,sb,g=[after[j,cuts[k]:cuts[k+1]] for k in range(5)];delta=n-native_origin.reshape(-1)
  assert np.linalg.norm(w)<=plan['visual_total_norm']+1e-6 and abs(w).max()<=2+1e-7 and np.linalg.norm(delta)<=plan['native_total_norm']+1e-6 and abs(delta).max()<=plan['native_entry_delta']+1e-7
  assert np.linalg.norm(sw)<=plan['speed_weight_total_norm']+1e-7 and abs(sw).max()<=plan['speed_weight_absolute_bound']+1e-7 and abs(sb).max()<=plan['speed_bias_absolute_bound']+1e-7 and sw[-1]<=0
  scale=cp['speed_policy']['scale'].numpy();sensitivity=plan['speed_max_delta_m_s']*(abs(sw[2])/scale[2]+abs(sw[5])/(2*scale[5]));assert sensitivity<=plan['maximum_power_input_sensitivity']+1e-3
  assert np.linalg.norm(g)<=.5+1e-7 and abs(g).max()<=.25+1e-7
  if a.arm=='frozen_direct':assert row['parameter_grad_is_none'][4] and np.count_nonzero(g)==0
  if row['kind'] in ['synthetic','image_exchange']:assert all(row['parameter_grad_is_none'][k] for k in [1,2,3]) and all(np.array_equal(before[j,cuts[k]:cuts[k+1]],after[j,cuts[k]:cuts[k+1]]) for k in [1,2,3])
 np.testing.assert_array_equal(after[-1],torch.cat([v.reshape(-1) for v in parameters(cp)]).numpy());assert maximum==0
 final=torch.load(fit/'optimizer-state.pt',map_location='cpu',weights_only=True);recovered=optimizer.state_dict();assert final['param_groups']==recovered['param_groups'] and final['state'].keys()==recovered['state'].keys()
 for key,state in final['state'].items():
  for name,value in state.items():assert torch.equal(value,recovered['state'][key][name])
 assert all(max_steps[k]<=plan[key]+1e-6 for k,key in enumerate(['visual_step_norm','native_step_norm','speed_weight_step_norm','speed_bias_step_norm','direct_gain_step_norm']))
 write_json(out/'result.json',dict(status='completed_independent_direct_scene_gain_adam_audit',updates=len(records),maximum_proposal_difference=maximum,all_final_adam_moments_exact=True,cumulative_bounds_verified=True,fixture_native_and_speed_updates_zero=True,maximum_parameter_step_norms=max_steps.tolist(),final_optimizer_steps=r['final_optimizer_steps'],vehicle_control_sent=False,actual_driving_improvement_verified=False))

if __name__=='__main__':main()
