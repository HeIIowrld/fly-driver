"""Cached readout evaluation and independent reflection/power/command constraints."""
import argparse
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.paired_vehicle_candidate import load_paired_head
from scripts.paired_decoder_data import load_data,normalization
from scripts.improve_sac_anticipation import command_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',choices=['monotone','free'],required=True);p.add_argument('--stimulus-response',required=True);a=p.parse_args();torch.set_num_threads(1);root=Path(a.root);fit=root/f'{a.arm}-fit-01';cp=fit/'paired-head.pt';head,receipt=load_paired_head(cp);m=read(fit/'manifest.json')['config'];plan=m['plan'];entries,items,muscles,baseline,pre=load_data(plan);stats=normalization(muscles,items,plan['normalization_floor'])
 for key,value in stats.items():torch.testing.assert_close(head.state_dict()[key],value,rtol=0,atol=0)
 original=torch.load(Path(plan['parent_fine_fit'])/'motion-spatial.pt',map_location='cpu',weights_only=True)['readout'];actual=[];saved={};maximum_redecode=0.
 with np.load(fit/'predictions.npz') as z:
  for e in range(3):
   prediction=head(torch.tensor(muscles[e])).numpy();maximum_redecode=max(maximum_redecode,float(abs(prediction-z[f'prediction_{e}']).max()));d=items[e][0];metrics={role:dict(parent=command_metrics(baseline[e],d['labels'],mask),paired=command_metrics(prediction,d['labels'],mask)) for role,mask in [('all',d['fit_mask']),('pre',pre[e])]};actual.append(dict(episode=e,events=len(prediction),metrics=metrics));saved[f'prediction_{e}']=prediction
 assert maximum_redecode==0.
 probes=[]
 for role,path in [('TRAIN',Path(plan['functional_probes'])),('development',root/'development-probes-01')]:
  result=read(path/'result.json');expected=result['raw_probes_sha256'];assert sha256(path/'probes.npz')==expected;tasks=[v['task'] for v in result['reports']] if role=='TRAIN' else result['tasks']
  with np.load(path/'probes.npz') as z:
   for j,task in enumerate(tasks):
    mu=z[f'case_{j}_step_0.01_muscles'];prediction=head(torch.tensor(mu.reshape(-1,6))).numpy().reshape(2,mu.shape[1],3);derivative=(prediction[1].astype(float)-prediction[0])/.02;probes.append(dict(role=role,case=j,used_in_functional_fit=role=='TRAIN' and j in plan['functional_training_cases'],task=task,vehicle_derivative=derivative.mean(0).tolist(),longitudinal_derivative=float(np.mean(derivative[:,1]-derivative[:,2]))));saved[f'{role}_probe_{j}_prediction']=prediction
 source=Path(a.stimulus_response);sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];assert sr['status']=='completed_controlled_motion_factorial_response' and sha256(source/'responses.npz')==sr['responses_sha256'] and sm['candidate']==plan['parent_candidate'];groups=read(source/'groups.json');counts={k:0 for k in ('fine','motion_off_after_prefix','speed_off_after_prefix')};target_count=0
 with np.load(source/'responses.npz') as z:
  for g in groups:
   j=g['group']
   for arm in counts:
    mu=z[f'group_{j}_{arm}_muscles'];prediction=head(torch.tensor(mu.reshape(-1,6))).numpy().reshape(2,mu.shape[1],3);saved[f'group_{j}_{arm}_prediction']=prediction;late=prediction[:,25:,0];margin=float(-np.mean(late[1]-late[0])/2);counts[arm]+=int(margin>.0001);target_count+=int(arm=='fine' and margin>=.002)
 x=torch.tensor(muscles[0][items[0][0]['fit_mask']][:1000]);y=head(x);reflection=head(x[:,[1,0,3,2,5,4]]);odd=float(abs(y[:,0]+reflection[:,0]).max());even=float(abs(y[:,1:]-reflection[:,1:]).max());common=x.clone();common[:,4:]+=.05*float(head.power_scale);shift=head(common);cross=float(abs(y[:,0]-shift[:,0]).max());gas=float((shift[:,1]-y[:,1]).min());brake=float((shift[:,2]-y[:,2]).max());assert odd<2e-6 and even<2e-6 and cross<2e-4
 out=begin_run(root/f'evaluation-{a.arm}-01',dict(program_sha256=sha256(__file__),fit_result_sha256=sha256(fit/'result.json'),checkpoint_sha256=sha256(cp),stimulus_response_result_sha256=sha256(source/'result.json'),receipt=receipt,actor_qualified=False,vehicle_control_sent=False));np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_paired_decoder_cached_evaluation',actual_recordings=actual,command_probes=probes,reserved_direction_counts=counts,reserved_target_pairs=target_count,reflection_steering_error=odd,reflection_pedal_error=even,common_power_steering_error=cross,minimum_common_power_gas_change=gas,maximum_common_power_brake_change=brake,pure_common_power_gate=bool((head.power_gain.detach()>0).all()) and gas>=-1e-7 and brake<=1e-7,command_speed_gate=all(v['longitudinal_derivative']>0 for v in probes),predictions_sha256=sha256(out/'predictions.npz'),actual_driving_improvement_verified=False,actor_qualified=False,vehicle_control_sent=False,scope='Readout-only counterfactual on original fixed full neural trajectories; no new recurrence asserted. Development probes are fresh queries in consumed same-course data, not new track or control.'))
if __name__=='__main__':main()
