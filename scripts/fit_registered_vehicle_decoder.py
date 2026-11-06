"""TRAIN-only affine steering registration with explicit yaw/speed sign supervision.

Steering bias is an engineering offset, not measured bilateral neutral force.
Raw reflection is deliberately not asserted. Common power steering independence
and positive individual power longitudinal signs are retained.
"""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.paired_vehicle_candidate import load_paired_head
from scripts.paired_decoder_data import load_data
from scripts.improve_sac_anticipation import command_metrics

def main():
 torch.set_num_threads(1);torch.manual_seed(781177);root=Path('runs/paired-muscle-decoder-20261104');plan=read(root/'plan-01.json');entries,items,muscles,baseline,pre=load_data(plan);head,_=load_paired_head(root/'monotone-fit-01/paired-head.pt');head.requires_grad_(True);bias=torch.nn.Parameter(torch.zeros(()));params=list(head.parameters())+[bias]
 def decode(x):
  d,_,_=head.features(x);return torch.cat([(d@head.steering_weight+bias).tanh()[:,None],head(x)[:,1:]],1)
 train=torch.tensor(np.concatenate([muscles[e][items[e][0]['fit_mask']] for e in [0,1]]));target=torch.tensor(np.concatenate([items[e][0]['labels'][items[e][0]['fit_mask']] for e in [0,1]]),dtype=torch.float32);mask=np.concatenate([pre[e][items[e][0]['fit_mask']] for e in [0,1]]);weights=torch.tensor(np.tile([4.,1.,1.],(len(train),1)),dtype=torch.float32);weights[torch.tensor(mask)]=torch.tensor([20.,1.,4.])
 yawpath=root/'yaw-probes-01/probes.npz';speedpath=Path(plan['functional_probes'])/'probes.npz';assert sha256(yawpath)==read(root/'yaw-probes-01/result.json')['raw_probes_sha256'];assert sha256(speedpath)==plan['functional_probes_sha256']
 with np.load(yawpath) as z:yaw=[torch.tensor(z[f'TRAIN_{j}_muscles']) for j in [0,2]]
 with np.load(speedpath) as z:speed=[torch.tensor(z[f'case_{j}_step_0.01_muscles']) for j in [0,2]]
 out=begin_run(root/'registered-fit-01',dict(program_sha256=sha256(__file__),seed=781177,parent_checkpoint_sha256=sha256(root/'monotone-fit-01/paired-head.pt'),yaw_probes_sha256=sha256(yawpath),speed_probes_sha256=sha256(speedpath),epochs=1600,learning_rate=.005,parameters=16,yaw_training_cases=[0,2],speed_training_cases=[0,2],minimum_yaw_derivative_magnitude=.005,minimum_speed_derivative=.005,functional_weight=100.,development_used_for_fit=False,selection='minimum TRAIN objective only',scope=__doc__,actor_qualified=False,vehicle_control_sent=False));(out/'executed-source.py').write_bytes(Path(__file__).read_bytes());opt=torch.optim.Adam(params,lr=.005);bestloss=float('inf');trace=[]
 for step in range(1600):
  pred=decode(train);data=((pred-target).square()*weights).sum(1).mean();yd=[];sd=[]
  for mu in yaw:
   p=decode(mu.reshape(-1,6)).reshape(2,-1,3);yd.append((p[1,:,0]-p[0,:,0]).mean()/.02)
  for mu in speed:
   p=decode(mu.reshape(-1,6)).reshape(2,-1,3);sd.append(((p[1,:,1]-p[1,:,2])-(p[0,:,1]-p[0,:,2])).mean()/.02)
  y=torch.stack(yd);s=torch.stack(sd);loss=data+100*(torch.relu(y+.005).square().mean()+torch.relu(.005-s).square().mean())
  if float(loss.detach())<bestloss:bestloss=float(loss.detach());best={k:v.detach().clone() for k,v in head.state_dict().items()};bestbias=bias.detach().clone();beststep=step
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,10.);opt.step();head.project()
  with torch.no_grad():bias.clamp_(-10,10)
  trace.append(dict(step=step,total=float(loss.detach()),data=float(data.detach()),yaw=y.detach().tolist(),speed=s.detach().tolist()))
 head.load_state_dict(best);bias.data.copy_(bestbias);torch.save(dict(schema='flybrain.registered_vehicle_readout.experimental.v1',readout=best,steering_bias=bestbias,actor_qualified=False),out/'registered-head.pt');saved={};reports=[]
 with torch.no_grad():
  for e in range(3):
   p=decode(torch.tensor(muscles[e])).numpy();saved[f'prediction_{e}']=p;d=items[e][0];reports.append(dict(episode=e,metrics={role:dict(parent=command_metrics(baseline[e],d['labels'],m),registered=command_metrics(p,d['labels'],m)) for role,m in [('all',d['fit_mask']),('pre',pre[e])]}))
  yr=read(root/'yaw-probes-01/result.json');probes=[]
  with np.load(yawpath) as z:
   for case in yr['cases']:
    key=f"{case['role']}_{case['case']}";mu=z[key+'_muscles'];p=decode(torch.tensor(mu.reshape(-1,6))).numpy().reshape(2,-1,3);saved[key+'_yaw_prediction']=p;probes.append(dict(role=case['role'],case=case['case'],mean_steering_derivative=float(((p[1,:,0]-p[0,:,0])/.02).mean())))
 np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'training-records.json',trace);write_json(out/'result.json',dict(status='completed_experimental_affine_yaw_registered_fit',selected_step=beststep,best_train_objective=bestloss,steering_bias=float(bestbias),actual_recordings=reports,yaw_probes=probes,checkpoint_sha256=sha256(out/'registered-head.pt'),predictions_sha256=sha256(out/'predictions.npz'),raw_reflection_asserted=False,actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False,termination_reason='Completed fixed TRAIN budget; minimum TRAIN objective selected'))
 print(out)
if __name__=='__main__':main()
