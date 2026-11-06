"""Cached TRAIN-only full decoder warmup; no new neural recurrence claimed."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.visual_dn_current_calibration import PowerMonotoneVehicleReadout
from scripts.paired_decoder_data import load_data

def main():
 torch.set_num_threads(1);root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');assert all(sha256(p)==h for p,h in plan['source_sha256'].items());torch.manual_seed(plan['seed']);data_plan=read(plan['paired_data_plan']);_,items,muscles,baseline,pre=load_data(data_plan);state=torch.load(Path(data_plan['parent_fine_fit'])/'motion-spatial.pt',map_location='cpu',weights_only=True)['readout'];head=PowerMonotoneVehicleReadout(state,plan['minimum_power_weight']);head.project();x=torch.tensor(np.concatenate([muscles[e][items[e][0]['fit_mask']] for e in [0,1]]));target=torch.tensor(np.concatenate([items[e][0]['labels'][items[e][0]['fit_mask']] for e in [0,1]]),dtype=torch.float32);mask=np.concatenate([pre[e][items[e][0]['fit_mask']] for e in [0,1]]);weights=torch.tensor(np.tile([4.,1.,1.],(len(x),1)),dtype=torch.float32);weights[torch.tensor(mask)]=torch.tensor([20.,1.,4.])
 with np.load(Path(plan['yaw_probes'])/'probes.npz') as z:yaw=[torch.tensor(z[f'TRAIN_{j}_muscles']) for j in plan['physical_cases']]
 with np.load(Path(data_plan['functional_probes'])/'probes.npz') as z:speed=[torch.tensor(z[f'case_{j}_step_0.01_muscles']) for j in plan['physical_cases']]
 out=begin_run(root/'warm-fit-01',dict(plan=plan,plan_sha256=sha256(root/'fit-plan-01.json'),training_points=len(x),development_used_for_fit=False,neural_optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));opt=torch.optim.Adam(head.parameters(),lr=plan['warm_learning_rate']);bestloss=float('inf');trace=[]
 for step in range(plan['warm_epochs']):
  pred=head(x);data=((pred-target).square()*weights).sum(1).mean();ys=[];ss=[]
  for mu in yaw:
   p=head(mu.reshape(-1,6)).reshape(2,-1,3);ys.append((p[1,:,0]-p[0,:,0]).mean()/.02)
  for mu in speed:
   p=head(mu.reshape(-1,6)).reshape(2,-1,3);ss.append(((p[1,:,1]-p[1,:,2])-(p[0,:,1]-p[0,:,2])).mean()/.02)
  y=torch.stack(ys);s=torch.stack(ss);loss=data+plan['physical_weight']*(torch.relu(y+plan['minimum_physical_derivative']).square().mean()+torch.relu(plan['minimum_physical_derivative']-s).square().mean())
  if float(loss.detach())<bestloss:bestloss=float(loss.detach());best={k:v.detach().clone() for k,v in head.state_dict().items()};beststep=step
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),10.);opt.step();head.project();trace.append(dict(step=step,total=float(loss.detach()),data=float(data.detach()),yaw=y.detach().tolist(),speed=s.detach().tolist()))
 head.load_state_dict(best);torch.save(dict(schema='flybrain.full_monotone_decoder.warm.v1',readout=best,actor_qualified=False),out/'warm-head.pt');np.savez_compressed(out/'predictions.npz',**{f'prediction_{e}':head(torch.tensor(muscles[e])).detach().numpy() for e in range(3)});write_json(out/'training-records.json',trace);write_json(out/'result.json',dict(status='completed_full_monotone_decoder_warmup',selected_step=beststep,best_train_objective=bestloss,checkpoint_sha256=sha256(out/'warm-head.pt'),predictions_sha256=sha256(out/'predictions.npz'),neural_optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,termination_reason='Fixed-budget cached TRAIN warmup, best TRAIN total objective selected'));print(out)
if __name__=='__main__':main()
