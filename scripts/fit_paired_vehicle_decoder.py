"""Fit engineering paired decoder; measured graph and cached recurrence unchanged."""
import argparse,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.paired_decoder_data import load_data,normalization,initial_head

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--arm',choices=['monotone','free'],required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=read(a.plan);torch.set_num_threads(1);torch.manual_seed(plan['seed']);assert all(sha256(f)==h for f,h in plan['source_sha256'].items());entries,items,muscles,baseline,pre=load_data(plan);stats=normalization(muscles,items,plan['normalization_floor']);head=initial_head(stats,items,muscles,plan,a.arm=='monotone');initial={k:v.clone() for k,v in head.state_dict().items()};m=torch.tensor(np.concatenate([muscles[e][items[e][0]['fit_mask']] for e in (0,1)]));target=torch.tensor(np.concatenate([items[e][0]['labels'][items[e][0]['fit_mask']] for e in (0,1)]),dtype=torch.float32);premask=np.concatenate([pre[e][items[e][0]['fit_mask']] for e in (0,1)]);weights=torch.tensor(np.tile([4.,1.,1.],(len(m),1)),dtype=torch.float32);weights[torch.tensor(premask)]=torch.tensor([20.,1.,4.]);probes=Path(plan['functional_probes']);pr=read(probes/'result.json');assert sha256(probes/'probes.npz')==pr['raw_probes_sha256']==plan['functional_probes_sha256'];pairs=[]
 with np.load(probes/'probes.npz') as z:
  for case in plan['functional_training_cases']:pairs.append(torch.tensor(z[f'case_{case}_step_0.01_muscles']))
 out=begin_run(a.output,dict(arguments=vars(a),plan=plan,plan_sha256=sha256(a.plan),entries=entries,train_episodes=[0,1],development_used_for_fit=False,training_points=len(m),parameters=sum(p.numel() for p in head.parameters()),scope=__doc__,actor_qualified=False,vehicle_control_sent=False));torch.save(initial,out/'initial-head.pt')
 for f in plan['source_sha256']:
  dest=out/'executed-source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
 optimizer=torch.optim.Adam(head.parameters(),lr=plan['learning_rate']);records=[];best=None;best_loss=float('inf');start=time.perf_counter()
 for step in range(plan['epochs']):
  pred=head(m);data=((pred-target).square()*weights).sum(1).mean();derivatives=[]
  for pair in pairs:
   value=head(pair.reshape(-1,6)).reshape(2,pair.shape[1],3);derivatives.append(((value[1,:,1]-value[1,:,2])-(value[0,:,1]-value[0,:,2])).mean()/.02)
  derivative=torch.stack(derivatives);functional=plan['functional_weight']*torch.relu(plan['minimum_longitudinal_derivative']-derivative).square().mean();loss=data+functional
  if float(loss.detach())<best_loss:best_loss=float(loss.detach());best={k:v.detach().clone() for k,v in head.state_dict().items()};best_step=step
  optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),10.);assert torch.isfinite(norm);optimizer.step();head.project();records.append(dict(step=step,data_loss=float(data.detach()),functional_loss=float(functional.detach()),total=float(loss.detach()),speed_derivatives=derivative.detach().tolist()))
  if step%100==0:write_json(out/'progress.json',dict(epoch=step,total=plan['epochs'],best_train_objective=best_loss,elapsed_seconds=time.perf_counter()-start,termination_reason='Completed fixed-budget decoder-only TRAIN fit; best pre-update TRAIN objective selected, no neural or simulator change'))
 head.load_state_dict(best,strict=True);cp=out/'paired-head.pt';torch.save(dict(schema='flybrain.paired_vehicle_readout.v1',readout=head.state_dict(),monotone=head.monotone,minimum_power_gain=head.minimum_power_gain,maximum_gain=head.maximum_gain,parent_candidate_plan=plan['candidate_plan'],parent_candidate_plan_sha256=plan['candidate_plan_sha256'],actor_qualified=False,vehicle_control_sent=False),cp);predictions={}
 with torch.no_grad():
  for e in range(3):predictions[f'prediction_{e}']=head(torch.tensor(muscles[e])).numpy()
 np.savez_compressed(out/'predictions.npz',**predictions);write_json(out/'training-records.json',records);write_json(out/'result.json',dict(status='completed_paired_vehicle_decoder_fit',arm=a.arm,epochs=plan['epochs'],best_step=best_step,selected_optimizer_updates=best_step,best_train_objective=best_loss,checkpoint_sha256=sha256(cp),predictions_sha256=sha256(out/'predictions.npz'),training_records_sha256=sha256(out/'training-records.json'),power_gains=head.power_gain.detach().tolist(),parent_chemical_and_all_sensory_weights_changed=False,recurrence_replayed=False,cached_recurrence_reuse_reason='Vehicle readout is stateless and not fed into this frozen parent sensory decision; muscle trajectories are original full audited parent runs.',development_used_for_fit=False,actor_qualified=False,vehicle_control_sent=False,elapsed_seconds=time.perf_counter()-start,termination_reason='Completed fixed-budget decoder-only TRAIN fit; best pre-update TRAIN objective selected, no neural or simulator change'))
if __name__=='__main__':main()
