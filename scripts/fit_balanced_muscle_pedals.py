"""TRAIN-only constrained optimization; genuine labels, no actuator postprocessing."""
import argparse,time
from pathlib import Path
import numpy as np,torch
from scipy.optimize import minimize
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from flybrain_drive.learned_flight_pedal_readout import LearnedFlightPedalReadout
from flybrain_drive.balance_muscle_pedal_readout import BalanceMusclePedalReadout
from scripts.fit_current_flight_launch_pedals import datasets,metrics,SPLIT
ROOT=Path('runs/lease-balanced-learning-20261105')

def main():
 p=argparse.ArgumentParser();p.add_argument('--actor-kind',choices=['parent','dual_guard'],required=True);p.add_argument('--variant',choices=['affine','balance_rbf'],required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(784105);plan=read('runs/dual-reference-learning-20261105/plan-01.json');evaluation=Path(plan['direct_parent_evaluation']) if a.actor_kind=='parent' else Path('runs/dual-reference-learning-20261105/dual_guard-evaluation-01');items=datasets(evaluation);parent,_=load_muscle_interaction_head(plan['fixed_decoder']);checkpoint='runs/fixture-adjoint-learning-20261104/input_adjoint-fit-01/ports.pt' if a.actor_kind=='parent' else 'runs/dual-reference-learning-20261105/dual_guard-fit-01/ports.pt'
 mu=torch.tensor(np.concatenate([v['muscles'] for v in items[:2]]));target=torch.tensor(np.concatenate([v['data']['labels'] for v in items[:2]]),dtype=torch.float64)[:,1:];anchor=torch.tensor(np.concatenate([v['original'] for v in items[:2]]),dtype=torch.float64)[:,1:];low=torch.tensor(np.concatenate([v['low'] for v in items[:2]]));z=((mu-parent.base.training_reference)/parent.base.training_scale).double();groups=[];shift=0
 for v in items[:2]:groups.append(torch.tensor(np.flatnonzero(v['high'])+shift));shift+=len(v['muscles'])
 limits=torch.stack([(anchor[idx]-target[idx]).square().mean(0)*1.019 for idx in groups]);old_logits=parent.base.projection(z.float())[:,1:].detach().double()
 centers=None;width=None
 if a.variant=='affine':
  origin=np.concatenate([parent.base.projection.weight.detach().numpy()[1:].reshape(-1),parent.base.projection.bias.detach().numpy()[1:]]).astype(float);x0=origin.copy();bounds=[(-10,10)]*14
  for j in [4,5]:bounds[j]=(.0001,10)
  for j in [10,11]:bounds[j]=(-10,-.0001)
  features=z
 else:
  cloud=z[low,:4].numpy();selected=[0];distance=((cloud-cloud[0])**2).sum(1)
  while len(selected)<min(24,len(cloud)):
   j=int(distance.argmax())
   if distance[j]<1e-12:break
   selected.append(j);distance=np.minimum(distance,((cloud-cloud[j])**2).sum(1))
  centers=cloud[selected];pair=np.sqrt(((centers[:,None]-centers[None])**2).sum(-1));np.fill_diagonal(pair,np.inf);width=float(max(.001,np.median(pair.min(1)))) if len(centers)>1 else 1.
  features=torch.exp(-.5*(z[:,:4,None].transpose(1,2)-torch.tensor(centers)[None]).square().sum(-1)/(width*width));origin=np.zeros(2*len(centers));x0=origin.copy();bounds=[(-10,10)]*len(x0)
 def prediction(theta):
  if a.variant=='affine':return (z@theta[:12].reshape(2,6).T+theta[12:]).sigmoid()
  return (old_logits+features@theta.reshape(2,-1).T).sigmoid()
 def values(x):
  theta=torch.tensor(x,dtype=torch.float64,requires_grad=True);pred=prediction(theta);loss=(pred[low]-target[low]).square().mean()+.0001*(theta-torch.tensor(origin)).square().mean();constraints=torch.stack([limits[k]-(pred[idx]-target[idx]).square().mean(0) for k,idx in enumerate(groups)]).reshape(-1);return theta,loss,constraints
 def objective(x):
  th,loss,con=values(x);g=torch.autograd.grad(loss,th)[0];return float(loss),g.detach().numpy()
 def constraints(x):
  th,loss,con=values(x);jac=torch.stack([torch.autograd.grad(v,th,retain_graph=True)[0] for v in con]);return con.detach().numpy(),jac.numpy()
 config=dict(actor_kind=a.actor_kind,variant=a.variant,parent_checkpoint=checkpoint,parent_checkpoint_sha256=sha256(checkpoint),fixed_decoder=plan['fixed_decoder'],fixed_decoder_sha256=plan['fixed_decoder_sha256'],split=str(SPLIT),split_sha256=sha256(SPLIT),evaluation=str(evaluation),evaluation_sha256=sha256(evaluation/'predictions.npz'),receipts=[v['receipt'] for v in items],seed=784105,optimizer='SLSQP with analytic gradients; max500 iterations; TRAIN constraints per episode/pedal',training_mse_ratio=1.019,runtime_admission_mse_ratio=1.02,development_used_for_fit=False,source_sha256={f:sha256(f) for f in [__file__,'src/flybrain_drive/balance_muscle_pedal_readout.py','scripts/fit_current_flight_launch_pedals.py']},representation='Existing six flight muscles; RBF basis uses only first4 balance proxies, never power/speed/teacher/map; original power derivatives and steering unchanged')
 out=begin_run(a.output,config);journal=[];start=time.perf_counter()
 def callback(x):
  loss,g=objective(x);con,jac=constraints(x);journal.append(dict(parameters=x.tolist(),loss=loss,minimum_train_slack=float(con.min())));write_json(out/'progress.json',dict(iterations=len(journal),loss=loss,minimum_train_slack=float(con.min()),elapsed=time.perf_counter()-start))
 opt=minimize(objective,x0,jac=True,bounds=bounds,constraints=[dict(type='ineq',fun=lambda x:constraints(x)[0],jac=lambda x:constraints(x)[1])],method='SLSQP',callback=callback,options=dict(maxiter=500,ftol=1e-10));con,_=constraints(opt.x)
 head=LearnedFlightPedalReadout(parent) if a.variant=='affine' else BalanceMusclePedalReadout(parent,torch.tensor(centers,dtype=torch.float32),torch.tensor(width,dtype=torch.float32))
 with torch.no_grad():
  if a.variant=='affine':head.pedal_projection.weight[1:].copy_(torch.tensor(opt.x[:12].reshape(2,6),dtype=torch.float32));head.pedal_projection.bias[1:].copy_(torch.tensor(opt.x[12:],dtype=torch.float32))
  else:head.offset.weight.copy_(torch.tensor(opt.x.reshape(2,-1),dtype=torch.float32))
 head.verify();rows,preds=metrics(head,items);ratios=[(np.array(v['groups']['high']['learned_rmse'][1:])/np.array(v['groups']['high']['baseline_rmse'][1:]))**2 for v in rows[:2]];feasible=bool(con.min()>=-1e-7 and np.max(ratios)<=1.02)
 state=head.pedal_projection.state_dict() if a.variant=='affine' else dict(centers=head.centers,width=head.width,offset=head.offset.state_dict())
 torch.save(dict(schema='flybrain.balanced_muscle_pedals.experimental.v1',variant=a.variant,parent_checkpoint_sha256=config['parent_checkpoint_sha256'],actor_kind=a.actor_kind,state=state,actor_qualified=False),out/'pedals.pt');np.savez_compressed(out/'predictions.npz',**preds);write_json(out/'training-journal.json',journal)
 result=dict(status='completed_balanced_muscle_pedal_fit' if feasible else 'failed_constraints',optimizer_success=bool(opt.success),optimizer_message=str(opt.message),iterations=int(opt.nit),checkpoint_sha256=sha256(out/'pedals.pt'),predictions_sha256=sha256(out/'predictions.npz'),metrics=rows,training_high_mse_ratios=[r.tolist() for r in ratios],minimum_train_slack=float(con.min()),low_train_labels=int(low.sum()),actor_qualified=False,vehicle_control_sent=False,development_used_for_fit=False,source_circuit_and_steering_unchanged=True,elapsed=time.perf_counter()-start)
 write_json(out/'result.json',result);print(result)
if __name__=='__main__':main()
