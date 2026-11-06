"""Predeclared affine/quadratic muscle-only steering under multi-state yaw constraints."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog,minimize,LinearConstraint,Bounds
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.muscle_interaction_vehicle_readout import MuscleInteractionVehicleReadout,PAIRS
from scripts.multicorner_command_data import ROOT,collections
from scripts.paired_decoder_data import load_data

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['affine','quadratic'],required=True);a=p.parse_args();torch.set_num_threads(1);protocol=read(ROOT/'protocol-01.json');sources=collections(protocol);_,items,_,_,pre=load_data(read(protocol['data_plan']));parent=torch.load(protocol['parent_checkpoint'],map_location='cpu',weights_only=True)['readout'];muscles=[]
 for path,r in sources:
  with np.load(path/'neutral-full.npz') as z:muscles.append(z['muscles'].copy())
 train=np.concatenate([muscles[e][items[e][0]['fit_mask']] for e in [0,1]]).astype(float);z=(train-parent['training_reference'].numpy().astype(float))/parent['training_scale'].numpy().astype(float);degree=1 if a.variant=='affine' else 2
 if degree==2:q=np.stack([z[:,i]*z[:,j] for i,j in PAIRS],1);qref=torch.tensor(q.mean(0),dtype=torch.float64);qscale=torch.tensor(np.maximum(q.std(0),1e-6),dtype=torch.float64)
 else:qref=qscale=torch.empty(0,dtype=torch.float64)
 head=MuscleInteractionVehicleReadout(parent,degree=degree,quadratic_reference=qref,quadratic_scale=qscale);feature_count=6+(21 if degree==2 else 0);dimension=feature_count+1
 def features(mu):return head.features(torch.tensor(mu,dtype=torch.float32)).detach().numpy()
 design=np.column_stack([features(train),np.ones(len(train))]);target=np.concatenate([items[e][0]['labels'][items[e][0]['fit_mask'],0] for e in [0,1]]).astype(float);ispre=np.concatenate([pre[e][items[e][0]['fit_mask']] for e in [0,1]]);weight=np.where(ispre,20.,4.);truth=np.arctanh(np.clip(target,-.999,.999));h=design.T@(weight[:,None]*design)/weight.sum()+protocol['ridge']*np.eye(dimension);g=design.T@(weight*truth)/weight.sum();frame_rows=[];mean_rows=[];training_tasks=[]
 for path,r in sources[:2]:
  with np.load(path/'probes.npz') as raw:
   for j,row in enumerate(r['reports']):
    task=row['task']
    if task['role']!='TRAIN':continue
    mu=raw[f'case{j}_yaw_muscles'];neutral=features(raw[f'case{j}_neutral_muscles']);branches=features(mu.reshape(-1,6)).reshape(2,-1,feature_count);delta=(branches-neutral)/protocol['offsets']['yaw'];training_tasks.append(task)
    for branch,sign in [(0,-1),(1,1)]:
     frame_rows.extend([np.r_[sign*v,0.] for v in delta[branch]]);mean_rows.append(np.r_[sign*delta[branch].mean(0),0.])
 assert len(training_tasks)==20;frames=np.asarray(frame_rows);means=np.asarray(mean_rows);b=np.concatenate([frames,means]);ub=np.r_[np.zeros(len(frames)),np.full(len(means),-protocol['minimum_mean_yaw_logit_slope'])];bound=protocol['coefficient_bound'];max_b=np.column_stack([b,np.r_[np.zeros(len(frames)),np.ones(len(means))]]);objective=np.r_[np.zeros(dimension),-1.];capacity=linprog(objective,A_ub=max_b,b_ub=np.zeros(len(b)),bounds=[(-bound,bound)]*dimension+[(0,None)],method='highs');assert capacity.success,capacity.message;feasibility=float((max_b@capacity.x).max());assert feasibility<1e-6;maximum_slope=float(capacity.x[-1]);source_hashes={p:sha256(p) for p in ['scripts/fit_multicorner_decoder.py','src/flybrain_drive/muscle_interaction_vehicle_readout.py','scripts/multicorner_command_data.py','tests/test_muscle_interaction_vehicle_readout.py']};out=begin_run(ROOT/f'{a.variant}-fit-01',dict(protocol=protocol,protocol_sha256=sha256(ROOT/'protocol-01.json'),seed=None,variant=a.variant,trainable_parameters=dimension,training_points=len(train),training_tasks=training_tasks,source_sha256=source_hashes,collection_results_sha256=[sha256(path/'result.json') for path,r in sources],neural_weights_changed=False,gas_brake_weights_changed=False,development_used_for_fit=False,reserved_features_read=False,scope=__doc__,actor_qualified=False,vehicle_control_sent=False));np.savez_compressed(out/'constraint-system.npz',frame_rows=frames,mean_rows=means,upper_bounds=ub,hessian=h,linear_objective=g,maximum_margin_solution=capacity.x,quadratic_reference=qref.numpy(),quadratic_scale=qscale.numpy());result=dict(variant=a.variant,maximum_uniform_mean_logit_slope=maximum_slope,capacity_constraint_violation=feasibility,minimum_hessian_eigenvalue=float(np.linalg.eigvalsh(h).min()),trainable_parameters=dimension,training_points=len(train),training_cases=len(training_tasks),actor_qualified=False,vehicle_control_sent=False,neural_recurrence_replayed=False,termination_reason='Declared TRAIN-only constrained steering fit protocol, no development/test selection')
 if maximum_slope<protocol['minimum_mean_yaw_logit_slope']-1e-7:
  result.update(status='infeasible_hard_bilateral_constraints',termination_reason='Maximum uniform TRAIN mean logit slope under all pointwise signs and coefficient bounds is below declared minimum; no actor checkpoint written');write_json(out/'result.json',result);print(result);return
 feasible=linprog(np.zeros(dimension),A_ub=b,b_ub=ub,bounds=[(-bound,bound)]*dimension,method='highs');assert feasible.success,feasible.message;solution=minimize(lambda w:float(w@h@w-2*g@w),feasible.x,jac=lambda w:2*(h@w-g),method='SLSQP',bounds=Bounds(-bound*np.ones(dimension),bound*np.ones(dimension)),constraints=[LinearConstraint(b,-np.inf,ub)],options=dict(ftol=protocol['solver_tolerance'],maxiter=protocol['maximum_solver_iterations']));violation=float((b@solution.x-ub).max())
 if not solution.success or violation>1e-7:
  result.update(status='failed_constrained_quadratic_solver',solver_message=solution.message,solver_constraint_violation=violation,solver_iterations=int(solution.nit));write_json(out/'result.json',result);raise SystemExit(1)
 with torch.no_grad():head.steering.weight.copy_(torch.tensor(solution.x[:-1][None],dtype=torch.float64));head.steering.bias.copy_(torch.tensor(solution.x[-1:],dtype=torch.float64))
 torch.save(dict(schema='flybrain.muscle_interaction_decoder.experimental.v1',degree=degree,readout=head.state_dict(),parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'],actor_qualified=False),out/'decoder.pt');write_json(out/'result.json',dict(result,status='completed_multicorner_constrained_decoder_fit',solver_message=solution.message,solver_iterations=int(solution.nit),solver_constraint_violation=violation,checkpoint_sha256=sha256(out/'decoder.pt'),constraint_system_sha256=sha256(out/'constraint-system.npz')));print(out)
if __name__=='__main__':main()
