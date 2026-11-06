"""Feasible-start exact-tanh worst-cohort steering optimization, no relaxed yaw signs."""
import numpy as np
import torch
from scipy.optimize import linprog,minimize,LinearConstraint,Bounds
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from scripts.multicorner_command_data import ROOT,collections
from scripts.paired_decoder_data import load_data

def main():
 torch.set_num_threads(1);protocol=read(ROOT/'protocol-01.json');head,receipt=load_muscle_interaction_head(ROOT/'quadratic-fit-01/decoder.pt');source=collections(protocol);_,items,_,baseline,pre=load_data(read(protocol['data_plan']));plan=dict(program_sha256=sha256(__file__),seed=None,training_roles=['TRAIN','TRAIN_diagnostic'],development_used_for_fit=False,reserved_features_read=False,maximum_steering_rmse_ratio=.98,coefficient_bound=10.,minimum_mean_yaw_logit_slope=.005,per_frame_wrong_logit_tolerance=0.,maximum_iterations=4000,scope=__doc__);out=begin_run(ROOT/'robust_quadratic-fit-01',dict(protocol=protocol,robust_fit=plan,variant='robust_quadratic',seed=None,actor_qualified=False,vehicle_control_sent=False));groups=[];designs=[];targets=[];weights=[];rows=[];upper=[]
 def features(mu):return head.features(torch.tensor(mu,dtype=torch.float32)).detach().numpy()
 for episode,(path,r) in enumerate(source[:2]):
  with np.load(path/'neutral-full.npz') as z:mu=z['muscles'].copy()
  d=items[episode][0];a=np.column_stack([features(mu),np.ones(len(mu))]);target=d['labels'][:,0].astype(float);y=np.arctanh(np.clip(target,-.999,.999));mask=d['fit_mask'];designs.append(a[mask]);targets.append(y[mask]);weights.append(np.where(pre[episode][mask],20.,4.))
  for role,selection in [('all',mask),('pre',pre[episode]&mask)]:groups.append(dict(episode=episode,role=role,a=a[selection],target=target[selection],original_mse=float(np.mean((baseline[episode][selection,0]-target[selection])**2))))
  with np.load(path/'probes.npz') as z:
   for k,row in enumerate(r['reports']):
    assert row['task']['role'] in plan['training_roles'];raw=z[f'case{k}_yaw_muscles'];neutral=features(z[f'case{k}_neutral_muscles']);delta=(features(raw.reshape(-1,6)).reshape(2,32,27)-neutral)/protocol['offsets']['yaw']
    for branch,sign in [(0,-1),(1,1)]:rows.extend([np.r_[sign*v,0.] for v in delta[branch]]);upper.extend([0.]*32);rows.append(np.r_[sign*delta[branch].mean(0),0.]);upper.append(-.005)
 b=np.asarray(rows);ub=np.asarray(upper);feasible=linprog(np.zeros(28),A_ub=b,b_ub=ub,bounds=[(-10,10)]*28,method='highs');assert feasible.success,feasible.message;x=np.concatenate(designs);y=np.concatenate(targets);weight=np.concatenate(weights);h=x.T@(weight[:,None]*x)/weight.sum()+1e-6*np.eye(28);g=x.T@(weight*y)/weight.sum();phase1=minimize(lambda w:float(w@h@w-2*g@w),feasible.x,jac=lambda w:2*(h@w-g),method='SLSQP',bounds=Bounds(-10*np.ones(28),10*np.ones(28)),constraints=[LinearConstraint(b,-np.inf,ub)],options=dict(ftol=1e-12,maxiter=4000));assert phase1.success and (b@phase1.x-ub).max()<1e-7,(phase1.message,float((b@phase1.x-ub).max()))
 def ratios(w):return np.array([np.mean((np.tanh(v['a']@w)-v['target'])**2)/v['original_mse'] for v in groups])
 def ratios_jac(w):
  result=[]
  for v in groups:
   p=np.tanh(v['a']@w);result.append(2*v['a'].T@((p-v['target'])*(1-p*p))/len(p)/v['original_mse'])
  return np.asarray(result)
 initial=np.r_[phase1.x,float(ratios(phase1.x).max()+.01)];solution=minimize(lambda v:float(v[-1]),initial,jac=lambda v:np.r_[np.zeros(28),1.],method='SLSQP',bounds=Bounds(np.r_[-10*np.ones(28),0.],np.r_[10*np.ones(28),np.inf]),constraints=[LinearConstraint(np.column_stack([b,np.zeros(len(b))]),-np.inf,ub),dict(type='ineq',fun=lambda v:v[-1]-ratios(v[:-1]),jac=lambda v:np.column_stack([-ratios_jac(v[:-1]),np.ones(4)]))],options=dict(ftol=1e-11,maxiter=4000));theta=solution.x[:-1];actual_ratios=ratios(theta);linear_violation=float((b@theta-ub).max());epigraph_violation=float((actual_ratios-solution.x[-1]).max());result=dict(variant='robust_quadratic',phase1_success=bool(phase1.success),phase1_iterations=int(phase1.nit),solver_success=bool(solution.success),solver_message=solution.message,solver_iterations=int(solution.nit),linear_constraint_violation=linear_violation,epigraph_constraint_violation=epigraph_violation,maximum_training_actual_mse_ratio=float(actual_ratios.max()),cohort_actual_rmse_ratios=[dict(episode=v['episode'],role=v['role'],ratio=float(np.sqrt(r))) for v,r in zip(groups,actual_ratios)],actor_qualified=False,vehicle_control_sent=False,global_optimum_proven=False)
 np.savez_compressed(out/'diagnostic-solution.npz',theta=theta,rows=b,upper_bounds=ub,hessian=h,linear_objective=g)
 if not solution.success or linear_violation>1e-7 or epigraph_violation>1e-7 or actual_ratios.max()>plan['maximum_steering_rmse_ratio']**2+1e-9:
  write_json(out/'result.json',dict(result,status='protected_feasible_solution_not_established',termination_reason='Exact tanh worst-cohort optimizer did not establish declared 2% TRAIN preservation and all30 command constraints; no checkpoint, no relaxed target. This is not an infeasibility proof.'));print(result);return
 with torch.no_grad():head.steering.weight.copy_(torch.tensor(theta[:-1][None],dtype=torch.float64));head.steering.bias.copy_(torch.tensor(theta[-1:],dtype=torch.float64))
 torch.save(dict(schema='flybrain.muscle_interaction_decoder.experimental.v1',degree=2,readout=head.state_dict(),parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'],actor_qualified=False),out/'decoder.pt');write_json(out/'result.json',dict(result,status='completed_multicorner_constrained_decoder_fit',checkpoint_sha256=sha256(out/'decoder.pt'),termination_reason='Feasible-start exact tanh worst-cohort optimization met declared TRAIN actual steering bounds and all30 bidirectional command constraints. No development/test selection.'));print(result)
if __name__=='__main__':main()
