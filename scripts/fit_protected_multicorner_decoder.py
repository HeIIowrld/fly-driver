"""Adaptive TRAIN-only protected quadratic decoder, all 30 TRAIN command windows.

Cold/hot all/pre steering logit RMSE bounds imply actual steering RMSE bounds
because tanh is 1-Lipschitz and labels are strictly within (-.999,.999).
Development and either reserved image set are never read by this optimizer.
"""
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
 torch.set_num_threads(1);protocol=read(ROOT/'protocol-01.json');source=collections(protocol);head,receipt=load_muscle_interaction_head(ROOT/'quadratic-fit-01/decoder.pt');_,items,_,baseline,pre=load_data(read(protocol['data_plan']));source_cp=ROOT/'quadratic-fit-01/decoder.pt';plan=dict(seed=None,program_sha256=sha256(__file__),source_checkpoint_sha256=sha256(source_cp),training_roles=['TRAIN','TRAIN_diagnostic'],previous_late_diagnostic_promoted_to_train=True,maximum_steering_rmse_ratio=.98,logit_bound_is_sufficient_for_actual_rmse=True,development_used_for_fit=False,reserved_features_read=False,new_reserved_texture_seed=915320087,coefficient_bound=10.,minimum_mean_yaw_logit_slope=.005,ridge=1e-6,solver_tolerance=1e-12,maximum_solver_iterations=4000,scope=__doc__);write_json(ROOT/'protected-fit-plan-01.json',plan);reserved=read(protocol['reserved_plan']);reserved['texture_seed']=915320087;reserved['role']='fresh_reserved_after_train_cohort_protection_refit';write_json(ROOT/'protected-reserved-plan-01.json',reserved);out=begin_run(ROOT/'protected_quadratic-fit-01',dict(protocol=protocol,protected_fit=plan,seed=None,variant='protected_quadratic',trainable_parameters=28,actor_qualified=False,vehicle_control_sent=False));dimension=28;designs=[];truths=[];weights=[];group_constraints=[];tasks=[];rows=[];upper=[]
 def features(mu):return head.features(torch.tensor(mu,dtype=torch.float32)).detach().numpy()
 for e,(path,r) in enumerate(source[:2]):
  with np.load(path/'neutral-full.npz') as z:mu=z['muscles'].copy()
  d=items[e][0];x=np.column_stack([features(mu),np.ones(len(mu))]);target=d['labels'][:,0].astype(float);assert np.max(abs(target[d['fit_mask']]))<.999;y=np.arctanh(np.clip(target,-.999,.999));mask=d['fit_mask'];designs.append(x[mask]);truths.append(y[mask]);weights.append(np.where(pre[e][mask],20.,4.))
  for role,selection in [('all',mask),('pre',pre[e]&mask)]:
   a=x[selection];b=y[selection];old_rmse=float(np.sqrt(np.mean((baseline[e][selection,0]-target[selection])**2)));limit=(plan['maximum_steering_rmse_ratio']*old_rmse)**2;group_constraints.append(dict(episode=e,role=role,h=a.T@a/len(a),g=a.T@b/len(a),c=float(np.mean(b*b)),limit=limit,original_actual_rmse=old_rmse))
  with np.load(path/'probes.npz') as z:
   for k,row in enumerate(r['reports']):
    task=row['task'];assert task['role'] in plan['training_roles'];tasks.append(task);mu=z[f'case{k}_yaw_muscles'];base=features(z[f'case{k}_neutral_muscles']);delta=(features(mu.reshape(-1,6)).reshape(2,32,27)-base)/protocol['offsets']['yaw']
    for branch,sign in [(0,-1),(1,1)]:
     rows.extend([np.r_[sign*v,0.] for v in delta[branch]]);upper.extend([0.]*32);rows.append(np.r_[sign*delta[branch].mean(0),0.]);upper.append(-plan['minimum_mean_yaw_logit_slope'])
 assert len(tasks)==30;x=np.concatenate(designs);y=np.concatenate(truths);weight=np.concatenate(weights);h=x.T@(weight[:,None]*x)/weight.sum()+plan['ridge']*np.eye(dimension);g=x.T@(weight*y)/weight.sum();b=np.asarray(rows);ub=np.asarray(upper);feasible=linprog(np.zeros(dimension),A_ub=b,b_ub=ub,bounds=[(-10,10)]*dimension,method='highs');assert feasible.success,feasible.message
 def protection(w):return np.array([v['limit']-(w@v['h']@w-2*v['g']@w+v['c']) for v in group_constraints])
 def protection_jac(w):return np.array([-2*(v['h']@w-v['g']) for v in group_constraints])
 initial=np.r_[head.steering.weight.detach().numpy()[0],float(head.steering.bias.detach()[0])];solution=minimize(lambda w:float(w@h@w-2*g@w),initial,jac=lambda w:2*(h@w-g),method='SLSQP',bounds=Bounds(-10*np.ones(28),10*np.ones(28)),constraints=[LinearConstraint(b,-np.inf,ub),dict(type='ineq',fun=protection,jac=protection_jac)],options=dict(ftol=plan['solver_tolerance'],maxiter=plan['maximum_solver_iterations']));linear_violation=float((b@solution.x-ub).max());protection_violation=float(-protection(solution.x).min());result=dict(variant='protected_quadratic',solver_success=bool(solution.success),solver_message=solution.message,solver_iterations=int(solution.nit),linear_constraint_violation=linear_violation,protection_constraint_violation=protection_violation,group_constraints=[{k:v for k,v in group.items() if k not in ['h','g']} for group in group_constraints],trainable_parameters=28,training_cases=30,actor_qualified=False,vehicle_control_sent=False,seed=None)
 if not solution.success or linear_violation>1e-7 or protection_violation>1e-9:
  write_json(out/'result.json',dict(result,status='protected_feasible_solution_not_established',termination_reason='Solver did not establish all declared TRAIN physical-direction and cohort-preservation inequalities; no candidate written, no relaxation or development selection.'));print(result);return
 with torch.no_grad():head.steering.weight.copy_(torch.tensor(solution.x[:-1][None],dtype=torch.float64));head.steering.bias.copy_(torch.tensor(solution.x[-1:],dtype=torch.float64))
 torch.save(dict(schema='flybrain.muscle_interaction_decoder.experimental.v1',degree=2,readout=head.state_dict(),parent_checkpoint_sha256=protocol['parent_checkpoint_sha256'],actor_qualified=False),out/'decoder.pt');np.savez_compressed(out/'constraint-system.npz',rows=b,upper_bounds=ub,hessian=h,linear_objective=g,group_hessians=np.array([v['h'] for v in group_constraints]),group_linear=np.array([v['g'] for v in group_constraints]),group_constants=np.array([v['c'] for v in group_constraints]),group_limits=np.array([v['limit'] for v in group_constraints]));write_json(out/'result.json',dict(result,status='completed_multicorner_constrained_decoder_fit',checkpoint_sha256=sha256(out/'decoder.pt'),constraint_system_sha256=sha256(out/'constraint-system.npz'),termination_reason='All30 TRAIN command windows and 4 cold/hot all/pre steering logit-error upper bounds satisfied; unchanged neural and pedal weights, no development/test selection.'));print(result)
if __name__=='__main__':main()
