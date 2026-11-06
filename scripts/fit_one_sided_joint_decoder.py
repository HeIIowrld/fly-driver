"""Convex TRAIN steering-logit fit with separate +/- yaw constraints.

Only seven steering coefficients change. Gas/brake and candidate neural paths
remain exact. Reuses complete fixed-parent muscle histories, not a neural replay.
"""
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize,linprog,LinearConstraint,Bounds
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from scripts.paired_decoder_data import load_data
from scripts.improve_sac_anticipation import command_metrics

def main():
 root=Path('runs/joint-dn-decoder-20261104');parent=root/'rank4-fit-01/joint-dn-decoder.pt';evaluation=root/'rank4-evaluation-01';cp=torch.load(parent,map_location='cpu',weights_only=True);assert sha256(parent)==read(parent.parent/'result.json')['checkpoint_sha256'];source=root/'yaw-decomposition-01';assert sha256(source/'muscle-probes.npz')==read(source/'result.json')['muscle_probes_sha256'];reserved=read(root/'reserved-plan-01.json');reserved['texture_seed']=781191;reserved['role']='reserved_after_adaptive_parent_selection_one_sided_refit';write_json(root/'one-sided-reserved-plan-01.json',reserved);plan=dict(program_sha256=sha256(__file__),parent_checkpoint_sha256=sha256(parent),muscle_history_sha256=sha256(evaluation/'predictions.npz'),yaw_muscles_sha256=sha256(source/'muscle-probes.npz'),training_episodes=[0,1],training_yaw_cases=[0,2],minimum_mean_logit_slope_per_rad_s=.005,each_frame_wrong_direction_allowed=False,coefficient_bound=10.,ridge=1e-6,maximum_iterations=2000,solver_tolerance=1e-12,actor_trainable_parameters=7,neural_parameters_changed=0,development_used_for_fit=False,parent_selected_after_consumed_diagnostics=True,reserved_plan_sha256=sha256(root/'one-sided-reserved-plan-01.json'),scope=__doc__);write_json(root/'one-sided-fit-plan-01.json',plan);out=begin_run(root/'one-sided-refit-01',plan);data_plan=read('runs/paired-muscle-decoder-20261104/plan-01.json');_,items,_,baseline,pre=load_data(data_plan);s={k:v.numpy().astype(float) for k,v in cp['readout'].items()};histories=np.load(evaluation/'predictions.npz');mu=[histories[f'muscles_{e}'].copy() for e in range(3)];x=np.concatenate([mu[e][items[e][0]['fit_mask']] for e in [0,1]]).astype(float);z=(x-s['training_reference'])/s['training_scale'];design=np.column_stack([z,np.ones(len(z))]);target=np.concatenate([items[e][0]['labels'][items[e][0]['fit_mask'],0] for e in [0,1]]).astype(float);premask=np.concatenate([pre[e][items[e][0]['fit_mask']] for e in [0,1]]);weight=np.where(premask,20.,4.);y=np.arctanh(np.clip(target,-.999,.999));ridge=plan['ridge'];h=design.T@(weight[:,None]*design)/weight.sum()+ridge*np.eye(7);g=design.T@(weight*y)/weight.sum();rows=[];upper=[]
 with np.load(source/'muscle-probes.npz') as probes:
  for case in plan['training_yaw_cases']:
   delta=(probes[f'TRAIN_{case}_muscles'].astype(float)-probes[f'TRAIN_{case}_neutral_muscles'])/s['training_scale']/.01
   for branch,sign in [(0,-1),(1,1)]:
    for frame in delta[branch]:rows.append(np.r_[sign*frame,0]);upper.append(0.)
    rows.append(np.r_[sign*delta[branch].mean(0),0]);upper.append(-plan['minimum_mean_logit_slope_per_rad_s'])
 b=np.array(rows);ub=np.array(upper);feasible=linprog(np.zeros(7),A_ub=b,b_ub=ub,bounds=[(-10,10)]*7,method='highs');assert feasible.success,feasible.message;old=np.r_[s['projection.weight'][0],s['projection.bias'][0]];solution=minimize(lambda w:float(w@h@w-2*g@w),feasible.x,jac=lambda w:2*(h@w-g),hess=lambda w:2*h,method='SLSQP',bounds=Bounds(-10*np.ones(7),10*np.ones(7)),constraints=[LinearConstraint(b,-np.inf,ub)],options=dict(ftol=plan['solver_tolerance'],maxiter=plan['maximum_iterations']));violation=float((b@solution.x-ub).max());assert solution.success and violation<1e-7,(solution.message,violation);state={k:v.clone() for k,v in cp['readout'].items()};state['projection.weight'][0]=torch.tensor(solution.x[:6],dtype=torch.float32);state['projection.bias'][0]=float(solution.x[6]);theta=np.r_[state['projection.weight'][0].numpy().astype(float),float(state['projection.bias'][0])];float_violation=float((b@theta-ub).max());assert float_violation<1e-6;predictions={};metrics=[]
 def decode(raw):
  features=(raw.astype(float)-s['training_reference'])/s['training_scale'];value=features@s['projection.weight'].T+s['projection.bias'];value[:,0]=features@theta[:6]+theta[6];return np.concatenate([np.tanh(value[:,:1]),1/(1+np.exp(-value[:,1:]))],1).astype(np.float32)
 for e in range(3):
  p=decode(mu[e]);predictions[f'prediction_{e}']=p;d=items[e][0];metrics.append(dict(episode=e,metrics={role:dict(parent=command_metrics(baseline[e],d['labels'],mask),refit=command_metrics(p,d['labels'],mask)) for role,mask in [('all',d['fit_mask']),('pre',pre[e])]}))
 reports=[]
 with np.load(source/'muscle-probes.npz') as probes:
  for case in read(source/'result.json')['reports']:
   key=f"{case['role']}_{case['case']}";branches=probes[key+'_muscles'];p=np.array([decode(v) for v in branches]);neutral=decode(probes[key+'_neutral_muscles']);minus=p[0,:,0]-neutral[:,0];plus=p[1,:,0]-neutral[:,0];predictions[key+'_yaw_prediction']=p;reports.append(dict(role=case['role'],case=case['case'],mean_minus_change=float(minus.mean()),mean_plus_change=float(plus.mean()),correct_minus_fraction=float((minus>1e-7).mean()),correct_plus_fraction=float((plus< -1e-7).mean()),mean_central_derivative=float(((p[1,:,0]-p[0,:,0])/.02).mean())))
 assert torch.equal(state['projection.weight'][1:],cp['readout']['projection.weight'][1:]) and torch.equal(state['projection.bias'][1:],cp['readout']['projection.bias'][1:]);torch.save(dict(schema='flybrain.one_sided_decoder.experimental.v1',readout=state,parent_checkpoint_sha256=sha256(parent),actor_qualified=False),out/'one-sided-head.pt');np.savez_compressed(out/'predictions.npz',**predictions);write_json(out/'result.json',dict(status='completed_one_sided_constrained_decoder_fit',solver_success=bool(solution.success),solver_message=solution.message,solver_iterations=int(solution.nit),minimum_hessian_eigenvalue=float(np.linalg.eigvalsh(h).min()),float64_constraint_violation=violation,float32_constraint_violation=float_violation,actual_recordings=metrics,command_probes=reports,gas_brake_coefficients_exact=True,neural_recurrence_replayed=False,checkpoint_sha256=sha256(out/'one-sided-head.pt'),predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False,termination_reason='Convex TRAIN-only inverse-tanh weighted regression under separate +/- frame and mean yaw inequalities; development evaluated after fixed fit, no candidate selection.'))
 print([(v['episode'],100*(1-v['metrics']['pre']['refit']['rmse'][0]/v['metrics']['pre']['parent']['rmse'][0])) for v in metrics]);print(reports)
if __name__=='__main__':main()
