"""Finite-difference speed sign and six-muscle vehicle-readout attribution, TRAIN only."""
import json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.turn_speed_experiment import TurnSpeedExperiment
from scripts.continuous_descending_experiment import real_window

def main():
 torch.set_num_threads(1);root=Path('runs/visual-turn-speed-20261104');e=TurnSpeedExperiment(read(root/'plan-01.json'));source=read(root/'control-dimensions-01/result.json');out=begin_run(root/'speed-sign-fd-02',dict(program_sha256=sha256(__file__),jacobian_result_sha256=sha256(root/'control-dimensions-01/result.json'),parent_candidate=e.parent,scope=__doc__,vehicle_control_sent=False));offset=torch.tensor(0.);muscles=[]
 def speed_hook(module,args):
  command,body,valid,goals=args[:4];target=torch.minimum((command[:,1:]+offset).clamp_min(0),goals[:,1:]*100/3.6);return (torch.cat([command[:,:1],target],1),*args[1:])
 def muscle_hook(module,args):muscles.append(args[0][0].detach().numpy().copy())
 hooks=[e.model.motor.command.register_forward_pre_hook(speed_hook),e.head.register_forward_pre_hook(muscle_hook)];rows=[];raw_arrays={};head={k:v.numpy().astype(float) for k,v in e.head.state_dict().items()};scale=head['training_scale'];weight=head['projection.weight'];reference=head['training_reference'];bias=head['projection.bias']
 try:
  for case_index,case in enumerate(source['reports']):
   task=case['task'];reports=[]
   for step in (.01,.001):
    values=[];mu=[]
    for sign in (-1,1):
     offset.fill_(sign*step);muscles.clear();_,prediction=real_window(e,task['episode'],task['begin'],task['end'],learn=True,parent_prefix=True);values.append(prediction.detach().numpy());mu.append(np.asarray(muscles))
    raw_arrays[f'case_{case_index}_step_{step}_prediction']=np.asarray(values);raw_arrays[f'case_{case_index}_step_{step}_muscles']=np.asarray(mu)
    derivative=(values[1].astype(float)-values[0])/(2*step);muscle_derivative=(mu[1].astype(float)-mu[0])/(2*step);logits=[((v.astype(float)-reference)/scale)@weight.T+bias for v in mu];sig=[1/(1+np.exp(-z[:,1:])) for z in logits];dz=logits[1][:,1:]-logits[0][:,1:];slopes=np.divide(sig[1]-sig[0],dz,out=sig[0]*(1-sig[0]),where=abs(dz)>1e-12);contrib=(slopes[:,0,None]*weight[1]-slopes[:,1,None]*weight[2])*(muscle_derivative/scale);total=float(np.mean(derivative[:,1]-derivative[:,2]));error=float(abs(contrib.sum(1)-(derivative[:,1]-derivative[:,2])).max());assert error<2e-4;analytic=np.asarray(case['jacobian_rows_steer_gas_brake_columns_turn_raw_and_speed_m_s'])[:,1];fd=derivative.mean(0);relative=float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-12));reports.append(dict(step_m_s=step,mean_vehicle_derivative=fd.tolist(),relative_vector_error_vs_autograd=relative,gas_minus_brake_derivative=total,six_muscle_mean_derivative=muscle_derivative.mean(0).tolist(),six_muscle_gas_minus_brake_contributions=contrib.mean(0).tolist(),readout_attribution_maximum_difference=error))
   rows.append(dict(task=task,reports=reports))
  offset.zero_();e.verify_fixed();np.savez_compressed(out/'probes.npz',**raw_arrays);result=dict(status='completed_train_speed_sign_finite_difference',reports=rows,raw_probes_sha256=sha256(out/'probes.npz'),all_speed_request_gas_minus_brake_derivatives_negative=all(v['gas_minus_brake_derivative']<0 for row in rows for v in row['reports']),all_small_step_vector_relative_errors_below_05=all(row['reports'][-1]['relative_vector_error_vs_autograd']<.05 for row in rows),muscle_order=['b1_L','b1_R','b2_L','b2_R','power_L','power_R'],optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope='Four TRAIN fixed-parent intervals. Numerical speed target increases existing power current, but observed vehicle decoder local longitudinal request may invert. Six outputs are modeled muscle proxies, not measured wing force. No actual vehicle acceleration or global monotonicity claim.');write_json(out/'result.json',result);print(json.dumps(result))
 finally:
  for hook in hooks:hook.remove()
  e.close()
if __name__=='__main__':main()
