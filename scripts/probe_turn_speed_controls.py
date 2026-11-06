"""TRAIN local turn/speed command controllability through fixed flight muscles."""
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
 torch.set_num_threads(1);root=Path('runs/visual-turn-speed-20261104');e=TurnSpeedExperiment(read(root/'plan-01.json'));out=begin_run(root/'control-dimensions-01',dict(program_sha256=sha256(__file__),parent_candidate=e.parent,scope=__doc__,vehicle_control_sent=False));turn=torch.tensor(0.,requires_grad=True);speed=torch.tensor(0.,requires_grad=True);hooks=[]
 def turn_hook(module,args,value):return value+turn
 def speed_hook(module,args):
  command,body,valid,goals=args[:4];target=torch.minimum((command[:,1:]+speed).clamp_min(0),goals[:,1:]*100/3.6);return (torch.cat([command[:,:1],target],1),*args[1:])
 hooks=[e.dn.projection.register_forward_hook(turn_hook),e.model.motor.command.register_forward_pre_hook(speed_hook)];tasks=[]
 for episode in (0,1):
  available=[t for t in e.real_tasks if t['episode']==episode and e.pre[episode][t['begin']:t['end']].any()];tasks.extend(available[:2])
 reports=[]
 try:
  for task in tasks:
   _,prediction=real_window(e,task['episode'],task['begin'],task['end'],learn=True,parent_prefix=True);mean=prediction.mean(0);jac=np.asarray([[float(v) for v in torch.autograd.grad(mean[axis],[turn,speed],retain_graph=True)] for axis in range(3)]);norm=np.linalg.norm(jac,axis=0);unit=jac/np.maximum(norm,1e-15);singular=np.linalg.svd(unit,compute_uv=False);cosine=float(unit[:,0]@unit[:,1]);reports.append(dict(task=task,jacobian_rows_steer_gas_brake_columns_turn_raw_and_speed_m_s=jac.tolist(),column_norms=norm.tolist(),normalized_column_cosine=cosine,normalized_singular_values=singular.tolist(),speed_derivative_gas_minus_brake=float(jac[1,1]-jac[2,1])))
  e.verify_fixed();write_json(out/'result.json',dict(status='completed_train_turn_speed_control_dimensions',reports=reports,optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope='Local four TRAIN windows; raw turn contrast and desired-speed m/s units differ, columns normalized only for local linear independence. No finite-difference certification, actual acceleration or physiological qualification.'));print(json.dumps(reports))
 finally:
  for hook in hooks:hook.remove()
  e.close()
if __name__=='__main__':main()
