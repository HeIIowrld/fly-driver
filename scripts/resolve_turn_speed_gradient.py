"""Resolve new speed-path derivative at smaller TRAIN perturbations; no fit."""
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
 torch.set_num_threads(1);root=Path('runs/visual-turn-speed-20261104');plan=read(root/'plan-01.json');e=TurnSpeedExperiment(plan);out=begin_run(root/'gradient-resolution-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(root/'plan-01.json'),parent_candidate=e.parent,scope=__doc__,vehicle_control_sent=False));task=read(root/'validation-01/result.json')['gradient_task'];episode,begin,end=task['episode'],task['begin'],task['end'];weight=e.speed.visual_projection.weight;initial=weight.detach().clone()
 try:
  parts,prediction=real_window(e,episode,begin,end,learn=True,parent_prefix=True);loss=sum(parts);baseline=float(loss.detach());base_prediction=prediction.detach().numpy();parity=float(abs(base_prediction-e.anchors[episode][begin:end]).max());assert parity<2e-6;loss.backward();gradient=weight.grad.detach().clone();norm=float(gradient.norm());direction=gradient/gradient.norm();reports=[]
  for step in (1e-4,2e-5,1e-5,2e-6):
   values=[];components=[];deltas=[]
   for sign in (-1,1):
    with torch.no_grad():weight.copy_(initial+sign*step*direction)
    parts,prediction=real_window(e,episode,begin,end,learn=True,parent_prefix=True);values.append(float(sum(parts).detach()));components.append([float(v.detach()) for v in parts]);deltas.append(np.max(abs(prediction.detach().numpy()-base_prediction),axis=0).tolist())
   estimate=(values[1]-values[0])/(2*step);reports.append(dict(step=step,loss_minus_plus=values,loss_parts_minus_plus=components,maximum_action_difference_minus_plus=deltas,analytic=norm,finite_difference=estimate,relative_error=abs(estimate-norm)/max(norm,1e-12)))
  with torch.no_grad():weight.copy_(initial)
  parts,prediction=real_window(e,episode,begin,end,learn=True,parent_prefix=True);repeat=float(abs(prediction.detach().numpy()-base_prediction).max());assert repeat==0.;e.verify_fixed();validated=all(v['relative_error']<.1 and v['finite_difference']>0 for v in reports[-2:]);result=dict(status='completed_speed_gradient_resolution',baseline_loss=baseline,baseline_parent_parity=parity,baseline_repeat_difference=repeat,task=task,reports=reports,gradient_validated=validated,required_adjacent_steps=[1e-5,2e-6],optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False,scope='Only one TRAIN point/direction; small-step convergence and baseline restart reproducibility, not global dynamics or physical speed control qualification.');write_json(out/'result.json',result);print(json.dumps(result))
 finally:e.close()
if __name__=='__main__':main()
