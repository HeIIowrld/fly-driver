"""Neutral speed-path parity and TRAIN gradient/command-current registration."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.turn_speed_experiment import TurnSpeedExperiment
from scripts.continuous_descending_experiment import real_window,detach_exact

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);e=TurnSpeedExperiment(read(a.plan));out=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(a.plan),parent_candidate=e.parent,scope=__doc__,vehicle_control_sent=False));predictions=[];powers=[]
 def capture(module,args,value):powers.append(value[1].detach().numpy().copy())
 hook=e.model.motor.command.register_forward_hook(capture)
 try:
  e.model.reset_episode()
  for begin in range(0,1024,32):
   parts,prediction=real_window(e,0,begin,begin+32,learn=True);predictions.append(prediction.detach().numpy());sum(parts).backward()
   for p in e.train_parameters():p.grad=None
   detach_exact(e.model)
  prediction=np.concatenate(predictions);parity=float(abs(prediction-e.anchors[0][:1024]).max());assert parity<2e-6;hook.remove();task=next(t for t in e.real_tasks if e.pre[t['episode']][t['begin']:t['end']].any());parts,_=real_window(e,task['episode'],task['begin'],task['end'],learn=True,parent_prefix=True);loss=sum(parts);loss.backward();weight=e.speed.visual_projection.weight;gradient=weight.grad.detach().clone();norm=float(gradient.norm());assert norm>0;direction=gradient/gradient.norm();initial=weight.detach().clone();fd=[]
  for step in (.001,.0005):
   values=[]
   for sign in (-1,1):
    with torch.no_grad():weight.copy_(initial+sign*step*direction)
    parts,_=real_window(e,task['episode'],task['begin'],task['end'],learn=True,parent_prefix=True);values.append(float(sum(parts).detach()))
   estimate=(values[1]-values[0])/(2*step);fd.append(dict(step=step,analytic=norm,finite_difference=estimate,relative_error=abs(estimate-norm)/max(norm,1e-12)))
  with torch.no_grad():weight.copy_(initial)
  e.verify_fixed();np.savez_compressed(out/'parity.npz',prediction=prediction,power=np.concatenate(powers));result=dict(status='completed_turn_speed_neutral_and_gradient_validation',neutral_real_events=1024,maximum_prediction_difference=parity,power_range=[float(np.min(powers)),float(np.max(powers))],speed_gradient_norm=norm,finite_differences=fd,gradient_validated=min(x['relative_error'] for x in fd)<.1,gradient_task=task,optimizer_updates=0,parity_sha256=sha256(out/'parity.npz'),vehicle_control_sent=False,actor_qualified=False,scope='Neutral path parity and one TRAIN point speed-parameter directional derivative. Desired-speed/current units verified in module tests; physical vehicle acceleration correctness not established.');write_json(out/'result.json',result);print(json.dumps(result))
 finally:hook.remove();e.close()
if __name__=='__main__':main()
