"""Local TRAIN-only full-routing gates, not full-history calibration or training."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.full_range_port_experiment import FullRangePortExperiment
from scripts.sac_steering_experiment import restore

ROOT=Path('runs/full-range-port-learning-20261104')

def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');e=FullRangePortExperiment(plan,train_body=False);states={i:torch.load(ROOT/f'initial-episode{i}-01/prefix-states.pt',map_location='cpu',weights_only=True) for i in [0,1]};tasks=[plan['physical_tasks'][i] for i in [0,3,5,15]];out=begin_run(ROOT/'full-routing-train-probe-01',dict(program_sha256=sha256(__file__),manual_gates=[.25,.5,.75,1.],tasks=tasks,seed=None,optimizer_updates=0,reserved_pixels_read=False,development_used=False,actor_qualified=False,vehicle_control_sent=False));rows=[]
 try:
  for task in tasks:
   episode,begin,end=task['episode'],task['begin'],task['end'];d=e.items[episode][0];target=torch.tensor(d['labels'][begin:end]);valid=torch.tensor(d['fit_mask'][begin:end]);assert valid.any()
   for gate in [.25,.5,.75,1.]:
    with torch.no_grad():e.calibration.gate.fill_(gate)
    e.holder.clear();e.holder.update(mode='real',episode=episode);restore(e.model,states[episode][begin]);e.calibration.zero_grad(set_to_none=True);p=torch.stack([e.step(i,True) for i in range(begin,end)]);loss=(((p-target).square()*p.new_tensor([20.,1.,4.])).sum(1)[valid]).mean();loss.backward();row=dict(task=task,manual_gate=gate,loss=float(loss.detach()),projection_gradient_norm_by_port=e.calibration.projection.weight.grad.norm(dim=1).tolist(),bias_gradient=e.calibration.projection.bias.grad.tolist(),gate_gradient=e.calibration.gate.grad.tolist());neutral=p.detach().numpy();commands={}
    for field in ['yaw','speed']:
     values=[]
     for sign in [-1,1]:
      restore(e.model,states[episode][begin]);e.yaw_offset=sign*.01 if field=='yaw' else 0.;e.speed_offset=sign*.01 if field=='speed' else 0.;v=np.asarray([e.step(i,False).numpy() for i in range(begin,end)]);values.append(v[:,0]-neutral[:,0] if field=='yaw' else (v[:,1]-v[:,2])-(neutral[:,1]-neutral[:,2]))
     minus,plus=map(np.mean,values);commands[field]=dict(mean_minus_change=float(minus),mean_plus_change=float(plus),both_correct=bool(minus>0 and plus<0) if field=='yaw' else bool(minus<0 and plus>0))
    e.yaw_offset=e.speed_offset=0.;row['commands']=commands;rows.append(row)
  write_json(out/'result.json',dict(status='completed_local_full_routing_train_probe',reports=rows,optimizer_updates=0,new_driver_trained=False,reserved_pixels_read=False,scope='Four original warm .01 TRAIN prefixes manually branched at each gate. Local gradient/body-command evidence only; whole history must be rebuilt at a chosen gate before claiming a calibrated initialization.',actor_qualified=False,vehicle_control_sent=False))
 finally:e.yaw_offset=e.speed_offset=0.;e.close()
if __name__=='__main__':main()
