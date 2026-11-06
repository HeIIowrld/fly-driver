"""Disposable projected gate-descent probe using valid TRAIN labels only."""
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.validate_full_range_dn_registration import ROOT,RegistrationProbe
from scripts.sac_steering_experiment import restore

def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');e=RegistrationProbe(plan);task=plan['physical_tasks'][0];begin,end=task['begin'],task['end'];states=torch.load(ROOT/'visual_native-evaluation-01/candidate-prefix-states.pt',map_location='cpu',weights_only=True);prefix=states[0,begin];d=e.items[0][0];target=torch.tensor(d['labels'][begin:end]);valid=torch.tensor(d['fit_mask'][begin:end]);assert valid.any();e.holder.update(mode='real',episode=0);out=begin_run(ROOT/'registration-descent-probe-01',dict(program_sha256=sha256(__file__),task=task,valid_teacher_points=int(valid.sum()),seed=None,actor_qualified=False,vehicle_control_sent=False));rows=[]
 try:
  def loss():
   restore(e.model,prefix);p=torch.stack([e.step(i,True) for i in range(begin,end)]);return (((p-target).square()*p.new_tensor([20.,1.,4.])).sum(1)[valid]).mean()
  value=loss();value.backward();g=e.registration.gate.grad.detach().clone();norm=float(g.norm());assert norm>0;initial=float(value.detach())
  for step in [.001,.0002]:
   proposed=(-step*g/g.norm()).clamp(0,1)
   with torch.no_grad():e.registration.gate.copy_(proposed)
   e.registration.zero_grad(set_to_none=True);test=loss();test.backward();rows.append(dict(projected_step_norm=step,gate=proposed.tolist(),valid_teacher_loss=float(test.detach()),loss_change=float(test.detach())-initial,projection_gradient_norm=float(e.registration.projection.weight.grad.norm())))
  write_json(out/'result.json',dict(status='completed_projected_registration_gate_descent_probe',initial_gate_gradient=g.tolist(),initial_valid_teacher_loss=initial,probes=rows,feasible_nonzero_gate=any(any(v>0 for v in row['gate']) for row in rows),observed_descent=any(row['loss_change']<0 for row in rows),checkpoint_written=False,new_driver_trained=False,actor_qualified=False,vehicle_control_sent=False,scope='One TRAIN prefix, disposable projected gate perturbations. Validates bootstrap possibility only, not a learned model or general driving improvement.'))
 finally:e.close()
if __name__=='__main__':main()
