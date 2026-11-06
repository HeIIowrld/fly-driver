"""TRAIN-only manually opened-gate operating points, no optimizer/candidate selection."""
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.validate_full_range_dn_registration import ROOT,RegistrationProbe
from scripts.sac_steering_experiment import restore

class CurrentProbe(RegistrationProbe):
 def _descending(self,module,args,value):
  static,_,_,_=self.inputs();steer,power=value;self.last_registered_current=self.registration(static,self.calibration(static,self.dn(static)));return steer+self.last_registered_current,power

def main():
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');e=CurrentProbe(plan);states=torch.load(ROOT/'visual_native-evaluation-01/candidate-prefix-states.pt',map_location='cpu',weights_only=True);tasks=[plan['physical_tasks'][i] for i in [0,3,5,15]];gates=[0.,.005,.01,.02,.05];out=begin_run(ROOT/'registration-tonic-grid-01',dict(program_sha256=sha256(__file__),tasks=tasks,manual_gates=gates,seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));reports=[]
 try:
  for task in tasks:
   episode,begin,end=task['episode'],task['begin'],task['end'];d=e.items[episode][0];target=torch.tensor(d['labels'][begin:end]);valid=torch.tensor(d['fit_mask'][begin:end]);assert valid.any();e.holder.clear();e.holder.update(mode='real',episode=episode)
   for gate in gates:
    restore(e.model,states[episode,begin]);e.registration.zero_grad(set_to_none=True)
    with torch.no_grad():e.registration.gate.fill_(gate)
    pred=[];currents=[]
    for i in range(begin,end):pred.append(e.step(i,True));currents.append(e.last_registered_current.detach().numpy()[0])
    p=torch.stack(pred);loss=(((p-target).square()*p.new_tensor([20.,1.,4.])).sum(1)[valid]).mean();loss.backward();reports.append(dict(task=task,manual_gate=gate,valid_teacher_points=int(valid.sum()),loss=float(loss.detach()),mean_current=np.mean(currents,0).tolist(),gate_gradient=e.registration.gate.grad.detach().tolist(),projection_gradient_norm_by_port=e.registration.projection.weight.grad.detach().norm(dim=1).tolist()))
  write_json(out/'result.json',dict(status='completed_manual_registration_operating_point_grid',reports=reports,registration_optimizer_updates=0,new_driver_trained=False,actor_qualified=False,vehicle_control_sent=False,scope='Four TRAIN prefixes with manually changed gate and original temporal/body context. Identifies local gradient availability only; no stationary neutral calibration, learned model, developmental selection or physiological claim.'))
 finally:e.close()
if __name__=='__main__':main()
