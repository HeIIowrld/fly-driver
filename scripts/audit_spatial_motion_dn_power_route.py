"""Observe dedicated physical speed-error/power currents in the factual actor path."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.spatial_motion_dn_experiment import SpatialMotionDNExperiment
ROOT=Path('runs/spatial-motion-dn-learning-20261104')
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=SpatialMotionDNExperiment(p,train_body=True);fit=ROOT/'visual_native-fit-01';assert sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.calibration.requires_grad_(False);e.body_port.requires_grad_(False);rows=[]
 def record(module,args,value):
  command,body=args[:2];rows.append(dict(desired_yaw=float(command[0,0]),desired_speed=float(command[0,1]),body_speed=float(body[0,16]),speed_error=float(command[0,1]-body[0,16]),power=value[1].detach().numpy()[0].copy()))
 hook=e.model.motor.command.register_forward_hook(record);out=begin_run(ROOT/'dedicated-power-route-01',dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(fit/'ports.pt'),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,scope='Factual first1024 decisions. Dedicated physical speed-error route only. Pedal outputs can still vary via visual/body effects on other chemical/muscle paths; this audit does not assert no acceleration control.'))
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=0);prediction=np.asarray([e.step(i,False).numpy() for i in range(1024)]);power=np.asarray([r['power'] for r in rows]);error=np.asarray([r['speed_error'] for r in rows]);assert len(rows)==1024
  with np.load(ROOT/'visual_native-evaluation-01/predictions.npz') as z:parity=float(abs(prediction-z['prediction_0'][:1024]).max())
  assert parity==0 and abs(error).max()==0;e.verify_fixed();np.savez_compressed(out/'values.npz',prediction=prediction,power=power,speed_error=error,desired_yaw=np.array([r['desired_yaw'] for r in rows]));write_json(out/'result.json',dict(status='completed_factual_dedicated_power_route_audit',events=1024,maximum_prediction_difference=parity,maximum_dedicated_speed_error=float(abs(error).max()),dedicated_power_minimum=float(power.min()),dedicated_power_maximum=float(power.max()),dedicated_power_variation=float(np.ptp(power,axis=0).max()),speed_command_boundary_attached=e.model.speed_command_boundary is not None,values_sha256=sha256(out/'values.npz'),actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False))
 finally:hook.remove();e.close()
if __name__=='__main__':main()
