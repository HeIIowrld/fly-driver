"""Causal attenuation of trained speed-command feedback on held synthetic input."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.probe_power_bounded_speed_held_inputs import HeldClockExperiment
from scripts.evaluate_controlled_motion_factorial import synthetic_body
ROOT=Path('runs/power-bounded-speed-learning-20261104')
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=HeldClockExperiment(p,learn_speed=True);fit=ROOT/'speed_policy-fit-01';cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);source=ROOT/'reserved-stimuli-01';frame=read(source/'manifest.json')['config']['plan']['branch_frame']-1;n=301
 with np.load(source/'features.npz') as z:
  receipt=100000.+np.arange(n)*.04;offset=z['render_lowers'][frame]-z['camera_receipt'][frame];x={k:np.repeat(z[k][0:1,frame:frame+1],n,axis=1) for k in ['scene','static','motion']};x.update(camera_receipt=receipt,render_lowers=receipt[:,None]+offset[None])
 out=begin_run(ROOT/'power-gain-negative-control-01',dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(fit/'ports.pt'),scope='Inference-only scaling of power coefficients2/5: multipliers above1 deliberately violate the new sensitivity bound as a negative control. Same bounded checkpoint, other parameters/features/body/clocks fixed. No checkpoint overwrite, control authority, measured neural lesion or model selection.',optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));saved={};reports=[];original=read(ROOT/'held-input-probe-03/result.json');assert sha256(ROOT/'held-input-probe-03/values.npz')==original['values_sha256'];maximum_replay=0.
 try:
  with np.load(ROOT/'held-input-probe-03/values.npz') as old:
   for speed,temp in [(30.,60.)]:
    for multiplier in [0.,1.,2.,4.,8.]:
     with torch.no_grad():
      e.speed_policy.projection.weight.copy_(cp['speed_policy']['projection.weight'])
      e.speed_policy.projection.weight[0,[2,5]]*=multiplier
     body,mask,native,valid=synthetic_body(speed,temp,e.body_port.reference.numpy());e.model.reset_episode();e.holder.clear();e.holder.update(mode='synthetic',source=dict(x=x),case=0,body=body,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]),speed_policy_off=False);pred=[];commands=[]
     for i in range(n):pred.append(e.step(i,False).numpy());commands.append(e.last_command[0].numpy().copy())
     pred=np.asarray(pred);commands=np.asarray(commands);j=len(reports);saved[f'case{j}_prediction']=pred;saved[f'case{j}_command']=commands;reports.append(dict(speed_m_s=speed,temperature_c=temp,power_coefficient_multiplier=multiplier,late_action_peak_to_peak=np.ptp(pred[-51:],axis=0).tolist(),late_command_peak_to_peak=np.ptp(commands[-51:],axis=0).tolist(),final_speed_error=float(commands[-1,1]-speed),final_action=pred[-1].tolist()))
     if multiplier==1.:
      k=next(k for k,row in enumerate(original['reports']) if row['arm']=='speed_policy' and row['speed_m_s']==speed and row['temperature_c']==temp and row['speed_policy_off'] is False);maximum_replay=max(maximum_replay,float(abs(pred-old[f'case{k}_prediction']).max()))
  assert maximum_replay<2e-6;e.speed_policy.load_state_dict(cp['speed_policy']);e.verify_fixed();np.savez_compressed(out/'values.npz',**saved);write_json(out/'result.json',dict(status='completed_bounded_policy_power_gain_negative_control',reports=reports,maximum_original_on_off_replay_error=maximum_replay,values_sha256=sha256(out/'values.npz'),actor_qualified=False,checkpoint_modified=False,physical_stability_verified=False,vehicle_control_sent=False));print(reports)
 finally:e.close()
if __name__=='__main__':main()
