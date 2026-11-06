"""Constant-feature/body recurrent stress; synthetic inputs, not driving stability."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.power_bounded_speed_experiment import PowerBoundedSpeedExperiment
from scripts.evaluate_controlled_motion_factorial import synthetic_body
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
ROOT=Path('runs/power-bounded-speed-learning-20261104')
class HeldClockExperiment(PowerBoundedSpeedExperiment):
 def step(self,i,learn):
  if learn:raise ValueError('Held input fixture is diagnostic only')
  h=self.holder;h['index']=i;x=h['source']['x'];clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005;values,mask=h['body'],h['mask'];current=NavigationBodyObservation('synthetic-factorial',2000000+5*i+4,now-.001,now,values.copy(),mask.copy(),self.contract);history=tuple(NavigationBodyObservation('synthetic-factorial',2000000+5*i+j,t-.001,camera,values.copy(),mask.copy(),self.contract) for j,t in enumerate(clocks));ctx=per_view_temporal_body_context(history,current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now)
  with torch.no_grad():
   _,mu=self.model.experimental_recurrent_speed_decision(decision_time=now,racing_goals=self.goals,scene_features=torch.tensor(x['scene'][0,i:i+1]),scene_valid=self.views,body_context=ctx);return self.head(mu)[0]

def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');source=ROOT/'reserved-stimuli-01';sr=read(source/'result.json');assert sha256(source/'features.npz')==sr['features_sha256'];sm=read(source/'manifest.json')['config'];frame=sm['plan']['branch_frame']-1;n=301
 with np.load(source/'features.npz') as z:
  receipt=100000.+np.arange(n)*.04;offset=z['render_lowers'][frame]-z['camera_receipt'][frame];x={k:np.repeat(z[k][0:1,frame:frame+1],n,axis=1) for k in ['scene','static','motion']};x.update(camera_receipt=receipt,render_lowers=receipt[:,None]+offset[None])
 out=begin_run(ROOT/'held-input-probe-01',dict(program_sha256=sha256(__file__),features_sha256=sr['features_sha256'],source_frame=frame,decisions=n,interval_seconds=.04,scope='Held synthetic straight-prefix features and constant declared body/native at3 speeds x2 temperatures. Zero state at each independent episode, advancing clocks. Full versus speed-policy-off. No retimed real RGB, gradient, physical stability qualification or tyre dynamics.',optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));saved={};reports=[]
 for arm in ['neutral_control','speed_policy']:
  e=HeldClockExperiment(p,learn_speed=arm=='speed_policy');fit=ROOT/f'{arm}-fit-01';assert sha256(fit/'ports.pt')==read(fit/'result.json')['checkpoint_sha256'];cp=torch.load(fit/'ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy'])
  try:
   for speed in [13.,30.,60.]:
    for temp in [60.,100.]:
     for off in [False,True]:
      body,mask,native,valid=synthetic_body(speed,temp,e.body_port.reference.numpy());e.model.reset_episode();e.holder.clear();e.holder.update(mode='synthetic',source=dict(x=x),case=0,body=body,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]),speed_policy_off=off);pred=[];commands=[];muscles=[]
      for i in range(n):
       try:pred.append(e.step(i,False).numpy());commands.append(e.last_command[0].numpy().copy());muscles.append(e.last_muscles[0].numpy().copy())
       except Exception as error:
        write_json(out/'result.json',dict(status='failed',event=i,arm=arm,speed_m_s=speed,temperature_c=temp,previous_body_receipt=e.model.previous_body_receipt,decision_time=float(receipt[i]+.005),error=type(error).__name__,reason=str(error),actor_qualified=False,vehicle_control_sent=False));raise
      pred=np.asarray(pred);commands=np.asarray(commands);muscles=np.asarray(muscles);assert np.isfinite(pred).all() and np.isfinite(commands).all() and np.isfinite(muscles).all();j=len(reports);saved[f'case{j}_prediction']=pred;saved[f'case{j}_command']=commands;saved[f'case{j}_muscles']=muscles;reports.append(dict(arm=arm,speed_m_s=speed,temperature_c=temp,speed_policy_off=off,late_action_peak_to_peak=np.ptp(pred[-51:],axis=0).tolist(),late_command_peak_to_peak=np.ptp(commands[-51:],axis=0).tolist(),late_muscle_peak_to_peak=np.ptp(muscles[-51:],axis=0).tolist(),last_step_action_delta=abs(pred[-1]-pred[-2]).tolist(),final_action=pred[-1].tolist(),final_speed_error=float(commands[-1,1]-speed)))
   e.verify_fixed()
  finally:e.close()
 np.savez_compressed(out/'values.npz',**saved);write_json(out/'result.json',dict(status='completed_held_symmetric_speed_input_stress',reports=reports,values_sha256=sha256(out/'values.npz'),actor_qualified=False,physical_stability_verified=False,actual_tyre_adaptation_verified=False,vehicle_control_sent=False));print(max(max(v['late_action_peak_to_peak']) for v in reports))
if __name__=='__main__':main()
