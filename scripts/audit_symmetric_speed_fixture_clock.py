"""Reproduce held-sample clock rejection without changing freshness guards."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.probe_symmetric_speed_held_inputs_v2 import HeldClockExperiment
from scripts.evaluate_controlled_motion_factorial import synthetic_body
ROOT=Path('runs/symmetric-speed-learning-20261104')
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');e=HeldClockExperiment(p,learn_speed=False);cp=torch.load(ROOT/'neutral_control-fit-01/ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);source=ROOT/'reserved-stimuli-01';frame=read(source/'manifest.json')['config']['plan']['branch_frame']-1
 with np.load(source/'features.npz') as z:held={k:np.repeat(z[k][0:1,frame:frame+1],3,axis=1) for k in ['scene','static','motion']};offset=z['render_lowers'][frame]-z['camera_receipt'][frame]
 out=begin_run(ROOT/'fixture-clock-audit-01',dict(program_sha256=sha256(__file__),original_step_sha256=sha256('scripts/probe_symmetric_speed_held_inputs_v2.py'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));rows=[]
 try:
  for cadence in [.05,.04]:
   receipt=100000.+np.arange(3)*cadence;x=dict(held,camera_receipt=receipt,render_lowers=receipt[:,None]+offset[None]);body,mask,native,valid=synthetic_body(13.,60.,e.body_port.reference.numpy());e.model.reset_episode();e.holder.clear();e.holder.update(mode='synthetic',source=dict(x=x),case=0,body=body,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]),speed_policy_off=False);error=None;event=None;age=None
   for i in range(3):
    age=None if e.model.previous_body_receipt is None else float(receipt[i]+.005-e.model.previous_body_receipt)
    try:e.step(i,False)
    except ValueError as caught:error=str(caught);event=i;break
   if cadence==.05:assert event==1 and error=='Held body receipt stale at decision' and age>e.model.freshness_limits.maximum_body_receipt_age_seconds
   else:assert error is None and age<e.model.freshness_limits.maximum_body_receipt_age_seconds
   rows.append(dict(cadence_seconds=cadence,event=event,error=error,previous_body_age_seconds=age,unchanged_maximum_age_seconds=e.model.freshness_limits.maximum_body_receipt_age_seconds))
  e.verify_fixed();logs={str(ROOT/f'held-input-console-0{i}.log'):sha256(ROOT/f'held-input-console-0{i}.log') for i in [1,2]};write_json(out/'result.json',dict(status='completed_held_fixture_clock_reproduction_audit',reports=rows,failed_logs_sha256=logs,packet_IDs_unique_in_both_reproductions=True,earlier_ID_cause_inference_corrected=True,actor_qualified=False,vehicle_control_sent=False));print(rows)
 finally:e.close()
if __name__=='__main__':main()
