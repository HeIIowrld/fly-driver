"""Seal TRAIN-only decoder fit and fresh component stimulus plans."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.evidence import write_json

def main():
 torch.set_num_threads(1);root=Path('runs/paired-muscle-decoder-20261104');candidate='runs/motion-live-integration-20261104/candidate-plan-03.json';components,motion,receipt=load_readonly_motion_candidate(read(candidate));fit=Path(read(candidate)['motion']['fit']);result=read(fit/'result.json');assert sha256(fit/'predictions.npz')==result['predictions_sha256'];parity=[]
 with np.load(fit/'predictions.npz') as z:
  with torch.no_grad():
   for e in range(3):parity.append(float(abs(components[1](torch.tensor(z[f'muscles_{e}'])).numpy()-z[f'prediction_{e}']).max()))
 assert max(parity)<2e-6 and components[0].speed_command_boundary is None and all(v is not components[1] for v in components[0].modules())
 plan=dict(candidate_plan=candidate,candidate_plan_sha256=sha256(candidate),parent_candidate=receipt,parent_fine_fit=str(fit),muscle_trajectories_sha256=sha256(fit/'predictions.npz'),parent_full_runtime_audit=read(candidate)['motion']['audit'],parent_cached_readout_parity=parity,readout_not_fed_into_parent_recurrence=True,source_sha256={f:sha256(f) for f in ['src/flybrain_drive/paired_flight_vehicle_readout.py','scripts/paired_decoder_data.py','scripts/fit_paired_vehicle_decoder.py','scripts/train_qualified_sac_unified.py','scripts/fit_bounded_sac_steering.py','src/flybrain_drive/sac_unified_student.py']},functional_probes='runs/visual-turn-speed-20261104/speed-sign-fd-02',functional_probes_sha256=sha256('runs/visual-turn-speed-20261104/speed-sign-fd-02/probes.npz'),functional_training_cases=[0,2],functional_diagnostic_cases=[1,3],minimum_longitudinal_derivative=.005,functional_weight=10.,normalization_floor=1e-6,minimum_power_gain=.02,maximum_gain=10.,initial_ridge=1e-4,learning_rate=.01,epochs=1200,seed=781167,train_episodes=[0,1],development_episode=2,selection='Minimum TRAIN teacher+functional objective; same initial weights for both arms, no development or fresh stimulus selection',scope='15-parameter stateless engineering six-muscle decoder. Monotone versus free power gains, paired odd steer/even pedals in both. Fixed original measured/sensory graph; full cached muscle trajectories reused exactly because decoder has no recurrence feedback. No physiological/physical driving qualification.',actor_qualified=False,vehicle_control_sent=False,source_revision_observed_before_submission=subprocess.check_output(['git','-c','safe.directory='+str(Path.cwd()).replace('\\','/'),'rev-parse','HEAD'],text=True).strip())
 stimulus=read('runs/visual-turn-speed-20261104/reserved-plan-01.json');stimulus.update(texture_seed=781163,role='fresh_reserved_paired_vehicle_decoder_trial');write_json(root/'reserved-plan-01.json',stimulus);plan['reserved_test_plan']=str(root/'reserved-plan-01.json');plan['reserved_test_plan_sha256']=sha256(root/'reserved-plan-01.json');write_json(root/'plan-01.json',plan);print(dict(parent_cached_readout_parity=parity,parameters=15,epochs=1200))
if __name__=='__main__':main()
