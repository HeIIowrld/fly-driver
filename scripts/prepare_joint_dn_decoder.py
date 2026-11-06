"""Seal comparison before training or fresh test pixels are generated."""
from pathlib import Path
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read

def main():
 root=Path('runs/joint-dn-decoder-20261104');root.mkdir(exist_ok=False);plan=read('runs/functional-priority-learning-20261104/fit-plan-01.json');reserved=read('runs/paired-muscle-decoder-20261104/reserved-plan-01.json');reserved['texture_seed']=781183
 reserved['role']='fresh_reserved_joint_dn_decoder_trial';write_json(root/'reserved-plan-01.json',reserved)
 plan.update(seed=781181,maximum_logit_shift=2.,maximum_current_weight=.05,minimum_power_weight=.0001,head_learning_rate=.001,current_learning_rate=.001,epochs=1,warm_epochs=1200,warm_learning_rate=.01,physical_weight=1000.,minimum_physical_derivative=.005,maximum_same_observation_steering_change=.01,reserved_test_plan=str(root/'reserved-plan-01.json'),reserved_test_plan_sha256=sha256(root/'reserved-plan-01.json'),paired_data_plan='runs/paired-muscle-decoder-20261104/plan-01.json',paired_data_plan_sha256=sha256('runs/paired-muscle-decoder-20261104/plan-01.json'),yaw_probes='runs/paired-muscle-decoder-20261104/yaw-probes-01',yaw_probes_sha256=sha256('runs/paired-muscle-decoder-20261104/yaw-probes-01/probes.npz'),physical_cases=[0,2],comparison='rank1 320+21 parameters vs rank4 1280+21; parameter counts differ, not matched-capacity evidence',scope='Fixed measured flight topology/body and original DN projection. Bounded per-port engineering visual current calibration and full six-muscle monotone-power vehicle decoder. No actor authority.')
 new=['src/flybrain_drive/visual_dn_current_calibration.py','scripts/joint_dn_decoder_experiment.py','scripts/prepare_joint_dn_decoder.py','scripts/warm_joint_dn_decoder.py','scripts/fit_joint_dn_decoder.py','scripts/evaluate_joint_dn_decoder.py','tests/test_visual_dn_current_calibration.py'];plan['source_sha256'].update({p:sha256(p) for p in new});write_json(root/'fit-plan-01.json',plan);print(root)
if __name__=='__main__':main()
