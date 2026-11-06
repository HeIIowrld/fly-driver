"""Reuse the sealed original-guard comparator; only the dual objective is retrained."""
import shutil
from pathlib import Path
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
ROOT=Path('runs/dual-reference-learning-20261105');PRIOR=Path('runs/direct-scene-gain-learning-20261105')

def main():
 ROOT.mkdir(exist_ok=False);p=read(PRIOR/'plan-01.json');control=PRIOR/'frozen_direct-fit-01';r=read(control/'result.json');assert r['status']=='completed_direct_scene_gain_fit' and sha256(control/'ports.pt')==r['checkpoint_sha256'];parent=Path(p['direct_parent_evaluation']);er=read(parent/'result.json');assert sha256(parent/'predictions.npz')==er['predictions_sha256']
 for f in ['normalization-01.json','direct-reference-01.json']:shutil.copyfile(PRIOR/f,ROOT/f)
 p.update(parent_reference_predictions_sha256=sha256(parent/'predictions.npz'),control_checkpoint=str(control/'ports.pt'),control_checkpoint_sha256=sha256(control/'ports.pt'),control_fit=str(control),control_evaluation=str(PRIOR/'frozen_direct-evaluation-01'),control_optimizer_trajectory_sha256=r['optimizer_trajectory_sha256'],reserved_plan=str(PRIOR/'next-reserved-plan-01.json'),reserved_plan_sha256=sha256(PRIOR/'next-reserved-plan-01.json'),reserved_seed=915320317,scope='Only recording protection changes: max(original-fine,parent) relative error, existing supervision/anchor unchanged. Same .723 initialization, exact4 Adam and frozen0 direct gain, seed782011 and same395 tasks. Sealed original-guard frozen_direct is reused; no duplicate fit or full control replay. Native/speed/motion cumulative bounds are never rebased. Parent prediction reference is TRAIN-only loss truth, never an actor input.')
 p['source_sha256']={f:sha256(f) for f in ['src/flybrain_drive/dual_reference_recording_objective.py','scripts/dual_reference_experiment.py','scripts/fit_dual_reference.py','scripts/evaluate_dual_reference.py','scripts/prepare_dual_reference.py','scripts/validate_dual_reference.py']};write_json(ROOT/'plan-01.json',p);write_json(ROOT/'control-reuse-01.json',dict(control_checkpoint_sha256=r['checkpoint_sha256'],control_fit_result_sha256=sha256(control/'result.json'),control_optimizer_trajectory_sha256=r['optimizer_trajectory_sha256'],control_training_records_sha256=r['training_records_sha256'],control_evaluation_result_sha256=sha256(PRIOR/'frozen_direct-evaluation-01/result.json'),control_predictions_sha256=sha256(PRIOR/'frozen_direct-evaluation-01/predictions.npz'),control_seed=782011,new_control_fit=False,full_control_replay_reused=True))

if __name__=='__main__':main()
