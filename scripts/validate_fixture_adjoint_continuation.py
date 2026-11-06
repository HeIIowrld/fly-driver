"""Verify same numeric continuation and exact four-group Adam carry without training."""
from pathlib import Path
import torch
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.fixture_adjoint_speed_experiment import FixtureAdjointSpeedExperiment
ROOT=Path('runs/power-bounded-speed-learning-20261104')
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');fit=ROOT/'speed_policy-fit-01';cp=fit/'ports.pt';adam=fit/'optimizer-state.pt';r=read(fit/'result.json');p.update(adjoint_parent_checkpoint=str(cp),adjoint_parent_checkpoint_sha256=sha256(cp),adjoint_parent_optimizer=str(adam),adjoint_parent_optimizer_sha256=sha256(adam),adjoint_initial_optimizer_steps=r['final_optimizer_steps']);out=begin_run(ROOT/'input-adjoint-continuation-01',dict(program_sha256=sha256(__file__),experiment_sha256=sha256('scripts/fixture_adjoint_speed_experiment.py'),parent_checkpoint_sha256=sha256(cp),parent_optimizer_sha256=sha256(adam),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));reports=[];source=torch.load(adam,map_location='cpu',weights_only=True)
 for mode in [False,True]:
  e=FixtureAdjointSpeedExperiment(p,fixture_input_adjoint=mode)
  try:
   optimizer=e.make_optimizer();parameters=e.train_parameters()
   for j,w in enumerate(parameters):
    old=source['state'][source['param_groups'][j]['params'][0]]
    for k in ['step','exp_avg','exp_avg_sq']:assert torch.equal(optimizer.state[w][k],old[k])
   e.model.reset_episode();e.holder.update(mode='real',episode=0);prediction=np.asarray([e.step(i,False).numpy() for i in range(128)])
   with np.load(ROOT/'speed_policy-evaluation-01/predictions.npz') as z:error=float(abs(prediction-z['prediction_0'][:128]).max())
   assert error==0;e.verify_fixed();reports.append(dict(fixture_input_adjoint=mode,events=128,maximum_prediction_difference=error,optimizer_steps=[int(optimizer.state[w]['step']) for w in parameters],all_four_adam_moments_exact=True))
  finally:e.close()
 write_json(out/'result.json',dict(status='completed_fixture_adjoint_four_group_continuation_validation',reports=reports,optimizer_updates=0,new_driver_training_completed=False,actor_qualified=False,vehicle_control_sent=False));print(reports)
if __name__=='__main__':main()
