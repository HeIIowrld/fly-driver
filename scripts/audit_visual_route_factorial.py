"""Original unsplit experiment replay and independent factorial arithmetic."""
import gc
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.fixture_adjoint_speed_experiment import FixtureAdjointSpeedExperiment
from scripts.sac_steering_experiment import restore
ROOT=Path('runs/visual-route-factorial-20261105');PARENT=Path('runs/fixture-adjoint-learning-20261104')

def main():
 torch.set_num_threads(1);source=ROOT/'diagnostic-01';r=read(source/'result.json');assert r['status']=='completed_train_five_visual_route_factorial' and sha256(source/'trajectories.npz')==r['trajectories_sha256'];groups=read(source/'groups.json');e=FixtureAdjointSpeedExperiment(read(PARENT/'plan-01.json'),fixture_input_adjoint=True);cp=torch.load(PARENT/'input_adjoint-fit-01/ports.pt',map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);states=torch.load(PARENT/'input_adjoint-evaluation-01/candidate-prefix-states.pt',map_location='cpu',weights_only=True);out=begin_run(ROOT/'independent-audit-01',dict(program_sha256=sha256(__file__),source_result_sha256=sha256(source/'result.json'),optimizer_updates=0,vehicle_control_sent=False));maximum=0.;margins=[];enabled=gc.isenabled();gc.disable()
 try:
  with np.load(source/'trajectories.npz') as z:
   for g in groups:
    j=g['group'];task=g['task'];episode,begin,end=task['episode'],task['begin'],task['end'];values=[]
    for mask in range(32):
     p=z[f'group{j}_mask{mask}_prediction'];v=float(-(p[0,:,0]-p[1,:,0]).mean()/2);assert v==g['masks'][mask]['steering_margin'];values.append(v)
    margins.append(values);branches=[]
    for donor in [task['left_donor'],task['right_donor']]:
     restore(e.model,states[episode,begin]);e.holder.clear();e.holder.update(mode='real',episode=episode,counterfactual_images=True);branch=[]
     for offset,i in enumerate(range(begin,end)):e.holder['donor']=(episode,donor+offset);branch.append(e.step(i,False).numpy())
     branches.append(branch)
    maximum=max(maximum,float(abs(np.asarray(branches)-z[f'group{j}_mask31_prediction']).max()))
  a=np.asarray(margins);np.testing.assert_array_equal(a.mean(0),r['mean_signed_margins_by_mask']);assert maximum<2e-6;e.verify_fixed();write_json(out/'result.json',dict(status='completed_independent_five_route_audit',groups=len(groups),all_full_exchange_predictions_verified=True,maximum_original_unsplit_replay_difference=maximum,all_32_mask_margins_verified=True,vehicle_control_sent=False,actual_driving_improvement_verified=False))
 finally:e.close();gc.enable() if enabled else gc.disable()

if __name__=='__main__':main()
