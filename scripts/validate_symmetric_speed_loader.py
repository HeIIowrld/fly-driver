"""Real checkpoint loader reconstruction without simulation or actuator access."""
from pathlib import Path
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.joint_dn_decoder_candidate import load_joint_dn_decoder
from flybrain_drive.symmetric_speed_candidate import apply_symmetric_speed
ROOT=Path('runs/symmetric-speed-learning-20261104')
def main():
 torch.set_num_threads(1);p=read(ROOT/'plan-01.json');candidate=read(read(read(p['base_port_plan'])['parent_fit_plan'])['candidate_plan']);components,motion,parent=load_readonly_motion_candidate(candidate);components,cal,receipt=load_joint_dn_decoder(p['parent_checkpoint'],components,parent);before={k:v.clone() for k,v in components[4].state_dict().items()};out=begin_run(ROOT/'loader-smoke-01',dict(program_sha256=sha256(__file__),loader_sha256=sha256('src/flybrain_drive/symmetric_speed_candidate.py'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));reports=[]
 for arm in ['neutral_control','speed_policy']:
  path=ROOT/f'{arm}-fit-01/ports.pt';changed,current,r=apply_symmetric_speed(path,components,cal,motion);assert components[0].speed_command_boundary is None and changed[0] is not components[0] and changed[4] is not components[4];assert all(torch.equal(v,components[4].state_dict()[k]) for k,v in before.items());assert not any(v.requires_grad for v in changed[0].speed_command_boundary.parameters());reports.append(dict(arm=arm,checkpoint_sha256=sha256(path),caller_model_and_body_unchanged=True,receipt=r))
 write_json(out/'result.json',dict(status='completed_symmetric_speed_real_loader_reconstruction',models=reports,actor_qualified=False,vehicle_control_sent=False));print('Exact real checkpoints reconstructed; callers unchanged.')
if __name__=='__main__':main()
