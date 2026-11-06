"""Fresh reserved texture only; reuse sealed control real/command/RGB evidence."""
import argparse,time,shutil
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.dual_reference_experiment import DualReferenceExperiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.evaluate_controlled_motion_factorial import synthetic_body
ROOT=Path('runs/dual-reference-learning-20261105')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'control-reserved-02'));a=parser.parse_args();start=time.perf_counter()
 torch.set_num_threads(1);plan=read(ROOT/'plan-01.json');assert sha256(plan['control_checkpoint'])==plan['control_checkpoint_sha256'];e=DualReferenceExperiment(plan);cp=torch.load(plan['control_checkpoint'],map_location='cpu',weights_only=True);e.calibration.load_state_dict(cp['current_calibration']);e.body_port.load_state_dict(cp['body_port']);e.speed_policy.load_state_dict(cp['speed_policy']);e.direct_gain.load_state_dict(cp['direct_gain']);e.calibration.requires_grad_(False);e.body_port.requires_grad_(False);e.speed_policy.requires_grad_(False);out=begin_run(Path(a.output),dict(program_sha256=sha256(__file__),checkpoint_sha256=plan['control_checkpoint_sha256'],reused_control_evaluation=plan['control_evaluation'],reused_control_result_sha256=sha256(Path(plan['control_evaluation'])/'result.json'),full_real_replay_reused=True,optimizer_updates=0,vehicle_control_sent=False));saved={};shutil.copyfile(__file__,out/'executed-entrypoint.py')
 try:
  source=ROOT/'reserved-stimuli-01';sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];sp=sm['plan'];assert sha256(source/'features.npz')==sr['features_sha256'] and sp['texture_seed']==plan['reserved_seed']
  with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
  item=dict(x=x,metadata=sm,plan=sp);groups=[];directions={k:0 for k in ['fine','visual_change_off','native_change_off','motion_off','speed_policy_off','direct_gain_off']};target=0;native_negative_control=0.;policy_negative_control=0.
  for k in range(0,12,2):
   case=sm['cases'][k]
   for temperature in sp['temperature_c']:
    body,mask,native,valid=synthetic_body(case['speed_mps'],temperature,e.body_port.reference.numpy());e.holder.clear();e.holder.update(mode='synthetic',source=item,case=k,body=body,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]));e.model.reset_episode()
    for i in range(sp['branch_frame']):e.step(i,False)
    prefix=snapshot(e.model);g=len(groups);row=dict(group=g,pair=k,temperature_c=temperature,prefix_sha256=state_digest(prefix),arms={});values={}
    for arm in directions:
     branches=[]
     for case_id in [k,k+1]:
      restore(e.model,prefix);e.holder.update(case=case_id,visual_change_off=arm=='visual_change_off',native_change_off=arm=='native_change_off',motion_off=arm=='motion_off',speed_policy_off=arm=='speed_policy_off',direct_gain_off=arm=='direct_gain_off');branches.append(np.asarray([e.step(i,False).numpy() for i in range(sp['branch_frame'],sp['frame_count'])]))
     p=np.asarray(branches);values[arm]=p;saved[f'group{g}_{arm}_prediction']=p;late=sp['evaluation_first_frame']-sp['branch_frame'];margin=float(-np.mean(p[1,late:,0]-p[0,late:,0])/2);directions[arm]+=int(margin>.0001);target+=int(arm=='fine' and margin>=.002);row['arms'][arm]=dict(signed_margin=margin)
    native_negative_control=max(native_negative_control,float(abs(values['fine']-values['native_change_off']).max()));policy_negative_control=max(policy_negative_control,float(abs(values['fine']-values['speed_policy_off']).max()));groups.append(row);write_json(out/'progress.json',dict(phase='reserved',groups=len(groups),elapsed_seconds=time.perf_counter()-start))
  e.verify_fixed();np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'reserved-groups.json',groups);write_json(out/'result.json',dict(status='completed_reused_control_fresh_reserved_evaluation',reserved_direction_counts=directions,reserved_target_pairs=target,groups=len(groups),predictions_sha256=sha256(out/'predictions.npz'),reserved_groups_sha256=sha256(out/'reserved-groups.json'),real_replay_reused=True,actor_qualified=False,vehicle_control_sent=False))
 except Exception as error:write_json(out/'result.json',dict(status='failed',error=type(error).__name__,reason=str(error),optimizer_updates=0,vehicle_control_sent=False));raise
 finally:e.close()

if __name__=='__main__':main()
