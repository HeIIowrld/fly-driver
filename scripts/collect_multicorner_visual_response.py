"""Frozen flight trajectories for reserved images; all decoder fits share these muscles."""
import time
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.sac_unified_student import read
from scripts.multicorner_command_data import ROOT,experiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.intervene_motion_spatial_same_body import state_digest
from scripts.evaluate_controlled_motion_factorial import synthetic_body

def main():
 torch.set_num_threads(1);plan=read(ROOT/'protocol-01.json');source=ROOT/'reserved-stimuli-01';sr=read(source/'result.json');sm=read(source/'manifest.json')['config'];sp=sm['plan'];assert sha256(source/'features.npz')==sr['features_sha256'] and sp['texture_seed']==781191;e=experiment(plan)
 with np.load(source/'features.npz') as z:x={k:z[k].copy() for k in z.files}
 item=dict(x=x,metadata=sm,plan=sp);out=begin_run(ROOT/'reserved-response-01',dict(program_sha256=sha256(__file__),parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],stimuli_result_sha256=sha256(source/'result.json'),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));capture=[];hook=e.head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().numpy().copy()));saved={};groups=[];start=time.perf_counter()
 try:
  for k in range(0,12,2):
   case=sm['cases'][k]
   for temperature in sp['temperature_c']:
    body,mask,native,valid=synthetic_body(case['speed_mps'],temperature,e.body_port.reference.numpy());e.holder.update(mode='synthetic',source=item,case=k,body=body,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]),calibration_off=False,motion_off=False);e.model.reset_episode()
    for i in range(sp['branch_frame']):e.step(i,False)
    prefix=snapshot(e.model);g=len(groups);row=dict(group=g,pair=k,temperature_c=temperature,prefix_sha256=state_digest(prefix))
    for arm in ['fine','calibration_off_after_prefix','motion_off_after_prefix']:
     branches=[]
     for case_id in [k,k+1]:
      restore(e.model,prefix);e.holder.update(case=case_id,calibration_off=arm=='calibration_off_after_prefix',motion_off=arm=='motion_off_after_prefix');capture.clear()
      for i in range(sp['branch_frame'],sp['frame_count']):e.step(i,False)
      branches.append(np.concatenate(capture))
     saved[f'group{g}_{arm}_muscles']=np.asarray(branches)
    # Original graph trajectory has its own original, calibration-off prefix.
    e.holder.update(case=k,calibration_off=True,motion_off=False);e.model.reset_episode()
    for i in range(sp['branch_frame']):e.step(i,False)
    original_prefix=snapshot(e.model);row['original_prefix_sha256']=state_digest(original_prefix);branches=[]
    for case_id in [k,k+1]:
     restore(e.model,original_prefix);e.holder['case']=case_id;capture.clear()
     for i in range(sp['branch_frame'],sp['frame_count']):e.step(i,False)
     branches.append(np.concatenate(capture))
    saved[f'group{g}_original_muscles']=np.asarray(branches);groups.append(row);write_json(out/'progress.json',dict(groups=len(groups),total_groups=18,elapsed_seconds=time.perf_counter()-start))
  e.verify_fixed();np.savez_compressed(out/'responses.npz',**saved);write_json(out/'groups.json',groups);write_json(out/'result.json',dict(status='completed_reserved_shared_muscle_trajectories',responses_sha256=sha256(out/'responses.npz'),groups_sha256=sha256(out/'groups.json'),groups=len(groups),late_index=sp['evaluation_first_frame']-sp['branch_frame'],seed=781191,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,termination_reason='Frozen source circuit, one new reserved texture, identical body and candidate prefix within each bend/ablation pair. Original baseline has explicit separate original prefix. All later external stateless decoders can reuse these trajectories.'))
 finally:hook.remove();e.close()
if __name__=='__main__':main()
