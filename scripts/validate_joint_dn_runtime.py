"""Actual native bytes and camera binding parity for both joint candidates."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.joint_dn_decoder_candidate import load_joint_dn_decoder,JointDNDecoderDecisionRuntime
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.train_qualified_sac_unified import context

def main():
 torch.set_num_threads(1);torch.set_flush_denormal(True);root=Path('runs/joint-dn-decoder-20261104');plan=read(root/'fit-plan-01.json');candidate=read(plan['candidate_plan']);original_components,motion,parent=load_readonly_motion_candidate(candidate);original=Path(candidate['motion']['fit']);om=read(original/'manifest.json')['config'];_,_,entries,items,extras,_,_,_,_,features,maps,coarse,_,_=load_motion_experiment(om['plan'],'fine_motion');d,c,_=items[0];n=128;needed=set(map(int,d['body_packet'][:n]));cfg=read(Path(entries[0]['data'])/'manifest.json')['config'];native=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl';qr=read(Path(cfg['arguments']['qualification'])/'result.json');assert sha256(native)==qr['native_body_sha256'];samples={}
 for line in native.open(encoding='utf-8'):
  auth=json.loads(line)['authority'];pid=auth['packet_id']
  if pid in needed and pid not in samples:
   raw=base64.b64decode(auth['prefix_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==auth['prefix_sha256'];samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
  if set(samples)==needed:break
 assert set(samples)==needed;out=begin_run(root/'runtime-validation-01',dict(program_sha256=sha256(__file__),loader_sha256=sha256('src/flybrain_drive/joint_dn_decoder_candidate.py'),plan_sha256=sha256(root/'fit-plan-01.json'),parent_candidate=parent,entry=entries[0],native_body_sha256=sha256(native),actor_qualified=False,vehicle_control_sent=False));saved={};reports=[]
 try:
  for rank in [1,4]:
   components,calibration,receipt=load_joint_dn_decoder(root/f'rank{rank}-fit-01/joint-dn-decoder.pt',original_components,parent);runtime=JointDNDecoderDecisionRuntime(*components,motion_port=motion,current_calibration=calibration);predictions=[];muscles=[];ages=[]
   try:
    prepare_fixed_cpu_backend(components[0]);runtime.begin_episode(c['episode_id'])
    for i in range(n):
     j=int(maps[0][i]);obs=MotionSpatialObservation(c['episode_id'],float(d['camera_receipt'][i]),tuple(map(float,d['render_lowers'][i])),coarse[0][j],features[0][j].numpy());output=runtime.decide(decision_time=float(d['decision_time'][i]),scene_features=d['scene'][i],static_features=extras[0][i].numpy(),body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])],motion_observation=obs);predictions.append(output['prediction']);muscles.append(output['muscles']);ages.append(output['estimated_output_body_age_seconds'])
    with np.load(root/f'rank{rank}-evaluation-01/predictions.npz') as z:baseline=z['prediction_0'][:n].copy();baseline_mu=z['muscles_0'][:n].copy()
    difference=float(abs(np.asarray(predictions)-baseline).max());mu_difference=float(abs(np.asarray(muscles)-baseline_mu).max());assert difference<2e-6 and mu_difference<2e-6;saved[f'rank{rank}_prediction']=np.asarray(predictions);saved[f'rank{rank}_muscles']=np.asarray(muscles);reports.append(dict(rank=rank,events=n,maximum_prediction_difference=difference,maximum_muscle_difference=mu_difference,maximum_estimated_output_body_age_seconds=max(ages),receipt=receipt))
   finally:runtime.close()
  np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_joint_native_runtime_parity',models=reports,body_native_camera_joins_verified=True,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False))
 finally:torch.set_flush_denormal(False)
if __name__=='__main__':main()
