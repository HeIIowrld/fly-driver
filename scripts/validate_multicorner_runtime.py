"""Actual native-byte/camera execution of float64 steering and fixed float32 circuit."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.joint_dn_decoder_candidate import load_joint_dn_decoder,JointDNDecoderDecisionRuntime
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.train_qualified_sac_unified import context
from scripts.multicorner_command_data import ROOT

def main():
 torch.set_num_threads(1);torch.set_flush_denormal(True);protocol=read(ROOT/'protocol-01.json');plan=read(protocol['parent_fit_plan']);candidate=read(plan['candidate_plan']);components,motion,parent=load_readonly_motion_candidate(candidate);components,calibration,receipt=load_joint_dn_decoder(protocol['parent_checkpoint'],components,parent);old=Path(candidate['motion']['fit']);om=read(old/'manifest.json')['config'];_,_,entries,items,extras,_,_,_,_,features,maps,coarse,_,_=load_motion_experiment(om['plan'],'fine_motion');d,c,_=items[0];n=128;needed=set(map(int,d['body_packet'][:n]));cfg=read(Path(entries[0]['data'])/'manifest.json')['config'];native=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl';qr=read(Path(cfg['arguments']['qualification'])/'result.json');assert sha256(native)==qr['native_body_sha256'];samples={}
 for line in native.open(encoding='utf-8'):
  auth=json.loads(line)['authority'];pid=auth['packet_id']
  if pid in needed and pid not in samples:
   raw=base64.b64decode(auth['prefix_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==auth['prefix_sha256'];samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
  if set(samples)==needed:break
 assert set(samples)==needed;out=begin_run(ROOT/'runtime-validation-01',dict(program_sha256=sha256(__file__),loader_sha256=sha256('src/flybrain_drive/muscle_interaction_candidate.py'),parent_receipt=receipt,native_body_sha256=sha256(native),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));saved={};reports=[]
 try:
  for variant in ['affine','quadratic']:
   head,hr=load_muscle_interaction_head(ROOT/f'{variant}-fit-01/decoder.pt');changed=(components[0],head,*components[2:]);runtime=JointDNDecoderDecisionRuntime(*changed,motion_port=motion,current_calibration=calibration);pred=[];muscles=[];ages=[]
   try:
    prepare_fixed_cpu_backend(components[0]);runtime.begin_episode(c['episode_id'])
    for i in range(n):
     j=int(maps[0][i]);obs=MotionSpatialObservation(c['episode_id'],float(d['camera_receipt'][i]),tuple(map(float,d['render_lowers'][i])),coarse[0][j],features[0][j].numpy());r=runtime.decide(decision_time=float(d['decision_time'][i]),scene_features=d['scene'][i],static_features=extras[0][i].numpy(),body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])],motion_observation=obs);pred.append(r['prediction']);muscles.append(r['muscles']);ages.append(r['estimated_output_body_age_seconds'])
    with np.load(ROOT/f'{variant}-evaluation-01/predictions.npz') as z:expected=z['prediction_0'][:n].copy()
    with np.load(ROOT/'episode0-probes-01/neutral-full.npz') as z:expected_mu=z['muscles'][:n].copy()
    pdiff=float(abs(np.asarray(pred)-expected).max());mdiff=float(abs(np.asarray(muscles)-expected_mu).max());assert pdiff<2e-6 and mdiff<2e-6;saved[variant+'_prediction']=np.asarray(pred);reports.append(dict(variant=variant,events=n,maximum_prediction_difference=pdiff,maximum_muscle_difference=mdiff,maximum_estimated_output_body_age_seconds=max(ages),receipt=hr))
   finally:runtime.close()
  np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_multicorner_native_runtime_validation',models=reports,body_native_camera_joins_verified=True,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False))
 finally:torch.set_flush_denormal(False)
if __name__=='__main__':main()
