"""Original native-byte/camera runtime neutral-path parity for new speed boundary."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.visual_flight_speed_intent import VisualFlightSpeedIntent
from flybrain_drive.turn_speed_runtime import TurnSpeedDecisionRuntime
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.train_qualified_sac_unified import context

def main():
 torch.set_num_threads(1);torch.set_flush_denormal(True);root=Path('runs/visual-turn-speed-20261104');plan=read(root/'plan-01.json');candidate=read(plan['candidate_plan']);components,motion,parent=load_readonly_motion_candidate(candidate);original=Path(candidate['motion']['fit']);om=read(original/'manifest.json')['config'];_,_,entries,items,extras,_,_,_,_,features,maps,coarse,_,_=load_motion_experiment(om['plan'],'fine_motion');d,c,_=items[0];n=128;needed=set(map(int,d['body_packet'][:n]));cfg=read(Path(entries[0]['data'])/'manifest.json')['config'];native=Path(cfg['arguments']['recording'])/'camera/native-body.jsonl';qr=read(Path(cfg['arguments']['qualification'])/'result.json');assert sha256(native)==qr['native_body_sha256'];samples={}
 for line in native.open(encoding='utf-8'):
  auth=json.loads(line)['authority'];pid=auth['packet_id']
  if pid in needed and pid not in samples:
   raw=base64.b64decode(auth['prefix_base64']);assert hashlib.sha256(raw).hexdigest()==auth['prefix_sha256'];samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],auth['received_monotonic_seconds'])
  if set(samples)==needed:break
 speed=VisualFlightSpeedIntent(components[2].reference,components[2].scale,components[4].reference,components[4].scale);out=begin_run(root/'runtime-neutral-01',dict(program_sha256=sha256(__file__),plan_sha256=sha256(root/'plan-01.json'),parent_candidate=parent,entry=entries[0],native_body_sha256=sha256(native),actor_qualified=False,vehicle_control_sent=False));runtime=None;predictions=[];targets=[];result=dict(status='failed',events=0,optimizer_updates=0,vehicle_control_sent=False)
 try:
  runtime=TurnSpeedDecisionRuntime(*components,motion_port=motion,speed_intent=speed,speed_active=True);prepare_fixed_cpu_backend(components[0]);runtime.begin_episode(c['episode_id'])
  for i in range(n):
   j=int(maps[0][i]);obs=MotionSpatialObservation(c['episode_id'],float(d['camera_receipt'][i]),tuple(map(float,d['render_lowers'][i])),coarse[0][j],features[0][j].numpy());output=runtime.decide(decision_time=float(d['decision_time'][i]),scene_features=d['scene'][i],static_features=extras[0][i].numpy(),body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])],motion_observation=obs);predictions.append(output['prediction']);targets.append(output['desired_speed_m_s']);assert output['speed_correction_m_s']==0. and output['native_gain']==1.
  with np.load(original/'predictions.npz') as z:baseline=z['prediction_0'][:n].copy()
  difference=float(abs(np.asarray(predictions)-baseline).max());assert difference<2e-6;np.testing.assert_array_equal(np.asarray(targets),d['body'][:n,16].astype(np.float32));np.savez_compressed(out/'predictions.npz',prediction=np.asarray(predictions),target=np.asarray(targets));result.update(status='completed_native_runtime_speed_neutral_parity',events=n,maximum_prediction_difference=difference,body_native_camera_join_verified=True,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,actual_driving_improvement_verified=False)
 except Exception as error:result.update(error=type(error).__name__,reason=str(error),events=len(predictions))
 finally:
  if runtime is not None:runtime.close()
  torch.set_flush_denormal(False)
 write_json(out/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
