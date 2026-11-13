"""Exact current CPU owner parity; original recorded clocks, no vehicle control."""
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.current_flight_owner import CurrentFlightOwnerProcess
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.joint_dn_decoder_candidate import load_joint_dn_decoder,JointDNDecoderDecisionRuntime
from flybrain_drive.fixture_adjoint_speed_candidate import apply_fixture_adjoint_speed,PowerBoundedSpeedDecisionRuntime
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.train_qualified_sac_unified import context

ROOT=Path('runs/fixture-adjoint-learning-20261104')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
 torch.set_num_threads(1);torch.set_flush_denormal(True);plan=read(ROOT/'plan-01.json');oldplan=read(read(plan['base_port_plan'])['parent_fit_plan']);candidate=read(oldplan['candidate_plan']);components,motion,parent=load_readonly_motion_candidate(candidate);components,calibration,receipt=load_joint_dn_decoder(plan['parent_checkpoint'],components,parent);baseline_body={k:v.clone() for k,v in components[4].state_dict().items()};baseline_calibration={k:v.clone() for k,v in calibration.state_dict().items()};fit=Path(candidate['motion']['fit']);cfg=read(fit/'manifest.json')['config'];_,_,entries,items,extras,_,_,_,_,features,maps,coarse,_,_=load_motion_experiment(cfg['plan'],'fine_motion');d,c,_=items[0];n=128;needed=set(map(int,d['body_packet'][:n]));data=read(Path(entries[0]['data'])/'manifest.json')['config'];native=Path(data['arguments']['recording'])/'camera/native-body.jsonl';qualification=read(Path(data['arguments']['qualification'])/'result.json');assert sha256(native)==qualification['native_body_sha256'];samples={}
 for line in native.open(encoding='utf-8'):
  authority=json.loads(line)['authority'];pid=authority['packet_id']
  if pid in needed and pid not in samples:
   raw=base64.b64decode(authority['prefix_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==authority['prefix_sha256'];samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],authority['received_monotonic_seconds'])
  if set(samples)==needed:break
 assert set(samples)==needed;out=begin_run(Path(args.output),dict(program_sha256=sha256(__file__),loader_sha256=sha256('src/flybrain_drive/fixture_adjoint_speed_candidate.py'),parent_receipt=receipt,native_body_sha256=sha256(native),seed=None,optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False));saved={};reports=[]
 try:
  for arm in ['input_adjoint']:
   checkpoint=ROOT/f'{arm}-fit-01/ports.pt';owner_dir=out/arm;owner_dir.mkdir()
   runtime=CurrentFlightOwnerProcess(checkpoint=checkpoint,checkpoint_sha256=sha256(checkpoint),directory=owner_dir,python_executable=str(Path.home() / 'AppData/Local/Programs/Python/Python312/python.exe'),library_directory=str(Path.home() / 'AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages'));pred=[];muscles=[];ages=[]
   try:
    runtime.begin_replay(c['episode_id'])
    for i in range(n):
     j=int(maps[0][i]);obs=MotionSpatialObservation(c['episode_id'],float(d['camera_receipt'][i]),tuple(map(float,d['render_lowers'][i])),coarse[0][j],features[0][j].numpy());r=runtime.replay_decide(decision_time=float(d['decision_time'][i]),scene_features=d['scene'][i],static_features=extras[0][i].numpy(),body_context=context(d,c,i),native_sample=samples[int(d['body_packet'][i])],motion_observation=obs);pred.append(r['prediction']);muscles.append(r['muscles']);ages.append(r['estimated_output_body_age_seconds'])
    with np.load(ROOT/f'{arm}-evaluation-01/predictions.npz') as z:expected=z['prediction_0'][:n].copy();expected_mu=z['muscles_0'][:n].copy()
    pdiff=float(abs(np.asarray(pred)-expected).max());mdiff=float(abs(np.asarray(muscles)-expected_mu).max());assert pdiff<2e-6 and mdiff<2e-6;saved[arm+'_prediction']=np.asarray(pred);reports.append(dict(arm=arm,events=n,maximum_prediction_difference=pdiff,maximum_muscle_difference=mdiff,maximum_estimated_output_body_age_seconds=max(ages),receipt=runtime.ready,request_duration_max_seconds=max(v['finished']-v['started'] for v in runtime.events if v['kind']=='replay_decide')))
   finally:runtime.stop()
  np.savez_compressed(out/'predictions.npz',**saved);write_json(out/'result.json',dict(status='completed_current_flight_owner_recorded_validation',models=reports,body_native_camera_joins_verified=True,predictions_sha256=sha256(out/'predictions.npz'),actor_qualified=False,vehicle_control_sent=False,actual_driving_improvement_verified=False))
 finally:torch.set_flush_denormal(False)
if __name__=='__main__':main()
