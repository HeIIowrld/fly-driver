"""Read-only bounded SDK-name vs native-array comparison, no control."""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,repeated_snapshot
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text(encoding='utf8'));run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),diagnostic_only=True,actor=False));r=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,optimizer_updates=0);mapping=None
 try:
  publisher=Path(plan['publisher_source'])
  if sha256(publisher)!=plan['publisher_sha256']:raise ValueError('publisher source changed')
  mapping=ExistingNamedMapping('Local\\acpmf_physics',PHYSICS_PREFIX.size);rows=[];errors=[];last=None
  for i in range(plan['attempts']):
   try:
    before=time.perf_counter();telemetry=Path(plan['telemetry_path']);stat_before=telemetry.stat();t=json.loads(telemetry.read_text(encoding='utf8'));stat_after=telemetry.stat();s=repeated_snapshot(mapping);after=time.perf_counter();age=time.time()-stat_after.st_mtime
    if stat_before.st_mtime_ns != stat_after.st_mtime_ns:raise ValueError('publisher replaced file during read')
    names=('fl_wheel_load','fr_wheel_load','rl_wheel_load','rr_wheel_load');sdk=np.array([float(t[k]) for k in names]);native=np.array(s.fields['wheelLoad'],dtype=float);speed=float(s.fields['speedKmh'])/3.6
    if not np.isfinite(sdk).all() or not np.isfinite(native).all() or not 0<=age<=plan['maximum_host_age_s'] or speed>plan['maximum_speed_m_s']:raise ValueError('stale, moving or nonfinite reading')
    if t.get('producer_session_id') is None:raise ValueError('publisher session missing')
    if last is not None and t['producer_session_id']!=last:raise ValueError('publisher restarted')
    last=t['producer_session_id'];rows.append(dict(sdk=sdk.tolist(),native=native.tolist(),native_packet=int(s.packet_id),publisher_steps=int(t['steps']),host_age_s=age,read_span_s=after-before,speed_m_s=speed,native_prefix_sha256=hashlib.sha256(s.prefix_bytes).hexdigest()))
   except (OSError,ValueError,KeyError) as e:errors.append(type(e).__name__+': '+str(e))
   time.sleep(plan['interval_s'])
  if len(rows)<plan['minimum_samples'] or len({x['publisher_steps'] for x in rows})<plan['minimum_samples']:raise ValueError('insufficient fresh distinct stationary samples')
  sdk=np.array([x['sdk'] for x in rows]);native=np.array([x['native'] for x in rows]);scores=[]
  for perm in itertools.permutations(range(4)):
   diff=np.abs(sdk-native[:,perm]);scores.append(dict(native_indices=list(perm),mean_abs_error=float(diff.mean()),max_abs_error=float(diff.max())))
  scores.sort(key=lambda x:x['mean_abs_error']);best=scores[0];r.update(status='completed_live_sdk_native_wheel_comparison',samples=rows,errors=errors,host_freshness_basis='latest-file mtime and advancing publisher steps, wall time only',permutations=scores,best=best,second=scores[1],identity_is_best=best['native_indices']==[0,1,2,3],source_time_verified=False,physical_wheel_geometry_verified=False,wheel_order_registered=False,training_admissible=False,interpretation='SDK output was labeled FL/FR/RL/RR by installed publisher from the vector returned by ac.getCarState Load; this does not independently verify physical wheel labels or sampling tick. Stationary host freshness only.')
 except Exception as e:r.update(error=type(e).__name__,reason=str(e))
 finally:
  if mapping is not None:mapping.close()
  write_json(run/'result.json',r)
 print(json.dumps(r,allow_nan=False))
 if r['status']!='completed_live_sdk_native_wheel_comparison':raise SystemExit(1)
if __name__=='__main__':main()
