"""Audit recorded CSP simulator and car timestamps; no physical-tick claim."""
import argparse,json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.source);receipt=read(root/'result.json');run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_result_sha256=sha256(root/'result.json'),actor=False,diagnostic_only=True));result=dict(status='failed',actor_deployed=False,optimizer_updates=0,vehicle_control_sent=False)
 try:
  p=root/'diagnostic-frames.json'
  if receipt['status']!='completed' or sha256(p)!=receipt['metadata_sha256']:raise ValueError('completed exact source required')
  rows=read(p);age=[];simchange=[];physchange=[];speed=[]
  for i,row in enumerate(rows):
   path=Path(row['source_file'])
   if row['index']!=i or sha256(path)!=row['sha256']:raise ValueError('frame identity or hash mismatch')
   frame=read(path);d=frame['wheel_geometry_diagnostic']
   if d!=row['wheel_geometry'] or d['actor_admissible'] is not False or d['source_verified'] is not False:raise ValueError('original unverified diagnostic required')
   age.append(float(d['sim_time_before_ms'])-float(d['car_physics_timestamp_ms']));simchange.append(float(d['sim_time_after_ms'])-float(d['sim_time_before_ms']));physchange.append(float(d['car_physics_timestamp_after_ms'])-float(d['car_physics_timestamp_ms']));speed.append(float(np.linalg.norm(row['diagnostic']['vehicle_velocity_candidate_m_s'])))
  age,simchange,physchange,speed=map(np.asarray,(age,simchange,physchange,speed))
  if len(age)<20 or not all(np.isfinite(v).all() for v in (age,simchange,physchange,speed)):raise ValueError('finite sufficient frames required')
  result.update(status='completed_csp_source_clock_audit',frames=len(age),moving_frames_1ms=int((speed>=1).sum()),max_speed_m_s=float(speed.max()),sim_minus_physics_ms_min=float(age.min()),sim_minus_physics_ms_max=float(age.max()),sim_before_after_ms_max=float(np.abs(simchange).max()),physics_before_after_ms_max=float(np.abs(physchange).max()),same_sim_and_physics_count=int((age==0).sum()),source_physics_time_verified=False,native_tick_verified=False,training_admissible=False,interpretation='Same-call CSP fields; exact equality may be shared or copied time and is not evidence of native sampling tick or camera exposure synchronization.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_csp_source_clock_audit':raise SystemExit(1)
if __name__=='__main__':main()
