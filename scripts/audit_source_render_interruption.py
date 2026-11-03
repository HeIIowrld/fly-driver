"""Join finite original CSP call timing to common-QPC teacher source gaps."""
import argparse,base64,collections,json
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.ac_camera_archive import CameraArchive

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',required=True);p.add_argument('--output',required=True);p.add_argument('--timing-validator-recovery');a=p.parse_args()
 root=Path(a.recording);cr=json.loads((root/'camera/result.json').read_text(encoding='utf8'));tr=json.loads((root/'teacher/result.json').read_text(encoding='utf8'))
 run=begin_run(a.output,dict(arguments=vars(a),seed=780210,program_sha256=sha256(__file__),camera_result_sha256=sha256(root/'camera/result.json'),teacher_result_sha256=sha256(root/'teacher/result.json')))
 expected_timing_hash=cr['camera_close']['source_timing'].get('sha256')
 if a.timing_validator_recovery:
  recovery=json.loads((Path(a.timing_validator_recovery)/'result.json').read_text(encoding='utf8'))
  if (recovery['status']!='recovered_original_camera_timing_validation' or recovery['camera_result_sha256']!=sha256(root/'camera/result.json')
      or recovery['teacher_result_sha256']!=sha256(root/'teacher/result.json')):raise ValueError('Original hash-bound timing recovery differs')
  expected_timing_hash=recovery['source_timing_sha256']
 with CameraArchive(root/'camera/vision') as archive:
  ledger=archive.json('source-timing-v1.json');clock=archive.json('clock-initial.json')
  if sha256_bytes(archive.read('source-timing-v1.json'))!=expected_timing_hash:raise ValueError('Original source timing hash differs')
 rows=ledger['records'];stages=collections.defaultdict(list)
 if ledger['overflow_records'] or not ledger['flushed_after_capture']:raise ValueError('Complete original bounded ledger required')
 for row in rows:stages[row[0]].append(row)
 packets=[json.loads(s) for s in (root/'teacher/tcp-source-packets.jsonl').read_text(encoding='utf8').splitlines()]
 raw=[json.loads(base64.b64decode(x['wire_base64'])) for x in packets]
 native=np.array([x['source_native_snapshot']['qpc_started_ticks']/x['source_native_snapshot']['qpc_frequency_hz'] for x in raw])
 stats={k:dict(count=len(v),duration_p50_p95_max_ms=np.quantile([z[3]*1000 for z in v],[.5,.95,1]).tolist()) for k,v in stages.items()}
 start,end=ledger['started_clock'],ledger['capture_closed_clock'];low,high=clock['offset_lower_seconds'],clock['offset_upper_seconds']
 gaps=[]
 for i,(before,after) in enumerate(zip(native,native[1:])):
  # Only complete source intervals strictly inside measured capture; excludes
  # the deliberately synchronous terminal ledger flush after controls stop.
  if before-high<start or after-low>end:continue
  if after-before<=.05:continue
  overlaps=[z for z in rows if z[0]=='write.io.save' and z[1]+low>=before and z[2]+high<=after]
  gaps.append(dict(source_interval_qpc=[float(before),float(after)],duration_ms=float((after-before)*1000),
   sdk_to_original_tcp_receipt_ms=[(packets[j]['received_monotonic_seconds']-native[j])*1000 for j in (i,i+1)],
   fully_contained_synchronous_saves=overlaps))
 calls=stages['script.update'];outside=[b[1]-a[2] for a,b in zip(calls,calls[1:])]
 result=dict(status='completed_original_source_render_interruption_audit',source_timing_sha256=expected_timing_hash,
  tcp_archive_sha256=sha256(root/'teacher/tcp-source-packets.jsonl'),stages=stats,source_gaps_above50ms_inside_capture=gaps,
  maximum_between_lua_updates_ms=max(outside)*1000,camera_frames=cr['frames'],
  measured_source_calls_above50ms=[z for z in rows if z[3]>.05],
  tcp_collector_error=tr.get('tcp_collector_error'),source_tail_censored_by_collector_error=tr.get('tcp_collector_error') is not None,
  source_clock_bounds=clock,student_optimizer_updates=0,actuator_messages_sent=0,
  interpretation='Original SDK/QPC intervals and measured CSP calls; a collector error censors the following source gap, so zero observed complete intervals does not mean no source interruption. No inference about the OS cause of slow disk I/O. Terminal diagnostic flush excluded from active capture.')
 write_json(run/'result.json',result);print(json.dumps(result))

def sha256_bytes(value):
 import hashlib
 return hashlib.sha256(value).hexdigest()
if __name__=='__main__':main()
