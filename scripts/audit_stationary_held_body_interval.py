"""Audit actual native packet arrivals across a guarded stationary shadow failure."""
import argparse, base64, hashlib, json
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();source=Path(a.source)
 result=json.loads((source/'result.json').read_text(encoding='utf8'));events_file=source/'events.json';native_file=source/'native-copy.json'
 if result['status']!='stopped_on_guard_rejection' or result['stop_reason']!='ValueError: Held body receipt stale at decision' or sha256(events_file)!=result['events_sha256'] or sha256(native_file)!=result['native_sha256']:raise ValueError('Exact verified held-body failure required')
 events=json.loads(events_file.read_text(encoding='utf8'));rows=json.loads(native_file.read_text(encoding='utf8'))
 if len(events)!=2 or events[0]['status']!='priming' or events[1]['status']!='guard_rejected':raise ValueError('Expected two ordered events required')
 t0,t1=(x['decision'] for x in events);prev=events[0]['decision_body']['receipt'];curr=events[1]['decision_body']['receipt']
 if not prev<=t0<t1 or not t0<=curr<=t1:raise ValueError('Original causal ordering required')
 selected=[];last=float('-inf')
 for row in rows:
  auth=row['authority'];received=row['copy']['second_copy_end'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
  if received!=auth['received_monotonic_seconds'] or hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256'] or received<last:raise ValueError('Original ordered native receipt and hash required')
  last=received
  if t0<received<=t1:selected.append((received,auth['packet_id']))
 if not selected:raise ValueError('Native records missing in decision interval')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_result_sha256=sha256(source/'result.json'),source_events_sha256=sha256(events_file),source_native_sha256=sha256(native_file)))
 gaps=np.diff([x[0] for x in selected]);out=dict(status='completed_read_only_native_interval_audit',decision_interval_ms=1000*(t1-t0),previous_body_age_ms=1000*(t1-prev),current_body_age_ms=1000*(t1-curr),native_copies_in_interval=len(selected),unique_packet_ids_in_interval=len(set(x[1] for x in selected)),native_copy_gap_ms_max=float(1000*gaps.max()) if len(gaps) else None,vehicle_control_sent=False,actor_qualified=False,interpretation='Observed native receipts during a single prior stationary failure; no body ticker was run, and physical sample timing is not verified.')
 write_json(run/'result.json',out);print(json.dumps(out))
if __name__=='__main__':main()
