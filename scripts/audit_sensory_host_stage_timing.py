"""Host-stage timing audit of a retained sensory recording; no physical time inference."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--recording',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.recording)
result=json.loads((root/'result.json').read_text(encoding='utf8'))
if result['status']!='completed' or result['training_admissible'] is not False: raise ValueError('Diagnostic completed recording required')
path=root/'client/events.jsonl';raw=path.read_bytes();events=[json.loads(x) for x in raw.splitlines()]
if len(events)!=result['frames'] or [x['index'] for x in events]!=list(range(len(events))): raise ValueError('Complete ordered events required')
series={
 'camera_receipt_to_native_receipt': [x['native']['received_monotonic_seconds']-x['host_image_receipt'] for x in events],
 'native_receipt_to_sdk_receipt': [x['sdk']['received_monotonic_seconds']-x['native']['received_monotonic_seconds'] for x in events],
 'sdk_receipt_to_decision': [x['decision_monotonic_seconds']-x['sdk']['received_monotonic_seconds'] for x in events],
 'camera_receipt_to_decision': [x['decision_monotonic_seconds']-x['host_image_receipt'] for x in events],
 'decision_spacing': np.diff([x['decision_monotonic_seconds'] for x in events]).tolist()}
if any(not np.isfinite(x).all() or np.any(np.asarray(x)<0) for x in series.values()): raise ValueError('Original host timing must be finite and ordered')
summary={k:dict(median_ms=float(np.median(v)*1000),p95_ms=float(np.quantile(v,.95)*1000),max_ms=float(max(v)*1000)) for k,v in series.items()}
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
report=dict(status='completed_host_stage_audit',events_sha256=hashlib.sha256(raw).hexdigest(),
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),observations=len(events),
 all_four_views_valid=result['all_four_views_valid_frames'],stages=summary,
 native_speed_min_kmh=min(x['native']['fields']['speedKmh'] for x in events),
 native_speed_max_kmh=max(x['native']['fields']['speedKmh'] for x in events),
 physical_capture_or_exposure_verified=False,training_admissible=False,vehicle_control_sent=False)
(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
