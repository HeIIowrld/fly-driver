"""Audit original four-view/native host chronology and archived pixels; no actuation."""
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.probe)
result=json.loads((root/'result.json').read_text());events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines() if json.loads(x)['kind']=='native']
if result['status']!='completed_low_speed_pulse_diagnostic' or not result['post_stop_verified']:raise ValueError('Stopped diagnostic required')
run=begin_run(a.output,dict(program_sha256=sha256(__file__),events_sha256=sha256(root/'events.jsonl'),bundle_sha256=result['camera_archive']['bundle_sha256'],actor=False))
ages=[];capture_ids=[];missing=0;checked={}
with CameraArchive(root/'vision') as archive:
 if archive.bundle_sha256!=result['camera_archive']['bundle_sha256']:raise ValueError('Bundle changed')
 initial=archive.json('clock-initial.json');clock=CSPClockBounds(initial['generation'],maximum_width=initial['maximum_width_seconds'],host_quantization=initial.get('host_clock_quantization_seconds',0.))
 for ping in initial['exchanges']:clock.observe(ping['sent'],ping['received'],ping['producer_clock'],initial['generation'])
 if clock.record()!=initial:raise ValueError('Clock bounds mismatch')
 for event in events:
  row=event.get('camera')
  if row is None:missing+=1;continue
  n=event['native']['received_monotonic_seconds'];idx=row['capture_id']
  if row['generation']!=clock.generation or row['received']>n or set(x['view_id'] for x in row['views'])!={'front','left','right','rear'}:raise ValueError('View ordering or receipt chronology mismatch')
  if idx not in checked:
   raw=archive.read(Path(row['file']).name)
   if hashlib.sha256(raw).hexdigest()!=row['file_sha256']:raise ValueError('Metadata hash mismatch')
   meta=json.loads(raw);mviews={x['view_id']:x for x in meta['views']}
   if meta['index']!=idx or meta['generation']!=clock.generation:raise ValueError('Metadata identity mismatch')
   for view in row['views']:
    file=Path(view['file']).name;content=archive.read(file)
    if hashlib.sha256(content).hexdigest()!=view['file_sha256']:raise ValueError('Image hash mismatch')
    with Image.open(io.BytesIO(content)) as img:rgb=np.asarray(img.convert('RGB'))
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=view['rgb_sha256']:raise ValueError('Pixel hash mismatch')
    lower=clock.render_start_lower(mviews[view['view_id']]['render_started_clock'],clock.generation)
    if abs(lower-view['render_started_lower'])>1e-9:raise ValueError('Render bound mismatch')
   checked[idx]=True
  values=[n-v['render_started_lower'] for v in row['views']]
  if not np.allclose(values,row['host_age_upper_seconds'],rtol=0,atol=1e-9) or min(values)<0 or max(values)>.15:raise ValueError('Freshness claim mismatch')
  ages.extend(values);capture_ids.append(idx)
summary=dict(status='completed_camera_native_host_audit',native_packets=len(events),camera_missing=missing,paired=len(capture_ids),
 distinct_camera_groups=len(set(capture_ids)),verified_distinct_groups=len(checked),view_age_median_ms=float(np.median(ages)*1000) if ages else None,
 view_age_max_ms=float(max(ages)*1000) if ages else None,clock_uncertainty_seconds=initial['uncertainty_seconds'],
 physical_sensor_exposure_sync_verified=False,training_admissible=False,actuator_messages_sent=0)
write_json(run/'result.json',summary);print(json.dumps(summary))
