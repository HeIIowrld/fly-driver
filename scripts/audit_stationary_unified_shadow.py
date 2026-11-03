"""Audit saved stationary unified shadow using original camera and native evidence."""
import argparse,hashlib,json,statistics,zipfile
from pathlib import Path
from flybrain_drive.graph import sha256


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();p=Path(a.run);r=read(p/'result.json')
 for f,k in [('events.json','events_sha256'),('ticker-events.json','ticker_events_sha256'),('native-copy.json','native_sha256')]:
  if sha256(p/f)!=r[k]:raise ValueError('Original recording hash mismatch')
 events=read(p/'events.json');ticks=read(p/'ticker-events.json');native=read(p/'native-copy.json');first={}
 for row in native:
  auth=row['authority'];first.setdefault(auth['packet_id'],auth['received_monotonic_seconds'])
 if any(first[x['packet']]!=x['body_receipt'] for x in ticks):raise ValueError('First observed native receipt mismatch')
 archive=p/'vision';bundle=read(archive/'bundle.json');manifest=read(archive/'archive.json');zp=archive/'camera.zip'
 if sha256(zp)!=bundle['sha256'] or sha256(archive/'archive.json')!=bundle['archive_manifest_sha256']:raise ValueError('Original archive mismatch')
 expected={x['file']:x for x in manifest};published={x['capture_id']:x for x in events if x['status']=='scene_published'}
 with zipfile.ZipFile(zp) as z:
  clock=json.loads(z.read('clock-initial.json'));lower=clock['offset_lower_seconds']-clock.get('host_clock_quantization_seconds',0)
  for event in events:
   capture=event['capture_id'];meta_name=f'frame-{capture:04d}.json';blob=z.read(meta_name)
   if hashlib.sha256(blob).hexdigest()!=expected[meta_name]['sha256']:raise ValueError('Original frame metadata mismatch')
   meta=json.loads(blob);views={x['view_id']:x for x in meta['views']}
   if meta['request_id']!=r['camera_request_id'] or meta['generation']!=clock['generation']:raise ValueError('Original camera identity mismatch')
   for i,name in enumerate(('front','left','right','rear')):
    view=views[name];bounds=view['render_started_clock']+lower
    if abs(bounds-event['render_start_lower'][i])>1e-8:raise ValueError('Original render lower mismatch')
    raw=z.read(view['file']);row=expected[view['file']]
    if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Original pixels mismatch')
    if event.get('view_body'):
     body=event['view_body'][i]
     if body is not None and not (first[body['packet']]==body['receipt']<bounds):raise ValueError('Prior body receipt mismatch')
 for tick in ticks:
  if tick['status']!='rejected':
   event=published[tick['capture_id']]
   if event['publication']!=tick['publication'] or event['render_start_lower']!=tick['view_render_lowers']:raise ValueError('Ticker publication mismatch')
 out=[x for x in ticks if x['status']=='shadow_output'];ages=[max((x['finished']-y)*1000 for y in x['view_render_lowers']) for x in out];held=[(x['finished']-x['held_body_receipt'])*1000 for x in out];cur=[(x['finished']-x['body_receipt'])*1000 for x in out];visual=[(x['visual_finished']-x['visual_snapshot'])*1000 for x in events];prefix=[(x['prefix_finished']-x['prefix_started'])*1000 for x in events if 'prefix_finished' in x]
 report={'outputs':len(out),'priming':sum(x['status']=='priming' for x in ticks),'rejected':sum(x['status']=='rejected' for x in ticks),'native_copies':len(native),'processed_camera_groups':len(events),'published_scenes':len(published),'first_receipts_verified':True,'original_pixel_hashes_and_clock_bounds_verified':True,'view_prior_receipts_precede_original_render_lower':True,'publication_and_original_render_bounds_verified':True,'view_age_max_at_event_finish_ms':max(ages),'held_age_max_at_event_finish_ms':max(held),'current_age_max_at_event_finish_ms':max(cur),'visual_compute_median_ms':statistics.median(visual),'visual_compute_max_ms':max(visual),'prefix_compute_median_ms':statistics.median(prefix),'prefix_compute_max_ms':max(prefix),'camera_terminal':r['camera_close']['producer_terminal']['status'],'physical_exposure_synchronization_verified':False,'vehicle_control_sent':False}
 (p/'independent-audit-v2.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
