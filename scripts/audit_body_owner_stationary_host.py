"""Independent arithmetic and provenance audit of one stationary body-owner run."""
import base64,hashlib,json
from pathlib import Path
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256

def main():
 p=Path('D:/CodexResearchRuns/flybrain-body-owner-live-20261102-06');read=lambda x:json.loads(x.read_text(encoding='utf8'))
 r=read(p/'result.json');events=read(p/'events.json');ticker=read(p/'body-owner/ticker-events.json');native=read(p/'body-owner/native-copy.json')
 terminal=r['worker_terminal'];hash_ok=sha256(p/'events.json')==r['events_sha256'] and sha256(p/'body-owner/ticker-events.json')==terminal['events_sha256'] and sha256(p/'body-owner/native-copy.json')==terminal['native_sha256']
 first={};bad_native=0
 for row in native:
  a=row['authority'];raw=base64.b64decode(a['prefix_base64'],validate=True);bad_native+=hashlib.sha256(raw).hexdigest()!=a['prefix_sha256'];first.setdefault(a['packet_id'],a['received_monotonic_seconds'])
 published={e['capture_id']:e for e in events if e['status']=='scene_published'};outputs=[x for x in ticker if x['status']=='shadow_output'];bad_receipts=sum(first.get(x['packet'])!=x['body_receipt'] for x in outputs);bad_frames=sum(x['capture_id'] not in published or published[x['capture_id']]['render_start_lower']!=x['view_render_lowers'] or published[x['capture_id']]['publication']!=x['publication'] for x in outputs)
 current=max(x['finished']-x['body_receipt'] for x in outputs);held=max(x['finished']-x['held_body_receipt'] for x in outputs);view=max(x['finished']-min(x['view_render_lowers']) for x in outputs)
 ok=(r['status']=='observation_finished' and hash_ok and not bad_native and not bad_receipts and not bad_frames and current<=.05 and held<=.05 and view<=.15 and len(outputs)==r['published'] and r['gc_restoration']['restored'] is True and r['camera_close']['producer_terminal']['status']=='stopped' and r['visual_worker_terminal']['status']=='stopped' and terminal['status']=='stopped')
 result=dict(status='completed_stationary_host_freshness_audit' if ok else 'failed',run_result_sha256=sha256(p/'result.json'),auditor_sha256=sha256(__file__),outputs=len(outputs),scenes=len(published),output_span_seconds=outputs[-1]['finished']-outputs[0]['finished'],max_current_age_seconds=current,max_held_age_seconds=held,max_view_age_seconds=view,hash_ok=hash_ok,bad_native_hashes=bad_native,bad_body_receipts=bad_receipts,bad_scene_links=bad_frames,physical_synchronization_verified=False,actor_qualified=False,interpretation='Stationary host-receipt audit; no physical synchronization, numerical replay, physiology, training or autonomous driving claim')
 write_json(p/'independent-host-audit.json',result);print(json.dumps(result))
 if not ok:raise ValueError('Audit failed')
if __name__=='__main__':main()
