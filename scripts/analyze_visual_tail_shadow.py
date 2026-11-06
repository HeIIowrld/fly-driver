"""Recompute admitted movement and phase tails from original shadow records."""
import argparse,base64,hashlib
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,decode_physics_prefix
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.shadow);r=read(root/'result.json');events=read(root/'events.json');ticker=read(root/'owner-ticker-events.json')
    assert sha256(root/'events.json')==r['events_sha256'] and sha256(root/'owner-ticker-events.json')==r['ticker_events_sha256']
    assert sha256(root/'owner-native-copy.json')==r['owner_summary']['native_sha256']
    native={}
    for row in read(root/'owner-native-copy.json'):
        auth=row['authority'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
        assert hashlib.sha256(raw).hexdigest()==auth['prefix_sha256']
        sample=decode_physics_prefix(raw[:PHYSICS_PREFIX.size],row['copy']['second_copy_end'])
        assert sample.packet_id==auth['packet_id']
        native.setdefault(sample.packet_id,sample)
    valid=[e for e in ticker if e['status']=='shadow_output'];speeds=[]
    for event in valid:
        sample=native[event['body_packet']];assert sample.received_monotonic_seconds==event['body_first_receipt']
        speeds.append(sample.fields['speedKmh'])
    percentiles=lambda values:np.percentile(values,[50,95,99,100]).tolist() if values else []
    profiles={}
    for key in ['visual_profile','prefix_profile']:
        values=[dict(capture_id=e['capture_id'],published=e.get('status')=='scene_published',**e[key]) for e in events if key in e]
        profiles[key]=dict(samples=len(values),wall_seconds_percentiles=percentiles([e['wall_seconds'] for e in values]),
            maximum_record=max(values,key=lambda e:e['wall_seconds']) if values else None,gc_collections=sum(len(e['gc_collections']) for e in values))
    result=dict(source_result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),
        admitted_outputs=len(valid),outputs_speed_at_least_2kmh=sum(v>=2 for v in speeds),maximum_output_body_speed_kmh=max(speeds) if speeds else None,
        continuous_output_span_seconds=valid[-1]['output_host_time']-valid[0]['output_host_time'] if valid else 0,
        body_age_max_seconds=max(e['output_body_age_seconds'] for e in valid) if valid else None,
        view_age_max_seconds=max(e['oldest_view_age_seconds'] for e in valid) if valid else None,
        inference_seconds_percentiles=percentiles([e['inference_seconds'] for e in valid]),
        visual_seconds_percentiles=percentiles([e['visual_seconds'] for e in events if 'visual_seconds' in e]),
        prefix_seconds_percentiles=percentiles([e['prefix_seconds'] for e in events if 'prefix_seconds' in e]),
        profiles=profiles,rejections=[e for e in ticker if e['status']=='rejected'],
        original_status=r['status'],actor_qualified=False,vehicle_control_sent=False)
    write_json(a.output,result)
    print({k:v for k,v in result.items() if k not in ['profiles','rejections']})

if __name__=='__main__':main()
