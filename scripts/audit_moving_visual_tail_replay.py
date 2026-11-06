"""Independent retained-pixel binding and feature/phase evidence audit."""
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.replay);r=read(root/'result.json');m=read(root/'manifest.json')['config'];source=Path(m['arguments']['recording'])
    assert r['status']=='completed_retained_moving_visual_phase_replay'
    for file,key in [('features.npz','features_sha256'),('phase-records.json','phase_records_sha256')]:assert sha256(root/file)==r[key]
    assert sha256(root/'executed-entrypoint.py')==m['program_sha256']
    assert sha256(source/'result.json')==m['source_result_sha256'] and sha256(source/'events.json')==m['source_events_sha256']
    original=[e for e in read(source/'events.json') if 'visual_seconds' in e];ids=[e['capture_id'] for e in original]
    with CameraArchive(source/'vision') as archive:
        assert archive.bundle_sha256==m['camera_bundle_sha256']
        rows={row['capture_id']:row for row in map(json.loads,archive.read('stream.jsonl').decode().splitlines())}
        for event in original:
            row=rows[event['capture_id']];frame=archive.json(Path(row['file']).name)
            assert row['received']==event['camera_received']
            ordered={v['view_id']:v for v in row['views']}
            assert [ordered[n]['render_started_lower'] for n in VIEW_ORDER]==event['render_start_lowers']
            for view in frame['views']:
                rgb=np.asarray(Image.open(io.BytesIO(archive.read(Path(view['file']).name))).convert('RGB'))
                assert rgb.shape==(180,320,3) and np.any(rgb)
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==ordered[view['view_id']]['rgb_sha256']
    differences={}
    with np.load(root/'features.npz') as arrays:
        for mode in ['observed_gc_enabled','observed_gc_deferred']:
            differences[mode]={}
            for field,size in [('scene',576),('static',320)]:
                ref=arrays['reference_'+field];value=arrays[mode+'_'+field]
                assert ref.shape==value.shape==(len(ids),size) and np.isfinite(value).all()
                delta=float(abs(ref-value).max());assert delta<=2e-6
                differences[mode][field]=delta
    profiles=read(root/'phase-records.json')
    for item in r['passes']:
        assert item['capture_ids']==ids and item['groups']==len(ids)
        records=[e for e in profiles if e['mode']==item['mode']];assert [e['capture_id'] for e in records]==ids
        if item['mode']=='reference':continue
        for entry in records:
            for key in ['process','prefix']:
                if key not in entry:continue
                trace=entry[key];assert 'error' not in trace
                assert trace['finished']>=trace['started']
                for phase in trace['phases']:assert trace['started']<=phase['started']<=phase['finished']<=trace['finished']
                if item['mode']=='observed_gc_deferred':assert not trace['automatic_gc_enabled'] and not trace['gc_collections']
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),replay_result_sha256=sha256(root/'result.json'),optimizer_updates=0,vehicle_control_sent=False,seed=None))
    write_json(run/'result.json',dict(status='completed_independent_moving_visual_tail_replay_audit',groups=len(ids),feature_max_differences=differences,
        raw_pixels_and_original_clocks_verified=True,original_unpublished_capture_ids=[e['capture_id'] for e in original if e.get('status')!='scene_published'],
        actor_qualified=False,live_tail_cause_identified=False,live_deadlines_verified=False,vehicle_control_sent=False,
        termination_reason='Retained original pixels/clocks and observational feature parity verified; replay is not live tail-cause or vehicle-control evidence'))

if __name__=='__main__':main()
