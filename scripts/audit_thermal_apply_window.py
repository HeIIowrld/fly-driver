"""Audit ordered SDK/native thermal observations, preserving failed capture/parser runs."""
import argparse,base64,hashlib,json,os,shutil
from pathlib import Path
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,decode_physics_prefix
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def native(authority):
    raw=base64.b64decode(authority['prefix_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==authority['prefix_sha256']
    return decode_physics_prefix(raw[:PHYSICS_PREFIX.size],authority['received_monotonic_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--window',required=True);p.add_argument('--followup',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.window);r=read(root/'result.json');assert sha256(root/'phases.json')==r['phases_sha256']
    phases=read(root/'phases.json');assert len(phases)==3
    output=[]
    for record in phases[:2]:
        with CameraArchive(root/record['name']) as archive:
            assert archive.bundle_sha256==record['archive']['bundle_sha256']
            rows=[json.loads(v) for v in archive.read('stream.jsonl').decode().splitlines()]
            samples=[]
            for row in rows:
                for view in row['views']:
                    body=view['source_render_metadata']['source_body_before_render'];samples.append([w['core_temperature_c'] for w in body['wheels']])
        before=native(record['native_before']);after=native(record['native_after'])
        assert before.packet_id==after.packet_id
        output.append(dict(phase=record['name'],sdk_core_temperatures_c=samples,native_before_core_c=list(before.fields['tyreCoreTemperature']),native_after_core_c=list(after.fields['tyreCoreTemperature']),native_packet_unchanged=True))
    request=read(root/'thermal-call/request.json');receipt=read(root/'thermal-call/api-receipt.json');assert request['temperature_c']==receipt['target_temperature_c']==40
    assert receipt['status']=='thermal_calls_returned_readback_required' and receipt['temperature_applied_verified'] is False
    followup=Path(a.followup);fr=read(followup/'result.json');fm=read(followup/'manifest.json')['config'];q=read(followup/'request.json');assert q==fm['request'] and fr['producer_terminal']['status']=='completed'
    folder=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-camera'/q['id'];samples=[];files=[]
    for i in range(3):
        f=folder/f'frame-{i:04d}.json';frame=read(f);assert frame['request_id']==q['id'] and frame['generation']==fm['initial_session']['generation']
        for view in frame['views']:samples.append([w['core_temperature_c'] for w in view['source_body_before_render']['wheels']])
        files.append(f)
    assert all(v==[80]*4 for v in output[0]['sdk_core_temperatures_c'])
    assert all(v==[40]*4 for v in output[1]['sdk_core_temperatures_c'])
    assert all(v==[80]*4 for v in samples)
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),window_result_sha256=sha256(root/'result.json'),followup_result_sha256=sha256(followup/'result.json'),seed=None,actor_admitted=False))
    for f in files:shutil.copy2(f,run/f.name)
    write_json(run/'result.json',dict(status='completed_independent_ordered_sdk_native_thermal_audit',paused_phases=output,
        post_resume_later_paused_sdk_core_c=samples,post_resume_native_core_c=list(native(read(followup/'native-after.json')).fields['tyreCoreTemperature']),
        exact_resume_frame_observed=False,original_window_status=r['status'],original_followup_status=fr['status'],
        sdk_setter_visible_while_paused=True,native40_verified=False,actor_admitted=False,vehicle_control_sent=False,
        interpretation='SDK40 visible while paused, native remained80; later after resume SDK and native80. Exact first physics update and reset mechanism not established. Original MMF/parser failures retained.'))

if __name__=='__main__':main()
