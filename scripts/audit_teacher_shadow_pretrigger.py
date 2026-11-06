"""Audit a retained zero-frame/zero-actuation teacher-shadow startup rejection."""
import argparse
import base64
import hashlib
from pathlib import Path
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.teacher_motion_evidence import audit_teacher_motion


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shadow',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.shadow);r=read(root/'result.json');m=read(root/'manifest.json')['config']
    assert r['status']=='failed' and r['shadow_outputs']==0 and r['teacher_actuator_messages_sent']==0
    assert r['cleanup_verified'] and r['teacher_exitcode']==0 and r['owner_terminal']['exitcode']==0 and not r['owner_terminal']['forced_termination']
    assert r['camera_close']['frames']==0 and r['camera_close']['worker_alive'] is False and r['camera_close']['producer_terminal'] is None
    assert r['native_sampler_stopped'] and r['ticker_stopped']
    assert sha256(root/'events.json')==r['events_sha256']
    assert sha256(root/'native-copy.json')==sha256(root/'owner-native-copy.json')==r['native_sha256']
    assert sha256(root/'owner-ticker-events.json')==r['ticker_events_sha256'] and read(root/'owner-ticker-events.json')==[]
    with CameraArchive(root/'vision') as archive:assert archive.bundle_sha256==r['camera_archive']['bundle_sha256']
    gears=set();packets=set()
    rows=read(root/'owner-native-copy.json')
    for row in rows:
        v=row['authority'];raw=base64.b64decode(v['prefix_base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==v['prefix_sha256']
        assert v['received_monotonic_seconds']==row['copy']['second_copy_end']
        values=iter(PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]));fields={k:tuple(next(values) for _ in range(n)) for k,_,n in PHYSICS_LAYOUT}
        assert fields['packetId'][0]==v['packet_id'];packets.add(v['packet_id']);gears.add(fields['gear'][0])
    motion=audit_teacher_motion(root,r,m);assert motion['teacher_motion_contract_passed'] is False
    run=begin_run(a.output,dict(shadow=str(root),shadow_result_sha256=sha256(root/'result.json'),program_sha256=sha256(__file__),seed=None,actor_qualified=False))
    write_json(run/'result.json',dict(status='completed_pretrigger_rejection_audit',whole_shadow_passed=False,
        raw_native_rows=len(rows),distinct_native_packets=len(packets),observed_native_gears=sorted(gears),teacher_motion=motion,
        original_reason=read(root/'teacher/result.json')['reason'],original_run_reclassified=False,
        actor_qualified=False,vehicle_control_sent=False,
        termination_reason='Startup rejection, original raw native state, zero capture/actuation and source-owner closure checked; not a completed moving experiment'))


if __name__=='__main__':main()
