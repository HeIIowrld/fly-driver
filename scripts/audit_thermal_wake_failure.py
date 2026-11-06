"""Audit failed wake preparation and the independent native/graphics prefix."""
import argparse,base64,hashlib,shutil
from pathlib import Path
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.ac_native_clock import decode_graphics_timing
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparation',required=True);p.add_argument('--progress',required=True);p.add_argument('--deployment',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    prep=Path(a.preparation);probe=Path(a.progress);deploy=Path(a.deployment);thermal=prep/'temperature'
    pr=read(prep/'result.json');tr=read(thermal/'result.json');nr=read(probe/'result.json');dr=read(deploy/'result.json')
    for directory in (prep,thermal,probe):assert sha256(directory/'executed-entrypoint.py')==read(directory/'manifest.json')['config']['program_sha256']
    assert pr['status']==tr['status']==nr['status']=='failed'
    assert tr['temperature_applied_verified'] is False and tr['lifecycle_requests_sent']==1
    assert not (thermal/'request.json').exists() and not (thermal/'api-receipt.json').exists()
    assert read(thermal/'live-before-native.json')==[]
    wr=read(thermal/'native-wake-request.json');wt=read(thermal/'native-wake-api-receipt.json')
    assert wt['id']==wr['id'] and wt['command']==wr['command']=='wake-thermal-native-initial'
    assert wt['status']=='native_wake_returned_readback_required'
    assert [(c['name'],c['call_returned'],c['returned_false']) for c in wt['calls']]==[('awakeCar',True,False)]
    assert wt['temperature_setter_called'] is False and wt['native_readiness_verified'] is False
    assert wt['same_epoch_wake_retry_prohibited'] is True
    assert wt['before']['generation']==wt['after']['generation']==dr['generation_after']
    assert wr['epoch']==wt['before']['epoch']==wt['after']['epoch']==wt['after']['tyres_native_wake_epoch']
    assert wt['after'].get('tyres_initialization_epoch')!=wr['epoch']
    assert dr['status']=='completed' and dr['installed'] is True
    assert sha256(deploy/'installed-before.lua')==read(deploy/'manifest.json')['config']['arguments']['expected_installed_sha256']
    assert nr['samples_sha256']==sha256(probe/'samples.json')
    rows=read(probe/'samples.json');live=[];maximum_speed=0.;invalid=[]
    for i,row in enumerate(rows):
        authority=row['native'];raw=base64.b64decode(authority['prefix_base64'],validate=True)
        assert hashlib.sha256(raw).hexdigest()==authority['prefix_sha256']
        sample=decode_teacher_physics(raw,authority['received_monotonic_seconds'])
        assert sample.physics.packet_id==authority['packet_id'] and sample.builtin_ai_controlled is False
        f=sample.physics.fields;assert f['gas']<.02 and f['brake']>.98
        maximum_speed=max(maximum_speed,abs(f['speedKmh']))
        if abs(f['speedKmh'])>=.1:invalid.append(i)
        for name in ('graphics_before','graphics_after'):
            value=row[name];raw=base64.b64decode(value['prefix_base64'],validate=True)
            assert hashlib.sha256(raw).hexdigest()==value['prefix_sha256']
            g=decode_graphics_timing(raw,value['received_monotonic_seconds'])
            assert g.packet_id==value['packet_id'] and g.status==value['status']
        state=row['session_after'];assert state['generation']==wr['generation'] and state['epoch']==wr['epoch']
        if not state['paused']:live.append(row)
    assert invalid and invalid[-1]==len(rows)-1
    assert nr['final_paused'] is True and read(probe/'after.json')['paused'] is True
    assert read(prep/'final-pause/result.json')['status']=='completed'
    assert read(probe/'resume/result.json')['target_state_observed'] is True
    report=dict(observations=len(rows),live_observations=len(live),live_graphics_packets=len({r['graphics_after']['packet_id'] for r in live}),
        live_physics_packets=len({r['native']['packet_id'] for r in live}),live_graphics_statuses=sorted({r['graphics_after']['status'] for r in live}),
        maximum_native_speed_kmh=maximum_speed,invalid_native_rows=invalid,final_paused=True,
        owned_temperature_calls=0,wake_calls_returned=1,wake_proved_native_readiness=False,
        resume_child_exit_recorded=any(s['command']=='resume' and 'returncode' in s for s in nr['stages']),
        native_atomicity_verified=False,source_of_stall_proven=False)
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),preparation_result_sha256=sha256(prep/'result.json'),
        native_progress_result_sha256=sha256(probe/'result.json'),deployment_result_sha256=sha256(deploy/'result.json'),optimizer_updates=0))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    write_json(run/'result.json',dict(status='completed_independent_thermal_failure_prefix_audit',**report,
        actor_qualified=False,termination_reason='Failure retained; one returned wake and no owned temperature calls, original native/graphics prefix and final API pause verified'))


if __name__=='__main__':main()
