"""Reconstruct thermal readback from original native bytes, not stored claims."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX,PHYSICS_LAYOUT
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.initial_thermal_contract import validate_thermal_state,THERMAL_COMMAND


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def initialization_receipts(directory,result):
    p=Path(directory)
    if not result.get('readback_only'):
        return read(p/'request.json'),read(p/'api-receipt.json'),read(p/'publication.json'),p
    arguments=read(p/'manifest.json')['config']['arguments']
    original=Path(arguments['initialization_run'])
    if original.resolve()==p.resolve():raise ValueError('Readback cannot reference itself')
    reference=read(p/'original-receipt.json')
    request=read(original/'request.json');receipt=read(original/'api-receipt.json');publication=read(original/'publication.json')
    if (reference['request']!=request or reference['terminal']!=receipt or reference['publication']!=publication
            or reference['original_request_sha256']!=sha256(original/'request.json')
            or reference['original_terminal_sha256']!=sha256(original/'api-receipt.json')):
        raise ValueError('Read-only verification must preserve exact original SDK call receipts')
    return request,receipt,publication,original


def audit(directory):
    p=Path(directory);result=read(p/'result.json')
    request,receipt,publication,original=initialization_receipts(p,result)
    rows=read(p/'native-readback.json');target=request['temperature_c']
    assert result['status']=='completed_initial_thermal_native_readback'
    assert result['temperature_applied_verified'] is True and result['training_admissible'] is False
    assert request['command']==THERMAL_COMMAND and request['phase'] in ('episode_initialization','episode_initialization_live')
    assert request['destination_flags']==15 and request['remove_blankets'] is True
    assert receipt['id']==request['id'] and receipt['command']==THERMAL_COMMAND
    assert receipt['status']=='thermal_calls_returned_readback_required'
    assert receipt['temperature_applied_verified'] is False
    assert receipt['current_lap_excluded'] is True and result['current_lap_excluded'] is True
    assert request['generation']==receipt['before']['generation']==receipt['after']['generation']
    assert request['epoch']==receipt['before']['epoch']==receipt['after']['epoch']==receipt['after']['tyres_initialization_epoch']
    live=request['phase']=='episode_initialization_live'
    validate_thermal_state(receipt['before'],target,initialization=True,live_initialization=live)
    if live:
        before_rows=read(original/'live-before-native.json')
        assert request['native_stationary_verified'] is True and len(before_rows)==3
        assert request['native_stationary_packets']==[r['packet_id'] for r in before_rows]
        assert all(a['packet_id']<b['packet_id'] for a,b in zip(before_rows,before_rows[1:]))
        for entry in before_rows:
            raw=base64.b64decode(entry['prefix_base64'],validate=True)
            assert hashlib.sha256(raw).hexdigest()==entry['prefix_sha256']
            values=PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]);fields={};offset=0
            for name,_,n in PHYSICS_LAYOUT:fields[name]=values[offset:offset+n];offset+=n
            assert abs(fields['speedKmh'][0])<.1 and fields['gas'][0]<.02 and fields['brake'][0]>.98
            assert entry['builtin_ai_controlled'] is False and entry['received_monotonic_seconds']<=publication['host_clock']
    initial_resume_evidence=[]
    initial_args=read(original/'manifest.json')['config']['arguments']
    if initial_args.get('resume_before_live_initialization',False):
        from flybrain_drive.ac_teacher_physics import decode_teacher_physics
        assert live
        rr=read(original/'initial-resume-request.json');rt=read(original/'initial-resume-api-receipt.json');rp=read(original/'initial-resume-publication.json')
        assert rr['command']=='resume' and rt['id']==rr['id'] and rt['command']=='resume' and rt['status']=='api_returned_true'
        assert rr['generation']==request['generation']==rt['before']['generation']==rt['after']['generation']
        assert rr['epoch']==request['epoch']==rt['before']['epoch']==rt['after']['epoch']
        assert rt['before']['paused'] is True
        # API after-state can precede the next engine transition; require observed live state below.
        validate_thermal_state(rt['before'],target,initialization=True,live_initialization=False)
        initial_rows=read(original/'initial-resume-observations.json');observed={}
        assert initial_rows and initial_rows[-1]['heartbeat']['paused'] is False and initial_rows[-1]['heartbeat']['live'] is True
        for row in initial_rows:
            assert row['actor_admitted'] is False
            native=row['native'];raw=base64.b64decode(native['prefix_base64'],validate=True)
            assert hashlib.sha256(raw).hexdigest()==native['prefix_sha256']
            coupled=decode_teacher_physics(raw,native['received_monotonic_seconds']);f=coupled.physics.fields
            assert coupled.physics.packet_id==native['packet_id'] and coupled.builtin_ai_controlled is False
            assert abs(f['speedKmh'])<=5 and 0<=f['gas']<.02 and .98<f['brake']<=1
            if row['ready_native']:
                assert abs(f['speedKmh'])<.1 and native['packet_id']>rp['initial_packet']
                observed.setdefault(native['packet_id'],native)
        assert all(row==observed[row['packet_id']] for row in before_rows)
        assert rp['host_clock']<publication['host_clock']
        assert all(row['packet_id']>rp['initial_packet'] and row['received_monotonic_seconds']>=rp['host_clock'] for row in before_rows)
        initial_resume_evidence=['initial-resume-request.json','initial-resume-api-receipt.json','initial-resume-publication.json','initial-resume-observations.json','live-before-native.json']
    wake_evidence=[]
    wake_requested=read(original/'manifest.json')['config']['arguments'].get('wake_native_before_live_initialization',False)
    if wake_requested or (original/'native-wake-request.json').exists():
        assert live and wake_requested
        wr=read(original/'native-wake-request.json');wt=read(original/'native-wake-api-receipt.json');wp=read(original/'native-wake-publication.json')
        assert wr['command']=='wake-thermal-native-initial' and wr['phase']=='episode_initialization_live'
        assert wt['id']==wr['id'] and wt['command']==wr['command'] and wt['status']=='native_wake_returned_readback_required'
        assert wr['generation']==request['generation']==wt['before']['generation']==wt['after']['generation']
        assert wr['epoch']==request['epoch']==wt['before']['epoch']==wt['after']['epoch']==wt['after']['tyres_native_wake_epoch']
        assert wt['current_lap_excluded'] is True and wt['temperature_setter_called'] is False and wt['native_readiness_verified'] is False
        assert wt['same_epoch_wake_retry_prohibited'] is True
        assert wt['before'].get('tyres_initialization_epoch')!=wr['epoch'] and wt['after'].get('tyres_initialization_epoch')!=wr['epoch']
        assert [(c['name'],c['call_returned'],c['returned_false']) for c in wt['calls']]==[('awakeCar',True,False)]
        assert wt['ui_actions_sent']==0 and wt['actuator_messages_sent']==0
        assert all(row['packet_id']>wp['initial_packet'] and row['received_monotonic_seconds']>=wp['host_clock'] for row in before_rows)
        assert wp['host_clock']<publication['host_clock']
        wake_evidence=['native-wake-request.json','native-wake-api-receipt.json','native-wake-publication.json','live-before-native.json']
    assert [(c['name'],c['call_returned'],c['returned_false']) for c in receipt['calls']]==[
        ('setTyresBlankets',True,False),('setTyresTemperature',True,False)]
    if (p/'resume-request.json').exists():
        rr=read(p/'resume-request.json');rt=read(p/'resume-api-receipt.json')
        assert rr['command']=='resume' and rt['id']==rr['id'] and rt['status']=='api_returned_true'
        assert rr['generation']==request['generation'] and rr['epoch']==request['epoch']
    previous_packet=publication['initial_packet'];previous_receipt=publication['host_clock']
    matched=0;temperatures=[]
    for row in rows:
        authority=row['same_copy_authority'];raw=base64.b64decode(authority['prefix_base64'],validate=True)
        assert len(raw)==AI_STATUS_PREFIX.size
        assert hashlib.sha256(raw).hexdigest()==authority['prefix_sha256']
        assert hashlib.sha256(raw[:PHYSICS_PREFIX.size]).hexdigest()==authority['body_prefix_sha256']
        packet,ai=AI_STATUS_PREFIX.unpack(raw);assert ai==0
        assert packet==row['packet_id']==authority['packet_id'] and packet>previous_packet
        clock=row['host_receipt'];assert clock==authority['received_monotonic_seconds'] and clock>previous_receipt
        values=PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]);fields={};offset=0
        for name,_,count in PHYSICS_LAYOUT:
            fields[name]=list(values[offset:offset+count]);offset+=count
        actual=fields['tyreCoreTemperature'];assert actual==row['temperature_c']
        assert abs(fields['speedKmh'][0])<.1 and 0<=fields['gas'][0]<.02 and .98<fields['brake'][0]<=1
        within=all(abs(v-target)<=2 for v in actual)
        assert within==row['all_four_within_two_c']
        matched=matched+1 if within else 0
        previous_packet,previous_receipt=packet,clock;temperatures.append(actual)
    post_evidence=[]
    post_args=read(p/'manifest.json')['config']['arguments']
    if post_args.get('settle_initial_thermal_readback',False):
        from flybrain_drive.ac_teacher_physics import decode_teacher_physics
        post_rows=read(p/'post-thermal-observations.json');qualified={};consecutive=0;last_qualified=publication['initial_packet']
        for row in post_rows:
            assert row['actor_admitted'] is False
            native=row['native'];raw=base64.b64decode(native['prefix_base64'],validate=True)
            assert hashlib.sha256(raw).hexdigest()==native['prefix_sha256']
            coupled=decode_teacher_physics(raw,native['received_monotonic_seconds']);f=coupled.physics.fields
            assert coupled.builtin_ai_controlled is False and abs(f['speedKmh'])<=5 and 0<=f['gas']<.02 and .98<f['brake']<=1
            assert row['admitted_state']['generation']==request['generation'] and row['admitted_state']['epoch']==request['epoch']
            if row['stationary_native']:
                assert abs(f['speedKmh'])<.1 and row['admitted_state']['paused'] is False and row['admitted_state']['live'] is True
                qualified.setdefault(native['packet_id'],native)
                assert native['packet_id']>=last_qualified
                if native['packet_id']>last_qualified:
                    consecutive=consecutive+1 if all(abs(v-target)<=2 for v in f['tyreCoreTemperature']) else 0
                    last_qualified=native['packet_id']
            elif row.get('rejection')=='nonstationary_initialization_sample':
                assert abs(f['speedKmh'])>=.1;consecutive=0
        assert consecutive>=3
        assert all(row['same_copy_authority']==qualified[row['packet_id']] for row in rows)
        post_evidence=['post-thermal-observations.json']
    assert matched>=3
    evidence=['result.json','native-readback.json']
    if original==p:evidence+=['request.json','api-receipt.json','publication.json']
    else:evidence+=['original-receipt.json','manifest.json']
    evidence += [name for name in ['resume-request.json','resume-api-receipt.json'] if (p/name).exists()]
    hashes={name:sha256(p/name) for name in evidence}
    hashes.update({'wake/'+name:sha256(original/name) for name in wake_evidence})
    hashes.update({'initial_resume/'+name:sha256(original/name) for name in initial_resume_evidence})
    hashes.update({'post_settle/'+name:sha256(p/name) for name in post_evidence})
    if original!=p:
        hashes.update({'original/'+name:sha256(original/name) for name in ['result.json','request.json','api-receipt.json','publication.json']})
    return dict(directory=str(p),target_c=target,epoch=request['epoch'],generation=request['generation'],
        first_temperature_c=temperatures[0],last_temperature_c=temperatures[-1],distinct_packets=len(rows),
        consecutive_all_four_within_two_c=consecutive if post_evidence else matched,evidence_sha256=hashes,
        readback_only=result.get('readback_only',False),original_initialization_directory=str(original),
        original_initialization_status=read(original/'result.json')['status'],original_run_reclassified=False,
        deployed_lua_sha256=publication['expected_lua_sha256'],current_lap_excluded=True,
        post_setter_transition_verified=bool(post_evidence),owned_initial_resume_verified=bool(initial_resume_evidence),native_wake_verified=bool(wake_evidence),packet_atomicity_verified=False,condition_preparation_verified=True,dynamic_driving_verified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--initialization',nargs='+',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    reports=[audit(x) for x in a.initialization]
    assert len({r['generation'] for r in reports})==1
    assert len({r['epoch'] for r in reports})==len(reports)
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),scope=__doc__,ui_actions_sent=0,actuator_messages_sent=0))
    out=dict(status='completed_independent_initial_thermal_raw_audit',conditions=reports,
        training_admissible=False,student_optimizer_updates=0,actor_qualified=False,
        termination_reason='Original raw bytes/hashes, same-copy AI/control state, actual four-wheel temperatures and immutable SDK/resume receipts verified; stationary preparation only')
    write_json(run/'result.json',out);print(json.dumps(out))


if __name__=='__main__':main()
