"""Independent authority/motion evidence for teacher-driven student shadow."""
import base64
import hashlib
import json
from pathlib import Path
import numpy as np
from .ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX
from .graph import sha256
from .sac_unified_student import read
from .teacher_shadow_camera import require_teacher_motion_state


def audit_teacher_motion(root,result,config):
    root=Path(root)
    if result['actuator_messages_sent']!=0 or result['ui_actions_sent']!=0:
        raise ValueError('Student shadow must have no actuator/UI writes')
    assert config['teacher_speed_targets_kmh']==[10,20] and config['teacher_seconds_per_stage']==2
    assert config['hard_speed_ceiling_kmh']==40
    assert sha256(root/'executed-entrypoint.py')==config['program_sha256']
    from .audited_motion_spatial_candidate import load_readonly_motion_candidate
    candidate_spec=read(config['arguments']['candidate_plan'])
    _,motion_port,candidate_receipt=load_readonly_motion_candidate(candidate_spec)
    assert candidate_receipt==config['candidate']
    motion_transport=None
    if motion_port is not None:
        from .audited_motion_spatial_candidate import require_motion_stream_parity
        assert require_motion_stream_parity(candidate_spec,candidate_receipt,'runs/thermal-driving-20261103/spatial-recovery-01/pool-geometry.npz')==read(root/'motion-preflight.json')
        assert result['motion_features_enabled'] is True and sha256(root/'published-visual-inputs.npz')==result['published_visual_inputs_sha256']
        ready=read(root/'worker-ready.json');assert ready['motion_features_required'] is True and ready['runtime_kind']=='MotionSpatialDecisionRuntime'
        scenes={e['capture_id']:e for e in read(root/'events.json') if e.get('status')=='scene_published'}
        with np.load(root/'published-visual-inputs.npz') as z:
            ids=z['capture_ids'];assert len(set(ids))==len(ids)==result['published_motion_groups'] and set(ids)==set(scenes)
            assert z['motion'].shape==(len(ids),2048) and np.isfinite(z['motion']).all()
            hashes={int(i):hashlib.sha256(z['motion'][j].tobytes()).hexdigest() for j,i in enumerate(ids)}
        assert all(hashes[i]==e['motion_features_sha256'] for i,e in scenes.items())
        outputs=[e for e in read(root/'owner-ticker-events.json') if e['status']=='shadow_output']
        assert all(e['motion_features_sha256']==hashes[e['capture_id']] for e in outputs)
        motion_transport=dict(published_groups=len(ids),outputs_bound=len(outputs),typed_cpu_owner_required=True,original_feature_hashes_verified=True,full_neural_replay_in_this_audit=False)
    assert sha256(root/'motion-states.json')==result['motion_states_sha256']
    plan=read(root/'camera-ownership-plan.json');before=config['preflight_session']
    states=read(root/'motion-states.json')
    for row in states:
        require_teacher_motion_state(row['state'],generation=before['generation'],epoch=before['epoch'],request_id=plan['request_id'])
    teacher_root=root/'teacher';tr=read(teacher_root/'result.json');tm=read(teacher_root/'manifest.json')['config']
    assert sha256(teacher_root/'result.json')==result['teacher_result_sha256']
    args=tm['arguments']
    assert args['targets']=='10,20' and args['stage_seconds']==args['final_seconds']==2 and args['unlimited_final_stage'] is False
    assert args['source_native'] is True and args['require_quiescent_file_bridge'] is True
    events=[]
    if 'events_sha256' in tr:
        assert sha256(teacher_root/'events.jsonl')==tr['events_sha256']
        events=[json.loads(line) for line in (teacher_root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    requests=[e for e in events if e['kind']=='teacher_request']
    for e in requests:
        assert 0<=e['source_capture_age_before_actuation_ms']<=50
        control=e['request']['requested_control']
        assert abs(control['steering'])<=.6+1e-9 and 0<=control['throttle']<=1 and 0<=control['brake']<=1
        assert e['request']['driver_accepted'] is True
    assert len(requests)==tr['teacher_requests']
    speeds=[];tyre_out=[];acceleration=[];temperature=[]
    for row in read(root/'owner-native-copy.json'):
        authority=row['authority'];raw=base64.b64decode(authority['prefix_base64'],validate=True)
        values=iter(PHYSICS_PREFIX.unpack(raw[:PHYSICS_PREFIX.size]));fields={k:tuple(next(values) for _ in range(n)) for k,_,n in PHYSICS_LAYOUT}
        speeds.append(fields['speedKmh'][0]);tyre_out.append(fields['numberOfTyresOut'][0]);acceleration.append(fields['accG']);temperature.append(fields['tyreCoreTemperature'])
    if (root/'body-start-admission.json').exists():
        start=read(root/'body-start-admission.json');owner=read(root/'owner-terminal.json')
        if 'prepared_session' in start:
            prepared=start['prepared_session'];assert prepared['paused'] is True and prepared['camera_active'] is False
            assert prepared['generation']==before['generation'] and prepared['epoch']==before['epoch']
            assert start['native_start_phase']=='after_teacher_preflight_before_camera_trigger'
            live=read(root/'live-admission.json');require_teacher_motion_state(live['session'],generation=before['generation'],epoch=before['epoch'],request_id=plan['request_id'])
            assert all(e['decision_time']>=live['observed_host_time'] for e in read(root/'owner-ticker-events.json') if e['status']=='shadow_output')
        else:require_teacher_motion_state(start['live_session'],generation=before['generation'],epoch=before['epoch'],request_id=plan['request_id'])
        assert start['prior_actor_steps']==0 and owner['sampler_started'] is True
        assert start['started']<=owner['native_start_host_time']<=start['finished']
        assert all(row['authority']['received_monotonic_seconds']>=owner['native_start_host_time'] for row in read(root/'owner-native-copy.json'))
    elif result['status']=='completed_teacher_moving_native_flight_shadow':raise ValueError('Verified live sensory-episode start receipt required')
    assert speeds and all(0<=s<=40 for s in speeds)
    normal_close=tr.get('closing_request',{}).get('requested_control')==dict(steering=0.,throttle=0.,brake=1.)
    clean=tr.get('control_metadata_process',{}).get('status')=='completed' and tr.get('control_metadata_process',{}).get('forced_termination') is False and tr.get('tcp_receiver_terminated') is True
    from .teacher_lateral_stop_evidence import audit_lateral_stop
    lateral=audit_lateral_stop(teacher_root,events,tr,tm)
    if lateral is not None:
        assert config['arguments']['teacher_lateral_calibration']==args['low_speed_lateral_calibration']
    passed=(result['status']=='completed_teacher_moving_native_flight_shadow'
        and result.get('teacher_exitcode')==0 and tr['status']=='curriculum_completed'
        and tr['actuator_messages_sent']>0 and bool(requests) and normal_close and clean
        and len(tr['stages'])==2 and all(s['completed'] for s in tr['stages'])
        and max(speeds)>=2 and max(tyre_out)==0 and bool(states) and (lateral is None or lateral['completed']))
    return dict(teacher_motion_contract_passed=passed,privileged_lateral_and_stop=lateral,teacher_requests=len(requests),teacher_actuator_messages_sent=tr['actuator_messages_sent'],
        teacher_status=tr['status'],teacher_cleanly_stopped=clean and normal_close,
        student_actuator_messages_sent=0,vehicle_control_role='External privileged teacher only; student inference read-only',maximum_native_speed_kmh=max(speeds),native_tyre_out_max=max(tyre_out),
        native_acceleration_g_range=np.stack([np.min(acceleration,axis=0),np.max(acceleration,axis=0)]).tolist(),
        native_temperature_c_range=[float(np.min(temperature)),float(np.max(temperature))],
        teacher_source_age_max_ms=max(e['source_capture_age_before_actuation_ms'] for e in requests) if requests else None,
        motion_input_transport=motion_transport,student_autonomous_lap_verified=False,limit_driving_verified=False)
