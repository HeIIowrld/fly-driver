"""Implementation behind control_ac_session.py thermal commands; no desktop input."""
import json
import os
import time
import uuid
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_lifecycle_commit import fresh_unpublished_request
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.initial_thermal_contract import (THERMAL_COMMAND,VERIFY_THERMAL_COMMAND,
    ThermalReadback,validate_thermal_state,validate_thermal_native)


def run_initial_thermal(args):
    from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat,terminal_receipt
    run=begin_run(args.output,dict(arguments=vars(args),program_sha256=sha256(__file__),seed=780511,
        ui_actions_sent=0,actuator_messages_sent=0,phase='episode_initialization_live' if getattr(args,'live_initialization',False) else 'episode_initialization',
        temperature_unit='Celsius per installed CSP SDK',current_lap_excluded=True))
    result=dict(status='failed',temperature_applied_verified=False,training_admissible=False,
        ui_actions_sent=0,actuator_messages_sent=0,lifecycle_requests_sent=0,
        current_lap_excluded=True,actual_lap_validity_observed=False)
    reader=None;observations=None;resume_request=None;resume_terminal=None;initial_resume_request=None;initial_resume_terminal=None;initial_resume_observations=[];pause_request=None;pause_terminal=None;final_native=None
    root=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
    initialization=args.command==THERMAL_COMMAND
    live_initialization=getattr(args,'live_initialization',False);live_native=[];live_pre_observations=[];wake_initial_packet=None
    wake_native=getattr(args,'wake_native_before_live_initialization',False)
    resume_inside=getattr(args,'resume_before_live_initialization',False)
    settle_readback=getattr(args,'settle_initial_thermal_readback',False);post_thermal_observations=[]
    try:
        if settle_readback and not (live_initialization or getattr(args,'resume_for_readback',False)):
            raise ValueError('Thermal transition observation requires explicit live preparation or owned readback resume')
        if resume_inside and (not live_initialization or wake_native or getattr(args,'resume_for_readback',False)):
            raise ValueError('Owned pre-setting resume requires exclusive live initialization, without wake or post-setting resume')
        if wake_native and not live_initialization:raise ValueError('Native wake is exclusive to explicit live initialization')
        if live_initialization and (not initialization or getattr(args,'resume_for_readback',False)):
            raise ValueError('Live initialization is a distinct one-shot live preparation, never resume/reapply')
        if not args.expected_lua_sha256:raise ValueError('Explicit reviewed deployed Lua hash required')
        if args.temperature_c is None:raise ValueError('Explicit temperature Celsius required')
        if getattr(args,'pause_after_readback',False) and not getattr(args,'resume_for_readback',False) and not live_initialization:
            raise ValueError('Immediate thermal pause requires explicit resume-for-readback')
        if initialization and args.initialization_run:raise ValueError('Initialization reference is readback-only')
        if not initialization and not args.initialization_run:raise ValueError('Original initialization run required for read-only verification')
        api=WindowsACWindow();windows=api.discover()
        if len(windows)!=1:raise ValueError('Exactly one original engine required')
        window=windows[0]
        deployed=Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
        axis=json.loads(Path(args.axis_config).read_text(encoding='utf-8'))
        sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
        reader=NativeTeacherPhysicsReader(timeout=.2,clock=time.perf_counter)
        def check_ownership(state):
            if (int(state['window_handle'].removesuffix('ULL'))!=window.hwnd
                    or api.probe(window.hwnd,require_foreground=False)!=window
                    or api.user.GetForegroundWindow()==window.hwnd
                    or sha256(deployed)!=args.expected_lua_sha256
                    or sdk.GetVJDStatus(axis['device_id'])!=1):
                raise ValueError('Pinned background engine and released controller required')
        def preflight(*,initializing=initialization,live_mode=None):
            state,mtime,_=read_session_heartbeat(root/'status.json')
            validate_heartbeat(state,time.time()-mtime,mutating=True)
            validate_thermal_state(state,args.temperature_c,initialization=initializing,live_initialization=live_initialization if live_mode is None else live_mode)
            check_ownership(state)
            native=repeated_teacher_snapshot(reader.mapping,clock=time.perf_counter)
            try:validate_thermal_native(native)
            except ValueError:
                write_json(run/'rejected-native-preflight.json',dict(session=state,
                    native=native.authority_record(),initializing=initializing,actor_admitted=False))
                raise
            return state,native
        before,sample=preflight(live_mode=False if resume_inside else None)
        if resume_inside and args.execute:
            from flybrain_drive.live_thermal_resume_readiness import wait_live_stationary
            if before.get('capabilities',{}).get('initial_tyre_temperature_live_v1') is not True:
                raise ValueError('Live initialization capability required before owned resume')
            fresh,sample=preflight(live_mode=False)
            initial_resume_request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,
                command='resume',generation=before['generation'],epoch=before['epoch'],expires_clock=fresh['script_clock']+3)
            initial_resume_request=fresh_unpublished_request(before,fresh,initial_resume_request,publication_started=False)
            journal=root/'intent-journal';journal.mkdir(exist_ok=True)
            write_json(journal/(initial_resume_request['id']+'.json'),dict(request=initial_resume_request,before=fresh,native=sample.authority_record(),phase='pre_thermal_native_resume'))
            write_json(run/'initial-resume-request.json',initial_resume_request)
            check_ownership(fresh)
            write_json(run/'initial-resume-publication.json',dict(host_clock=time.perf_counter(),initial_packet=sample.physics.packet_id,native_before=sample.authority_record()))
            result['initial_resume_publication_attempts']=1;result['lifecycle_requests_sent']+=1
            write_json(root/'request.json',initial_resume_request)
            def read_initial_state():
                current,mtime,_=read_session_heartbeat(root/'status.json')
                validate_heartbeat(current,time.time()-mtime,mutating=False)
                return current
            before,sample,live_native,initial_resume_terminal=wait_live_stationary(initial_resume_request,before,sample.physics.packet_id,
                read_state=read_initial_state,read_native=lambda:repeated_teacher_snapshot(reader.mapping,clock=time.perf_counter),
                poll_terminal=lambda:terminal_receipt(root,initial_resume_request),check_ownership=check_ownership,observations=initial_resume_observations)
            result['initial_resume_stationary_native_verified']=True
        if wake_native and args.execute:
            if before.get('capabilities',{}).get('native_thermal_wake_v1') is not True or before.get('tyres_native_wake_epoch')==before['epoch']:
                raise ValueError('Unused explicit native-wake capability required')
            wake_request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,
                command='wake-thermal-native-initial',generation=before['generation'],epoch=before['epoch'],
                expires_clock=before['script_clock']+3,phase='episode_initialization_live',temperature_c=args.temperature_c,
                destination_flags=15,remove_blankets=True,released_controller_verified=True,
                expected_car_id='bmw_z4_gt3',expected_track_id='monza')
            fresh,sample=preflight();wake_request=fresh_unpublished_request(before,fresh,wake_request,publication_started=False)
            journal=root/'intent-journal';journal.mkdir(exist_ok=True)
            write_json(journal/(wake_request['id']+'.json'),dict(request=wake_request,before=fresh,phase='native_readiness_wake'))
            write_json(run/'native-wake-request.json',wake_request)
            if sdk.GetVJDStatus(axis['device_id'])!=1:raise ValueError('Controller acquired before native wake publication')
            wake_initial_packet=sample.physics.packet_id
            write_json(run/'native-wake-publication.json',dict(host_clock=time.perf_counter(),initial_packet=wake_initial_packet,native_before=sample.authority_record()))
            result['native_wake_publication_attempts']=1;result['lifecycle_requests_sent']+=1
            write_json(root/'request.json',wake_request)
            deadline=time.perf_counter()+5;wake_receipt=None
            while time.perf_counter()<deadline:
                wake_receipt=terminal_receipt(root,wake_request)
                if wake_receipt is not None:break
                time.sleep(.01)
            if wake_receipt is None:raise TimeoutError('Native wake outcome unknown; never retry')
            write_json(run/'native-wake-api-receipt.json',wake_receipt)
            if wake_receipt['status']!='native_wake_returned_readback_required':raise ValueError('Native wake failed; no retry')
        if live_initialization and (args.execute or not resume_inside):
            deadline=time.perf_counter()+1
            while True:
                after_wake=wake_initial_packet is None or sample.physics.packet_id>wake_initial_packet
                live_pre_observations.append(dict(session=before,native=sample.authority_record(),after_wake=after_wake))
                if after_wake and (not live_native or sample.physics.packet_id>live_native[-1]['packet_id']):
                    live_native.append(sample.authority_record())
                elif live_native and sample.physics.packet_id<live_native[-1]['packet_id']:raise ValueError('Live native preparation packet reset')
                if len(live_native)>=3:break
                if time.perf_counter()>=deadline:raise TimeoutError('Three advancing stationary native packets required before live setter')
                time.sleep(.02);before,sample=preflight()
            write_json(run/'live-before-native.json',live_native)
        write_json(run/'before.json',dict(session=before,native=sample.authority_record()))
        if not args.execute:
            result.update(status='read_only',temperature_target_c=args.temperature_c)
        else:
            if initialization:
                request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,
                    command=THERMAL_COMMAND,generation=before['generation'],epoch=before['epoch'],
                    expires_clock=before['script_clock']+3,phase='episode_initialization_live' if live_initialization else 'episode_initialization',
                    temperature_c=args.temperature_c,destination_flags=15,remove_blankets=True,
                    released_controller_verified=True,expected_car_id='bmw_z4_gt3',expected_track_id='monza')
                if live_initialization:request.update(native_stationary_verified=True,native_stationary_packets=[r['packet_id'] for r in live_native])
                fresh,sample=preflight()
                request=fresh_unpublished_request(before,fresh,request,publication_started=False)
                final,sample=preflight()
                request=fresh_unpublished_request(before,final,request,publication_started=False)
                journal=root/'intent-journal';journal.mkdir(exist_ok=True)
                intent=journal/(request['id']+'.json')
                write_json(intent,dict(request=request,before=final,publication_attempts=0,
                    host_clock=time.perf_counter(),native=sample.authority_record()))
                # Exact published intent, local I/O only; never renew after this journal.
                if sdk.GetVJDStatus(axis['device_id'])!=1:
                    raise ValueError('Controller acquired before thermal publication')
                issued=time.perf_counter()
                result['lifecycle_publication_attempts']=1
                write_json(root/'request.json',request)
                result['lifecycle_requests_sent']+=1
                write_json(run/'request.json',request)
                write_json(run/'publication.json',dict(host_clock=issued,initial_packet=sample.physics.packet_id,
                    local_intent=str(intent),expected_lua_sha256=args.expected_lua_sha256))
                deadline=time.monotonic()+5;terminal=None
                while time.monotonic()<deadline:
                    terminal=terminal_receipt(root,request)
                    if terminal is not None:break
                    time.sleep(.02)
                if terminal is None:raise TimeoutError('Thermal call outcome unknown; never republish/retry')
                write_json(run/'api-receipt.json',terminal)
                if terminal.get('status')!='thermal_calls_returned_readback_required':
                    raise RuntimeError('Thermal SDK calls rejected/partial/unknown; never replay')
                original_packet=sample.physics.packet_id
            else:
                original=Path(args.initialization_run)
                request=json.loads((original/'request.json').read_text(encoding='utf-8'))
                terminal=json.loads((original/'api-receipt.json').read_text(encoding='utf-8'))
                publication=json.loads((original/'publication.json').read_text(encoding='utf-8'))
                if (request.get('command')!=THERMAL_COMMAND or request.get('temperature_c')!=args.temperature_c
                        or request.get('generation')!=before['generation'] or request.get('epoch')!=before['epoch']
                        or terminal.get('id')!=request['id']
                        or terminal.get('status')!='thermal_calls_returned_readback_required'
                        or before.get('tyres_initialization_epoch')!=before['epoch']
                        or publication.get('expected_lua_sha256')!=args.expected_lua_sha256):
                    raise ValueError('Exact successful call receipt and same initialized producer/epoch required')
                issued=publication['host_clock'];original_packet=publication['initial_packet']
                write_json(run/'original-receipt.json',dict(request=request,terminal=terminal,publication=publication,
                    original_request_sha256=sha256(original/'request.json'),original_terminal_sha256=sha256(original/'api-receipt.json')))
            if getattr(args,'resume_for_readback',False):
                paused,native=preflight(initializing=False)
                if paused['paused'] is not True:
                    raise ValueError('Explicit thermal readback resume requires paused state')
                resume_request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,
                    command='resume',generation=paused['generation'],epoch=paused['epoch'],
                    expires_clock=paused['script_clock']+3)
                resume_request=fresh_unpublished_request(paused,paused,resume_request,publication_started=False)
                journal=root/'intent-journal';journal.mkdir(exist_ok=True)
                write_json(journal/(resume_request['id']+'.json'),dict(request=resume_request,
                    before=paused,phase='thermal_native_readback_resume',host_clock=time.perf_counter()))
                if sdk.GetVJDStatus(axis['device_id'])!=1:
                    raise ValueError('Controller acquired before owned readback resume')
                result['resume_publication_attempts']=1
                write_json(root/'request.json',resume_request)
                result['lifecycle_requests_sent']+=1
                # No NAS artifact writes until immediate native readback ends.
                # Independent CLI begin_run/I/O after resume let cold tyres cool
                # beyond the target tolerance before the first observation.
            observations=ThermalReadback(args.temperature_c,after_host_clock=issued,initial_packet=original_packet)
            deadline=time.monotonic()+(5 if settle_readback else 2)
            last_transition_packet=original_packet
            while time.monotonic()<deadline:
                if settle_readback:
                    from flybrain_drive.thermal_transition_observation import validate_transition_context,validate_transition_native
                    heartbeat,mtime,_=read_session_heartbeat(root/'status.json')
                    validate_heartbeat(heartbeat,time.time()-mtime,mutating=False)
                    native=repeated_teacher_snapshot(reader.mapping,clock=time.perf_counter)
                    state=heartbeat
                    receipt_state=terminal.get('after')
                    if receipt_state is not None and heartbeat['script_clock']<receipt_state['script_clock']:state=receipt_state
                    record=dict(heartbeat=heartbeat,admitted_state=state,native=native.authority_record(),actor_admitted=False,stationary_native=False)
                    post_thermal_observations.append(record)
                    check_ownership(heartbeat)
                    if heartbeat['generation']!=before['generation'] or heartbeat['epoch']!=before['epoch']:raise ValueError('Thermal transition source changed')
                    allowed=(None,request['id'],resume_request['id'] if resume_request else None)
                    if heartbeat.get('pending_receipt_id') not in allowed:raise ValueError('Unowned pending receipt during thermal transition')
                    validate_transition_context(state,before,allowed);validate_transition_native(native)
                    if native.physics.packet_id<last_transition_packet:raise ValueError('Native packet reset during thermal transition')
                    last_transition_packet=native.physics.packet_id
                    if state.get('paused') is not False or state.get('live') is not True:
                        time.sleep(.01);continue
                    try:validate_thermal_native(native)
                    except ValueError:
                        record['rejection']='nonstationary_initialization_sample';observations.consecutive=0
                        time.sleep(.01);continue
                    record['stationary_native']=True
                else:state,native=preflight(initializing=False)
                if state['generation']!=before['generation'] or state['epoch']!=before['epoch']:
                    raise ValueError('Producer/session changed during native thermal readback')
                if resume_request is not None:
                    resume_terminal=terminal_receipt(root,resume_request)
                if observations.observe(native) and (resume_request is None or resume_terminal is not None):break
                time.sleep(.02)
            if resume_request is not None:
                if resume_terminal is None or resume_terminal.get('status')!='api_returned_true':
                    raise RuntimeError('Owned thermal resume outcome unverified; never reissue')
                metadata_deadline=time.monotonic()+1
                while True:
                    state,_=preflight(initializing=False)
                    if state['paused'] is False and state.get('live') is True:break
                    if time.monotonic()>=metadata_deadline:
                        raise ValueError('Owned readback resume did not reach observed live state')
                    time.sleep(.02)
            if observations.consecutive<3:
                raise TimeoutError('SDK call known but three distinct native four-wheel temperatures unverified; use read-only verification after explicit lifecycle resume, never reapply')
            if getattr(args,'pause_after_readback',False):
                live,native=preflight(initializing=False)
                if live['paused'] is not False or live.get('live') is not True:
                    raise ValueError('Immediate thermal pause requires observed live readback state')
                pause_request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,
                    command='pause',generation=live['generation'],epoch=live['epoch'],expires_clock=live['script_clock']+3)
                pause_request=fresh_unpublished_request(live,live,pause_request,publication_started=False)
                write_json(root/'intent-journal'/(pause_request['id']+'.json'),dict(request=pause_request,
                    before=live,phase='thermal_verified_freeze',host_clock=time.perf_counter()))
                if sdk.GetVJDStatus(axis['device_id'])!=1:raise ValueError('Controller acquired before thermal pause')
                result['pause_publication_attempts']=1
                write_json(root/'request.json',pause_request);result['lifecycle_requests_sent']+=1
                deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    pause_terminal=terminal_receipt(root,pause_request)
                    state,final_native=preflight(initializing=False)
                    if pause_terminal is not None and state['paused'] is True:break
                    time.sleep(.02)
                if (pause_terminal is None or pause_terminal.get('status')!='api_returned_true'
                        or state['paused'] is not True):
                    raise RuntimeError('Thermal freeze pause outcome unverified; no automatic retry')
                actual=list(final_native.physics.fields['tyreCoreTemperature'])
                if not all(abs(t-args.temperature_c)<=2 for t in actual):
                    raise ValueError('Native freeze temperature left target±2C; retain failure')
                result['paused_after_readback_verified']=True
                result['native_frozen_temperature_c']=actual
            result.update(status='completed_initial_thermal_native_readback',temperature_applied_verified=True,
                target_temperature_c=args.temperature_c,consecutive_distinct_packets=observations.consecutive,
                readback_only=not initialization,current_lap_excluded=True,
                preparation_only_not_measured_lap=True)
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
    finally:
        if settle_readback:write_json(run/'post-thermal-observations.json',post_thermal_observations)
        if initial_resume_request is not None:
            write_json(run/'initial-resume-observations.json',initial_resume_observations)
            if initial_resume_terminal is None:initial_resume_terminal=terminal_receipt(root,initial_resume_request)
            if initial_resume_terminal is not None:write_json(run/'initial-resume-api-receipt.json',initial_resume_terminal)
        if live_initialization:
            write_json(run/'live-before-native.json',live_native)
            write_json(run/'live-pre-observations.json',live_pre_observations)
        if observations is not None:write_json(run/'native-readback.json',observations.observations)
        if resume_request is not None:
            write_json(run/'resume-request.json',resume_request)
            if resume_terminal is not None:write_json(run/'resume-api-receipt.json',resume_terminal)
        if pause_request is not None:
            write_json(run/'pause-request.json',pause_request)
            if pause_terminal is not None:write_json(run/'pause-api-receipt.json',pause_terminal)
            if final_native is not None:write_json(run/'frozen-native.json',final_native.authority_record())
        if reader is not None:reader.close()
        result['termination_reason']='Initial thermal SDK/preflight and independent native readback completed' if result['temperature_applied_verified'] else 'Explicit read-only/preflight/API/readback outcome; no automatic side-effect retry'
        write_json(run/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)
