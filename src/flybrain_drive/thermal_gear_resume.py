"""Warm all host readers before one receipt-bound thermal gear-preparation resume."""
import os,time,uuid
from pathlib import Path
from .ac_capture import WindowsACWindow
from .ac_lifecycle_commit import fresh_unpublished_request
from .ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
from .evidence import write_json
from .graph import sha256
from .live_thermal_resume_readiness import wait_live_stationary
from .thermal_teacher_start import admit_thermal_teacher_start


def resume_prepared_thermal_for_gear(*,output,window,axis,sdk,source_sha256,initialization):
    from scripts.audit_initial_thermal_evidence import audit
    from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat,terminal_receipt
    out=Path(output);out.mkdir(exist_ok=False);root=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
    request=None;terminal=None;records=[];reader=None;result=dict(status='failed',ui_actions_sent=0,actuator_messages_sent=0,temperature_calls=0,lifecycle_requests_sent=0)
    api=WindowsACWindow();proof=audit(initialization)
    def read_state():
        state,mtime,_=read_session_heartbeat(root/'status.json');validate_heartbeat(state,time.time()-mtime,mutating=False);return state
    def ownership(state):
        if (int(state['window_handle'].removesuffix('ULL'))!=window.hwnd or api.probe(window.hwnd,require_foreground=False)!=window
                or api.user.GetForegroundWindow()==window.hwnd or sdk.GetVJDStatus(axis['device_id'])!=1
                or sha256(axis['profile_path'])!=axis['profile_sha256']
                or sha256(Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua')!=source_sha256):
            raise ValueError('Released background engine/profile changed before gear resume')
    try:
        reader=NativeTeacherPhysicsReader();before=read_state();ownership(before);native=repeated_teacher_snapshot(reader.mapping)
        admission=admit_thermal_teacher_start(proof,before,native.physics.fields['tyreCoreTemperature'],source_sha256=source_sha256)
        write_json(out/'before.json',dict(session=before,native=native.authority_record(),thermal_admission=admission))
        fresh=read_state();ownership(fresh)
        request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,command='resume',generation=before['generation'],epoch=before['epoch'],expires_clock=fresh['script_clock']+3)
        request=fresh_unpublished_request(before,fresh,request,publication_started=False)
        journal=root/'intent-journal';journal.mkdir(exist_ok=True)
        write_json(journal/(request['id']+'.json'),dict(request=request,before=fresh,phase='thermal_gear_preparation_resume',native=native.authority_record()))
        write_json(out/'request.json',request);ownership(fresh)
        result['lifecycle_requests_sent']=1
        write_json(root/'request.json',request)
        state,sample,witnesses,terminal=wait_live_stationary(request,before,native.physics.packet_id,read_state=read_state,
            read_native=lambda:repeated_teacher_snapshot(reader.mapping),poll_terminal=lambda:terminal_receipt(root,request),
            check_ownership=ownership,observations=records,initialized_thermal=True)
        actual=list(sample.physics.fields['tyreCoreTemperature'])
        if not all(abs(v-proof['target_c'])<=5 for v in actual):raise ValueError('Thermal condition left original5C band before gear action')
        write_json(out/'stationary-native.json',witnesses)
        result.update(status='completed_prepared_thermal_gear_resume',generation=state['generation'],epoch=state['epoch'],actual_temperature_c=actual,
            initialization_directory=initialization,source_sha256=source_sha256,actor_input=False,thermal_setter_calls=0)
    except Exception as error:result.update(error=type(error).__name__,reason=str(error));raise
    finally:
        if request is not None and terminal is None:terminal=terminal_receipt(root,request)
        write_json(out/'receipt.json',terminal);write_json(out/'observations.json',records);write_json(out/'result.json',result)
        if reader is not None:reader.close()
