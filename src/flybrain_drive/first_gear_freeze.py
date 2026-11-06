"""One background CSP pause immediately after stationary first-gear engagement."""
import os
import time
import uuid
from pathlib import Path
from .evidence import write_json
from .graph import sha256
from .ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX,repeated_snapshot


def require_first_gear_freeze_state(state,fields):
    if (any(state.get(k) is not v for k,v in dict(paused=False,live=True,in_menu=False,
            is_online=False,ai_controlled=False,camera_active=False,expert_probe_active=False).items())
            or state.get('pending_receipt_id') or state.get('error')
            or fields['gear']!=2 or fields['autoShifterOn']!=1
            or not 0<=fields['speedKmh']<.1 or not 0<=fields['gas']<.02 or not .98<=fields['brake']<=1):
        raise ValueError('Live idle first-gear/autoshifter stationary brake hold required before freeze')


def pause_first_gear(*,output,window,axis,sdk,reader,source_sha256):
    from .ac_capture import WindowsACWindow
    from .ac_lifecycle_commit import fresh_unpublished_request
    from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat,terminal_receipt
    out=Path(output);out.mkdir(exist_ok=False)
    root=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session'
    def native_fields():
        sample=repeated_snapshot(reader.mapping)
        v=iter(PHYSICS_PREFIX.unpack(sample.prefix_bytes));fields={}
        for key,_,n in PHYSICS_LAYOUT:
            values=tuple(next(v) for _ in range(n));fields[key]=values[0] if n==1 else values
        return fields
    before,modified,_=read_session_heartbeat(root/'status.json');validate_heartbeat(before,time.time()-modified,mutating=True)
    fields=native_fields();require_first_gear_freeze_state(before,fields)
    api=WindowsACWindow()
    if (not source_sha256 or sha256(Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua')!=source_sha256
            or sdk.GetVJDStatus(axis['device_id'])!=1 or sha256(axis['profile_path'])!=axis['profile_sha256']
            or api.user.GetForegroundWindow()==window.hwnd or api.probe(window.hwnd,require_foreground=False)!=window):
        raise ValueError('Released original controller and pinned background lifecycle source required')
    request=dict(schema='flybrain.csp.session.request.v1',id=uuid.uuid4().hex,command='pause',
        generation=before['generation'],epoch=before['epoch'],expires_clock=before['script_clock']+3)
    fresh,modified,_=read_session_heartbeat(root/'status.json');validate_heartbeat(fresh,time.time()-modified,mutating=True)
    require_first_gear_freeze_state(fresh,native_fields())
    request=fresh_unpublished_request(before,fresh,request,publication_started=False)
    journal=root/'intent-journal';journal.mkdir(exist_ok=True)
    write_json(journal/(request['id']+'.json'),dict(request=request,role='stationary_first_gear_freeze',controller_released=True))
    # All source hashing happened before engagement or above on local files.
    # Retain the request on NAS only after the one local API publication.
    write_json(root/'request.json',request)
    deadline=time.perf_counter()+4;receipt=None;after=None
    try:
        while time.perf_counter()<deadline:
            receipt=terminal_receipt(root,request);after,_,_=read_session_heartbeat(root/'status.json')
            if receipt is not None and after.get('paused') is True:break
            time.sleep(.01)
        if receipt is None or receipt.get('status')!='api_returned_true' or after.get('paused') is not True:
            raise RuntimeError('First-gear pause outcome unknown/rejected; never retry')
        final=native_fields()
        if (after['generation']!=before['generation'] or after['epoch']!=before['epoch']
                or final['gear']!=2 or final['autoShifterOn']!=1 or final['brake']<.98 or final['gas']>=.02 or final['speedKmh']>=.1):
            raise ValueError('Actual first gear/brake hold did not survive pause')
        write_json(out/'result.json',dict(status='completed_first_gear_freeze',generation=after['generation'],epoch=after['epoch'],
            native_gear=final['gear'],native_speed_kmh=final['speedKmh'],native_brake=final['brake'],
            lifecycle_requests_sent=1,actuator_messages_sent=0,ui_actions_sent=0))
    finally:
        write_json(out/'request.json',request);write_json(out/'receipt.json',receipt);write_json(out/'after.json',after)
