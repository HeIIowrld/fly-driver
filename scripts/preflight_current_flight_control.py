"""Read-only current session/control contract admission. Every failure has a run receipt."""
import argparse,os,time,json
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read

AXIS='runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json'
CALIBRATION='runs/thermal-driving-20261103/steering-calibration-02/result.json'
PIN='0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e'


def inspect(*,expected_lua_sha256=PIN,calibration_result=CALIBRATION,axis_config=AXIS,magione_stage_audit=None):
    from flybrain_drive.ac_capture import WindowsACWindow
    from flybrain_drive.ac_dinput import inspect_vjoy_controllers
    from flybrain_drive.ac_image_buffer import ExistingNamedMapping
    from flybrain_drive.ac_native_physics import decode_static_identity_prefix,PHYSICS_LAYOUT,PHYSICS_PREFIX
    from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
    from flybrain_drive.ac_file import FileTelemetryClient
    from flybrain_drive.ac_teacher_pilot import validate_calibration
    from flybrain_drive.ac_vjoy import load_vjoy_sdk
    from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat
    local=Path(os.environ['LOCALAPPDATA'])/'flybrain'
    state,mtime,_=read_session_heartbeat(local/'csp-session/status.json')
    validate_heartbeat(state,time.time()-mtime,mutating=True)
    api=WindowsACWindow();windows=api.discover()
    if len(windows)!=1:raise ValueError('Exactly one engine required')
    window=windows[0]
    if api.user.GetForegroundWindow()==window.hwnd:raise ValueError('Background engine required; no focus input permitted')
    if int(state['window_handle'].removesuffix('ULL'))!=window.hwnd:raise ValueError('Window identity differs')
    installed=Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    if sha256(installed)!=expected_lua_sha256:raise ValueError('Installed Lua source mismatch')
    with ExistingNamedMapping('Local\\acpmf_static',200) as mapping:identity=decode_static_identity_prefix(mapping.read())
    axis=read(axis_config);calibration=read(calibration_result)
    if identity['car_model']!=axis['car_model'] or identity['track']!=axis['track']:raise ValueError('Car/track identity differs')
    if state['track_id']!=identity['track'] or state['car_id']!=identity['car_model']:
        raise ValueError('CSP/native track and car identity differ')
    if sha256(axis['profile_path'])!=axis['profile_sha256']:raise ValueError('Profile hash changed')
    devices=inspect_vjoy_controllers()
    if devices['candidate_ac_joy_index']!=0 or len(devices['vjoy_devices'])!=1 or devices['vjoy_devices'][0]['instance_guid']!=axis['instance_guid']:raise ValueError('DirectInput identity differs')
    producer=FileTelemetryClient(local/'telemetry',timeout=.05).session_id
    validate_calibration(calibration,axis,producer,steering_request_scale=.6)
    from flybrain_drive.current_flight_native_control_feedback import calibration_native_steering
    feedback=calibration_native_steering(Path(calibration_result).parent)
    if feedback['profile_sha256']!=axis['profile_sha256'] or feedback['producer_session_id']!=producer:raise ValueError('Native feedback calibration identity differs')
    sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
    if sdk.GetVJDStatus(axis['device_id'])!=1:raise ValueError('vJoy not FREE; no ownership takeover')
    with NativeTeacherPhysicsReader() as reader:coupled=repeated_teacher_snapshot(reader.mapping)
    if coupled.builtin_ai_controlled:raise ValueError('Native AI active')
    fields=dict(coupled.physics.fields);raw=PHYSICS_PREFIX.unpack(coupled.physics.prefix_bytes);offset=0;guard={}
    for name,_,width in PHYSICS_LAYOUT:
        if name in ('gear','numberOfTyresOut'):guard[name]=raw[offset]
        offset+=width
    if abs(fields['speedKmh'])>=.1 or fields['gas']>=.02 or fields['brake']<=.98:raise ValueError('Stationary brake-held start required')
    stage = None
    if magione_stage_audit is not None:
        from flybrain_drive.ac_native_position import NativeDiagnosticPositionReader
        from flybrain_drive.magione_stage import admit_magione_stage
        pose_reader = NativeDiagnosticPositionReader()
        try:
            stage = admit_magione_stage(magione_stage_audit,state,coupled.physics,pose_reader.snapshot(),
                installed_sha256=expected_lua_sha256,now=time.perf_counter())
        finally:
            pose_reader.close()
    return dict(session=state,identity=identity,axis=axis,calibration=calibration,native_feedback=feedback,
                magione_staged_start=stage,
                installed_lua_sha256=expected_lua_sha256,window_handle=window.hwnd,
                native=coupled.authority_record(),native_fields=fields,native_guard=guard,
                producer_session_id=producer,student_trial_start_ready=guard['gear']>=2 and guard['numberOfTyresOut']==0,actuator_messages_sent=0,ui_actions_sent=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--calibration-result',default=CALIBRATION)
    parser.add_argument('--axis-config',default=AXIS);parser.add_argument('--expected-lua-sha256',default=PIN)
    parser.add_argument('--magione-stage-audit');args=parser.parse_args();out=begin_run(args.output,dict(arguments=vars(args),program_sha256=sha256(__file__),seed=None,read_only=True))
    try:result=dict(status='completed_readonly_control_preflight',**inspect(calibration_result=args.calibration_result,axis_config=args.axis_config,
        expected_lua_sha256=args.expected_lua_sha256,magione_stage_audit=args.magione_stage_audit))
    except Exception as error:result=dict(status='failed',error=type(error).__name__,reason=str(error),actuator_messages_sent=0,ui_actions_sent=0)
    write_json(out/'result.json',result);print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
