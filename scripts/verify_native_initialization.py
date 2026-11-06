"""Observe settled native physics after restart/resume, before temperature initialization."""
import argparse,os,time
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_initialization_readiness import NativeInitializationReadiness
from flybrain_drive.sac_unified_student import read
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--axis-config',required=True);p.add_argument('--expected-lua-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),seed=None,actuator_messages_sent=0,ui_actions_sent=0,thermal_setter_calls=0))
    readiness=NativeInitializationReadiness();result=dict(status='failed',actor_admitted=False,thermal_setter_calls=0)
    reader=None
    try:
        api=WindowsACWindow();windows=api.discover();assert len(windows)==1;window=windows[0]
        assert sha256(Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua')==a.expected_lua_sha256
        axis=read(a.axis_config);assert sha256(axis['profile_path'])==axis['profile_sha256'];sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
        path=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
        before,modified,_=read_session_heartbeat(path);validate_heartbeat(before,time.time()-modified,mutating=False)
        if before.get('tyres_initialization_epoch')==before['epoch']:raise ValueError('Readiness must precede this epoch thermal setter')
        reader=NativeTeacherPhysicsReader(timeout=.2,clock=time.perf_counter);deadline=time.perf_counter()+5
        while time.perf_counter()<deadline:
            state,modified,_=read_session_heartbeat(path);validate_heartbeat(state,time.time()-modified,mutating=False)
            if (state['generation']!=before['generation'] or state['epoch']!=before['epoch'] or state['paused']
                    or not state['live'] or state['camera_active'] or state['ai_controlled'] or state['is_online']
                    or state['expert_probe_active'] or state.get('pending_receipt_id') or state.get('error')
                    or sdk.GetVJDStatus(axis['device_id'])!=1 or api.user.GetForegroundWindow()==window.hwnd):
                raise ValueError('Live original offline AI/camera-off background initialization required')
            coupled=repeated_teacher_snapshot(reader.mapping,clock=time.perf_counter)
            if readiness.observe(coupled):
                result.update(status='completed_native_initialization_readiness',epoch=state['epoch'],generation=state['generation'],
                    consecutive_distinct_stationary_packets=readiness.consecutive,
                    observed_temperatures_c=list(coupled.physics.fields['tyreCoreTemperature']),
                    termination_reason='Three distinct native stationary AI-off brake-hold packets after live restart/resume, before any thermal setting')
                break
            time.sleep(.02)
        if result['status']=='failed':raise TimeoutError('Native initialization did not settle within5s')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    finally:
        if reader is not None:reader.close()
    write_json(run/'observations.json',readiness.records);result['observations_sha256']=sha256(run/'observations.json');write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
