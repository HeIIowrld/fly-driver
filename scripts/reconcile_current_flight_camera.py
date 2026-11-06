"""One stationary camera diagnostic after a proven stopped/failed-metadata conflict.

No student, teacher, vJoy acquisition or desktop input. The original discrepancy
is preserved. Driving admission requires a NEW normally completed camera lease.
"""
import argparse,os,time,json,subprocess,sys
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.paused_mmf_startup import PausedMMFStartupGate
from flybrain_drive.ac_csp_clock import CSPClockBounds,initialize_clock
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
from flybrain_drive.stationary_camera_admission import validate_known_stopped_metadata_failure_probe
from scripts.preflight_current_flight_control import inspect,PIN,CALIBRATION
from scripts.shadow_current_flight_teacher_motion import previous_camera
from scripts.record_native_process_unified_shadow_v2 import owned_resume
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous-evidence',required=True);p.add_argument('--calibration-result',default=CALIBRATION);p.add_argument('--output',required=True);a=p.parse_args()
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),seed=None,actuator_messages_sent=0,ui_actions_sent=0))
    local=Path(os.environ['LOCALAPPDATA'])/'flybrain';sp=local/'csp-session/status.json';cr=local/'csp-camera'
    camera=None;resumed=False;result=dict(status='failed',actuator_messages_sent=0,ui_actions_sent=0,training_admissible=False)
    def state():
        s,t,_=read_session_heartbeat(sp);validate_heartbeat(s,time.time()-t,mutating=True);return s
    def lifecycle(name,command):
        child=subprocess.run([sys.executable,'-m','scripts.control_ac_session','--command',command,'--expected-lua-sha256',PIN,'--execute','--output',str(out/name)],cwd=Path.cwd(),capture_output=True,text=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write_json(out/(name+'-process.json'),dict(returncode=child.returncode,stdout=child.stdout,stderr=child.stderr))
        if child.returncode!=0:raise RuntimeError('Stationary diagnostic lifecycle '+name+' failed')
    try:
        initial=inspect(calibration_result=a.calibration_result);s=initial['session'];previous=previous_camera(cr)
        try:PausedMMFStartupGate(s,*previous);result['status']='normal_camera_admission_already_valid'
        except ValueError:
            root=Path(a.previous_evidence);old=read(root/'result.json');m=read(root/'manifest.json')['config']
            if old['camera_close']['producer_terminal']!=previous[2] or old['camera_close']['worker_alive'] is not False:raise ValueError('Original closed camera terminal mismatch')
            terminal=previous[2];echo=terminal.get('camera_stop_policy_echo',{})
            if (terminal.get('status')!='stopped' or echo.get('stop_signal_observed') is not True
                    or echo.get('request_id')!=previous[0]['id'] or echo.get('generation')!=s['generation']):raise ValueError('Original owned stop signal not proven')
            with CameraArchive(root/'vision') as archive:actual=archive.bundle_sha256
            if actual!=old['camera_archive']['bundle_sha256'] or not old['camera_archive']['all_hashes_verified']:raise ValueError('Original archive changed')
            if old.get('owner_summary'):
                owner=old['owner_summary'];client=old['owner_terminal']
            else:owner=old['owner_terminal'];client=old['owner_process_terminal']
            if (client['exitcode']!=0 or client['forced_termination'] or owner.get('native_producer_exitcode')!=0
                    or m.get('source_sha256',m['arguments'].get('expected_lua_sha256'))!=PIN):raise ValueError('Prior owners/source not closed and pinned')
            if 'actuator_terminal' in old and (not old['actuator_terminal']['device_released'] or old['actuator_process_terminal']['exitcode']!=0):raise ValueError('Prior device release/owner exit not proven')
            write_json(out/'original-discrepancy.json',dict(session=s,request=previous[0],camera=previous[1],terminal=previous[2],previous_result_sha256=sha256(root/'result.json'),archive_sha256=actual))
            lifecycle('resume-for-stationary-diagnostic','resume');resumed=True
            bounds=CSPClockBounds(s['generation'],maximum_width=.01);initialize_clock(cr,bounds,initial['window_handle'],host_clock=time.perf_counter)
            fresh=state();observed=previous_camera(cr)
            if observed[0]!=previous[0] or observed[2]!=previous[2]:raise ValueError('Original request/terminal changed')
            validate_known_stopped_metadata_failure_probe(fresh,*observed,bounds.record(),now=time.perf_counter())
            write_json(out/'stationary-only-admission.json',dict(session=fresh,request=observed[0],camera=observed[1],terminal=observed[2],clock=bounds.record(),scope='One finite stationary diagnostic, no vehicle controller'))
            lifecycle('pause-for-owned-camera','pause');resumed=False
            paused=state();again=previous_camera(cr)
            if (paused['generation']!=s['generation'] or again[0]!=observed[0] or again[2]!=observed[2]
                    or again[1].get('state')!='failed' or 'Cannot commit camera metadata' not in again[1].get('reason','')):raise ValueError('Stationary diagnostic original evidence changed during pause')
            # Same narrowly scoped boundary used by record_native_process_unified_shadow_v2.
            gate=PausedMMFStartupGate(paused,observed[0],None,observed[2])
            config=read('configs/ac-csp-exterior-front-v3-retina.json')
            camera=CSPVisionRecorder(initial['window_handle'],config,max_frames=60,period=.04,max_bytes=64*1024**2,image_format='bmp',shared_transport=True,batched_shared_copy=True,async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,expected_source_sha256=PIN,mmf_stop_signal=True,memory_only_receive_checks=True)
            camera.wait_ready();owned_resume(out,gate,camera,sp,cr);resumed=True
            deadline=time.perf_counter()+5.;observations=[]
            with NativeTeacherPhysicsReader() as reader:
                while camera.thread.is_alive():
                    now=time.perf_counter();sample=repeated_teacher_snapshot(reader.mapping)
                    if sample.builtin_ai_controlled or abs(sample.physics.fields['speedKmh'])>=.1 or sample.physics.fields['gas']>=.02 or sample.physics.fields['brake']<=.98:raise ValueError('Stationary diagnostic moved or lost brake/authority')
                    observations.append(sample.authority_record())
                    if now>deadline:raise TimeoutError('Stationary camera did not complete its 60 frames')
                    time.sleep(.01)
            write_json(out/'native-stationary-observations.json',observations)
            result['camera_close']=camera.close()
            if result['camera_close']['producer_terminal']['status']!='completed':raise ValueError('New stationary camera did not normally complete')
            result['camera_archive']=camera.archive_to(out/'vision',bundle=True)
            lifecycle('final-pause','pause');resumed=False
            final=inspect(calibration_result=a.calibration_result);new_previous=previous_camera(cr);PausedMMFStartupGate(final['session'],*new_previous)
            if new_previous[0]['id']!=camera.request_id or new_previous[1]['state']!='completed' or new_previous[2]['status']!='completed':raise ValueError('New completed terminal/heartbeat did not agree')
            write_json(out/'new-normal-admission.json',dict(session=final['session'],request=new_previous[0],camera=new_previous[1],terminal=new_previous[2]))
            result.update(status='completed_stationary_camera_reconciliation',original_discrepancy_preserved=True,new_request_id=camera.request_id,vehicle_control_sent=False)
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    finally:
        if camera is not None:
            try:
                if camera.thread.is_alive():result['camera_close']=camera.close()
            except Exception as error:result['camera_close_error']=str(error)
        if resumed:
            try:lifecycle('failure-pause','pause')
            except Exception as error:result['pause_error']=str(error)
        write_json(out/'result.json',result)
    print(json.dumps(result))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
