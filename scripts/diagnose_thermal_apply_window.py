"""One thermal setter, paired native/SDK rendering observations before/after/resume.

Diagnostic only: no actor admission, vehicle control or training. Existing deployed
source is pinned and never modified. API return is not temperature verification.
"""
import argparse,os,shutil,subprocess,sys,time
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader,repeated_teacher_snapshot
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.control_ac_session import read_session_heartbeat,validate_heartbeat

PIN='b2227739fb11d6ccce2020e976b2e0b3626d41ecd7dd8defb42ea8105a0495c2'
AXIS='runs/teacher-recovery-20261102/reproduction-settings-02/axis-config.json'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output',required=True);p.add_argument('--temperature-c',type=int,choices=[40,110],default=40);a=p.parse_args()
    if not a.execute:raise ValueError('Explicit finite one-shot diagnostic required')
    root=Path(os.environ['LOCALAPPDATA'])/'flybrain';api=WindowsACWindow();windows=api.discover();assert len(windows)==1;window=windows[0]
    assert sha256(Path(window.executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua')==PIN
    axis=read(AXIS);assert sha256(axis['profile_path'])==axis['profile_sha256'];sdk=load_vjoy_sdk(axis['dll_path'],axis['dll_sha256'])
    def state():
        s,m,_=read_session_heartbeat(root/'csp-session/status.json');validate_heartbeat(s,time.time()-m,mutating=False)
        if (s['camera_active'] or s['ai_controlled'] or s['is_online'] or s['expert_probe_active'] or abs(s['speed_kmh'])>=.1
                or s['gas']>=.02 or s['brake']<=.98 or sdk.GetVJDStatus(axis['device_id'])!=1 or api.user.GetForegroundWindow()==window.hwnd):
            raise ValueError('Pinned stationary offline AI/camera-off released-controller background required')
        return s
    initial=state()
    if not initial['paused'] or initial.get('tyres_initialization_epoch')==initial['epoch']:raise ValueError('Paused unused thermal epoch required')
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),source_pin=PIN,initial_session=initial,seed=None,actor_admitted=False,actuator_messages_sent=0,ui_actions_sent=0))
    shutil.copy2(__file__,run/'executed-entrypoint.py');result=dict(status='failed',actor_admitted=False,actuator_messages_sent=0,thermal_setter_calls=0);phases=[]
    def command(name,args):
        with (run/(name+'.log')).open('x',encoding='utf8') as log:
            code=subprocess.run([sys.executable,'-m','scripts.control_ac_session',*args,'--expected-lua-sha256',PIN,'--execute','--output',str(run/name)],stdout=log,stderr=subprocess.STDOUT,cwd=Path.cwd(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
        return code
    def capture(name):
        before=state();camera=None;reader=NativeTeacherPhysicsReader(timeout=.2,clock=time.perf_counter);record=dict(name=name,session=before,started=time.perf_counter())
        try:
            record['native_before']=repeated_teacher_snapshot(reader.mapping,clock=time.perf_counter).authority_record()
            camera=CSPVisionRecorder(window.hwnd,read('configs/ac-csp-retina.json'),max_frames=4,period=.1,max_bytes=16*1024**2,
                image_format='bmp',shared_transport=True,batched_shared_copy=True,async_frame_persistence=True,mmf_metadata=True,
                preserve_producer_time=True,expected_source_sha256=PIN,mmf_stop_signal=True,memory_only_receive_checks=True)
            camera.thread.join(12)
            if camera.thread.is_alive():raise TimeoutError('Owned diagnostic camera incomplete')
            record['camera_close']=camera.close();record['archive']=camera.archive_to(run/name,bundle=True)
            if record['camera_close']['error'] or camera.count!=4:raise ValueError('Four original diagnostic frames required')
            samples=[]
            with CameraArchive(run/name) as archive:
                import json
                for row in map(json.loads,archive.read('stream.jsonl').decode().splitlines()):
                    for view in row['views']:
                        witness=view['source_render_metadata']['source_body_before_render']
                        samples.append(dict(capture_id=row['capture_id'],view_id=view['view_id'],received=row['received'],
                            source_clock=witness['source_clock_finished'],physics_timestamp_ms=witness['physics_timestamp_ms'],
                            sdk_core_temperatures_c=[w['core_temperature_c'] for w in witness['wheels']]))
            record['sdk_samples']=samples
            record['native_after']=repeated_teacher_snapshot(reader.mapping,clock=time.perf_counter).authority_record()
        finally:
            if camera is not None and camera.thread.is_alive():record['camera_emergency_close']=camera.close()
            reader.close();record['finished']=time.perf_counter();phases.append(record);write_json(run/'phases.json',phases)
        deadline=time.perf_counter()+2
        while time.perf_counter()<deadline:
            try:state();return
            except ValueError:time.sleep(.02)
        raise ValueError('Original idle camera session not restored')
    try:
        capture('before-paused')
        # Once only; an expected paused-native timeout is not a setter retry.
        code=command('thermal-call',['--command','set-tyres-temperature-initial','--temperature-c',str(a.temperature_c),'--axis-config',AXIS])
        receipt=read(run/'thermal-call/api-receipt.json');assert receipt['status']=='thermal_calls_returned_readback_required'
        result.update(thermal_setter_calls=1,thermal_call_returncode=code)
        capture('after-call-paused')
        if command('resume',['--command','resume'])!=0:raise ValueError('Explicit diagnostic resume failed')
        capture('after-resume')
        result.update(status='completed_thermal_apply_window_diagnostic',temperature_applied_verified=False,
            termination_reason='Paired source SDK and native observations around one setter/resume; no actor, adaptation or temperature qualification')
    except Exception as error:result.update(error=type(error).__name__,reason=str(error))
    finally:
        result['final_pause_returncode']=command('final-pause',['--command','pause'])
        write_json(run/'phases.json',phases);result['phases_sha256']=sha256(run/'phases.json');write_json(run/'result.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
