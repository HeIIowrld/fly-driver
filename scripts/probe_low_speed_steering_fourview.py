"""Bounded low-speed simulator steering diagnostic; no actor and no window input."""
import argparse,json,os,struct,time
from pathlib import Path
from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder,NoFreshCSPFrame
from flybrain_drive.ac_dinput import inspect_vjoy_controllers
from flybrain_drive.ac_native_physics import NativePhysicsReader,PHYSICS_LAYOUT,PHYSICS_PREFIX
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import AxisBinding,VJoySession,load_vjoy_sdk
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def diagnostic(prefix):
    values=PHYSICS_PREFIX.unpack(prefix);offset=0;out={}
    for name,_,count in PHYSICS_LAYOUT:
        if name in ('gear','numberOfTyresOut'):out[name]=values[offset]
        offset+=count
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf8'))
    run=begin_run(a.output,dict(program_sha256=sha256(__file__),config_sha256=sha256(a.config),
        speed_limit_kmh=8.,steering_amplitude=.1,pulse_seconds=.6,maximum_acceleration_seconds=4.,
        sensor_scope='native and four-view camera host receipts, physical timing unverified',actor=False,ui_actions_sent=0,pulse_order='positive then negative, higher excitation',initial_gear='first'))
    result={'status':'failed','actor_deployed':False,'training_admissible':False,'ui_actions_sent':0,
            'actuator_messages_sent':0,'post_stop_verified':False}
    session=reader=ai=camera=None;log=None
    try:
        native=WindowsACWindow();windows=native.discover()
        if len(windows)!=1 or native.user.GetForegroundWindow()==windows[0].hwnd:raise ValueError('One background simulator required')
        window=windows[0];status_path=Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
        before,mt=read_published_json(status_path);generation=before['generation']
        devices=inspect_vjoy_controllers()
        if (len(devices['vjoy_devices'])!=1 or devices['candidate_ac_joy_index']!=0
            or devices['vjoy_devices'][0]['instance_guid']!=cfg['instance_guid']
            or sha256(cfg['profile_path'])!=cfg['profile_sha256']):raise ValueError('Exact device and profile required')
        sdk=load_vjoy_sdk(cfg['dll_path'],cfg['dll_sha256'])
        if sdk.GetVJDStatus(cfg['device_id'])!=1:raise ValueError('Free vJoy required')
        reader=NativePhysicsReader(timeout=.2,clock=time.perf_counter);ai=NativeTeacherStatusReader()
        log=(run/'events.jsonl').open('x',encoding='utf8')
        def sample():
            st,modified=read_published_json(status_path);flag=ai.snapshot();body=reader.receive();d=diagnostic(body.prefix_bytes)
            if (not 0<=time.time()-modified<=2 or st['generation']!=generation or st.get('is_online') is not False
                or st.get('in_menu') is not False or st.get('paused') is not False or st.get('ai_controlled') is not False
                or (camera is None and st.get('camera_active') is not False) or st.get('expert_probe_active') is not False
                or flag['builtin_ai_controlled'] is not False or native.probe(window.hwnd,require_foreground=False)!=window
                or native.user.GetForegroundWindow()==window.hwnd or sha256(cfg['profile_path'])!=cfg['profile_sha256']
                or abs(body.fields['speedKmh'])>8 or d['numberOfTyresOut']!=0):raise ValueError('Motion or authority envelope violated')
            candidate=None
            if camera is not None:
                try: candidate=camera.candidate_before(body.received_monotonic_seconds,max_age=.15)
                except NoFreshCSPFrame: pass
            row=dict(kind='native',host_time=time.perf_counter(),native=physics_record(body),native_ai=flag,diagnostic=d,camera=candidate)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            return body,d
        def send(steer,gas,brake,buttons=()):
            x=session.submit(steer,gas,brake,pressed_buttons=buttons);log.write(json.dumps(dict(kind='request',request=x))+'\n');log.flush()
        initial,d=sample()
        if abs(initial.fields['speedKmh'])>=.1 or initial.fields['brake']<=.98 or initial.fields['gas']>=.02 or d['gear']!=2:
            raise ValueError('Stopped full-brake first-gear initial state required')
        session=VJoySession(sdk,cfg['device_id'],[AxisBinding(**x) for x in cfg['bindings']]);opening=session.open()
        log.write(json.dumps(dict(kind='open',request=opening))+'\n');log.flush()
        gear_deadline=time.monotonic()+2
        while True:
            body,d=sample()
            if d['gear']==2 and body.fields['brake']>.98:break
            if time.monotonic()>gear_deadline:raise TimeoutError('First gear not confirmed')
            time.sleep(.04)
        camera=CSPVisionRecorder(window.hwnd,json.loads(Path('configs/ac-csp-yaw-retina.json').read_text(encoding='utf8')),
            max_frames=150,period=.08,max_bytes=256*1024**2,image_format='png',preserve_producer_time=True)
        camera.wait_ready()
        result['camera_request_id']=camera.request_id
        start=time.monotonic()
        while True:
            body,d=sample()
            if d['gear']!=2:raise ValueError('First gear changed')
            if body.fields['speedKmh']>=5.:break
            if time.monotonic()-start>4.:raise TimeoutError('Low speed not reached')
            send(0.,.15,0.);time.sleep(.04)
        phases=[(.1,.6),(0.,.6),(-.1,.6),(0.,.6)]
        for steer,duration in phases:
            until=time.monotonic()+duration
            while time.monotonic()<until:
                body,d=sample()
                if d['gear']!=2 or body.fields['speedKmh']<.5:raise ValueError('Motion/gear not maintained')
                send(steer,0.,0.);time.sleep(.04)
        result['status']='completed_pulses_pending_stop'
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
    finally:
        if session is not None and session.owned:
            try:
                send(0.,0.,1.)
                deadline=time.monotonic()+5.
                while time.monotonic()<deadline:
                    body,d=sample()
                    if abs(body.fields['speedKmh'])<.1 and body.fields['brake']>.98 and body.fields['gas']<.02:
                        result['post_stop_verified']=True;break
                    time.sleep(.04)
            except Exception as error:result['stop_error']=str(error)
            result['actuator_messages_sent']=session.submit_attempts
            try:write_json(run/'close-request.json',session.close())
            except Exception as error:result['close_error']=str(error)
        if camera is not None:
            try:
                result['camera_close']=camera.close()
                if not camera.thread.is_alive() and camera.triggered:
                    result['camera_archive']=camera.archive_to(run/'vision',bundle=True)
            except Exception as error:result['camera_error']=type(error).__name__+': '+str(error)
        if reader is not None:reader.close()
        if ai is not None:ai.close()
        if log is not None:log.close()
        if result['status']=='completed_pulses_pending_stop' and result['post_stop_verified'] and result.get('camera_close',{}).get('status')=='stopped' and 'camera_error' not in result:
            result['status']='completed_low_speed_pulse_diagnostic'
        elif result['status']=='completed_pulses_pending_stop':result['status']='failed_stop_or_camera_unverified'
        result['final_vjoy_status']=sdk.GetVJDStatus(cfg['device_id']) if 'sdk' in locals() else None
        write_json(run/'result.json',result);print(json.dumps(result),flush=True)
    if not result['status'].startswith('completed'):raise RuntimeError(result.get('reason',result['status']))

if __name__=='__main__':main()
