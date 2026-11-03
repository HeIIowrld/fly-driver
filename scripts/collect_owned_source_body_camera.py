"""Finite prepared four-view recorder in its own process, no vehicle controls.

The owner signals a unique trigger and stop file. Capture remains unqualified
until an independent pixel/body/time audit; no student optimizer is invoked.
"""
import argparse
import json
import os
import threading
import time
import shutil
import subprocess
import sys
import gc
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_body_file_process import NativeBodyFileProcess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--coordination-directory',required=True)
    p.add_argument('--expected-lua-sha256',required=True)
    p.add_argument('--frames',type=int,default=1200)
    p.add_argument('--period',type=float,default=.1)
    p.add_argument('--retina-config',default='configs/ac-csp-exterior-front-v3-retina.json')
    p.add_argument('--stop-on-complete',action='store_true')
    p.add_argument('--source-body-recording',action='store_true')
    p.add_argument('--source-timing-diagnostic',action='store_true')
    p.add_argument('--lifecycle-only-camera-status',action='store_true')
    p.add_argument('--mmf-stop-signal',action='store_true')
    p.add_argument('--memory-only-receive-checks',action='store_true')
    a=p.parse_args()
    if a.source_timing_diagnostic and (a.frames*a.period+5>(150 if a.source_body_recording else 60)):
        raise ValueError('Separate finite under60s timing diagnostic required')
    coordination=Path(a.coordination_directory).resolve()
    allowed=(Path(os.environ['LOCALAPPDATA'])/'flybrain/camera-coordination').resolve()
    if allowed not in coordination.parents or not coordination.is_dir() or any(coordination.iterdir()):
        raise ValueError('Fresh owned local camera coordination subdirectory required')
    run=begin_run(a.output,dict(arguments=vars(a),seed=780203,
        camera_configuration_sha256=sha256(a.retina_config),actuator_messages_sent=0,
        training_admissible=False,student_optimizer_updates=0))
    gate,prepared=threading.Event(),threading.Event()
    camera=body_archive=None
    result=dict(status='failed',actuator_messages_sent=0,ui_actions_sent=0,
        training_admissible=False,student_optimizer_updates=0)
    gc_was_enabled=gc.isenabled()
    if a.memory_only_receive_checks:gc.disable()
    result['finite_camera_cyclic_gc_deferred']=a.memory_only_receive_checks
    try:
        body_path=coordination/'native-body.jsonl'
        body_archive=NativeBodyFileProcess(body_path)
        windows=WindowsACWindow().discover()
        if len(windows)!=1:
            raise ValueError('Exactly one original engine required')
        camera=CSPVisionRecorder(windows[0].hwnd,
            json.loads(Path(a.retina_config).read_text(encoding='utf8')),
            max_frames=a.frames,period=a.period,max_bytes=2*1024**3,
            image_format='bmp',shared_transport=True,batched_shared_copy=True,
            async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,
            expected_source_sha256=a.expected_lua_sha256,start_gate=gate,prepared_event=prepared,
            source_body_recording=a.source_body_recording,source_timing_v1=a.source_timing_diagnostic,
            lifecycle_only_camera_status=a.lifecycle_only_camera_status,mmf_stop_signal=a.mmf_stop_signal,
            memory_only_receive_checks=a.memory_only_receive_checks)
        deadline=time.perf_counter()+45
        while not prepared.wait(.02):
            if camera.error is not None:
                raise RuntimeError('Camera preparation failed') from camera.error
            if time.perf_counter()>deadline:
                raise TimeoutError('Camera preparation exceeded 45 seconds')
        plan=dict(schema='flybrain.owned.source_body_camera.v1',request_id=camera.request_id,
            generation=camera.clock.generation,window_handle=windows[0].hwnd,
            recorder_pid=os.getpid(),source_sha256=a.expected_lua_sha256,
            trigger_file=str(coordination/'trigger'),stop_file=str(coordination/'stop'),
            health_file=str(coordination/'health.json'),recording=str(run),
            prepared_host_time=time.perf_counter(),frames=a.frames,period=a.period)
        write_json(run/'ownership-plan.json',plan)
        write_json(coordination/'health.json',dict(state='prepared',request_id=camera.request_id,
            observed_host_time=time.perf_counter()))
        write_json(coordination/'ready.json',plan)
        deadline=time.perf_counter()+540
        while not (coordination/'stop').exists():
            body_archive.check()
            if (coordination/'trigger').exists():
                gate.set()
            state=('failed' if camera.error else 'completed' if camera.count==a.frames else
                'capturing' if camera.triggered else 'prepared')
            write_json(coordination/'health.json',dict(state=state,request_id=camera.request_id,
                frames=camera.count,error=str(camera.error) if camera.error else None,
                observed_host_time=time.perf_counter()))
            if camera.error:
                raise RuntimeError('Camera collection failed') from camera.error
            if a.stop_on_complete and camera.count==a.frames:
                break
            if time.perf_counter()>deadline:
                raise TimeoutError('Owned trigger/stop lifetime exceeded 540 seconds')
            time.sleep(.1)
        result['status']='completed_owned_recording' if camera.count==a.frames else 'stopped_owned_recording'
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
    finally:
        if camera is not None:
            try:
                if a.source_timing_diagnostic and camera.triggered:
                    # Release the owned renderer first. Its bounded timing ledger
                    # is only committed after stationary pause, outside driving.
                    camera.request_stop()
                    camera.thread.join(5)
                    if camera.thread.is_alive():raise RuntimeError('Original camera still active before diagnostic pause')
                    completed=subprocess.run([sys.executable,'-m','scripts.control_ac_session',
                        '--command','pause','--execute','--expected-lua-sha256',a.expected_lua_sha256,
                        '--output',str(run/'pause-for-timing-ledger')],cwd=Path.cwd(),check=True)
                    pause=json.loads((run/'pause-for-timing-ledger/result.json').read_text(encoding='utf8'))
                    if pause['status']!='completed':raise RuntimeError('Verified diagnostic stationary pause failed')
                    result['diagnostic_pause_result_sha256']=sha256(run/'pause-for-timing-ledger/result.json')
                result['camera_close']=camera.close()
                result['frames']=camera.count
                result['camera_local_source_retained']=str(camera.local)
                if camera.error:
                    result.update(status='failed',reason=str(camera.error))
            except Exception as error:
                result.update(status='failed',closing_error=str(error))
        if body_archive is not None:
            result['native_body_close']=body_archive.close()
            if result['native_body_close']['status']!='completed':
                result['status']='failed'
            shutil.copyfile(body_path,run/'native-body.jsonl')
            if sha256(body_path)!=sha256(run/'native-body.jsonl'):
                raise ValueError('Native body archive copy differs')
            result['native_body_sha256']=sha256(body_path)
            result['native_body_local_retained']=str(body_path)
        if camera is not None and not camera.thread.is_alive():
            result['camera_archive']=camera.archive_to(run/'vision',bundle=True)
        if gc_was_enabled:gc.enable()
        write_json(run/'result.json',result)
        write_json(coordination/'terminal.json',result)
        print(json.dumps(result))


if __name__=='__main__':
    main()
