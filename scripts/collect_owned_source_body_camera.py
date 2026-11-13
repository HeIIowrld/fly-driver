"""Finite prepared four-view recorder in its own process, no vehicle controls.

The owner signals a unique trigger and stop file. Capture remains unqualified
until an independent pixel/body/time audit; no student optimizer is invoked.
"""
import argparse
import base64
import json
import os
import threading
import time
import shutil
import subprocess
import sys
import gc
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256 as hash_bytes
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_body_file_process import NativeBodyFileProcess


def compute_flight_shadow(model, frame, prediction, packet):
    started = time.perf_counter()
    if started-frame['received'] > .15:
        model.reset()
        output = dict(status='stale_visual_at_worker_start')
    else:
        output = model.process(frame['capture_id'],frame['generation'],prediction,packet)
    return dict(output,started_s=started,finished_s=time.perf_counter(),controls_transmitted=False,
                prefix_sha256=hash_bytes(packet.prefix_bytes).hexdigest(),
                prefix_base64=base64.b64encode(packet.prefix_bytes).decode('ascii'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--coordination-directory',required=True)
    p.add_argument('--expected-lua-sha256',required=True)
    p.add_argument('--expected-track',default='monza',
                   help='Require this track at admission and in every frame')
    p.add_argument('--imola-motion-diagnostic',action='store_true')
    p.add_argument('--live-frozen-pins')
    p.add_argument('--live-flyvis-cache')
    p.add_argument('--live-flight-shadow',action='store_true')
    p.add_argument('--flight-shadow-backend',choices=('scipy_csr_compact_cpu','numba_csr8_cpu'))
    p.add_argument('--flight-shadow-worker',action='store_true')
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
    if ((a.live_frozen_pins is None) != (a.live_flyvis_cache is None)
            or a.live_frozen_pins and not a.imola_motion_diagnostic):
        p.error('Live frozen readout requires pins, cache and the Imola motion diagnostic')
    if a.live_flight_shadow and not a.live_frozen_pins:
        p.error('Live flight shadow requires the live frozen readout')
    if a.flight_shadow_backend and not a.live_flight_shadow:
        p.error('Flight shadow backend requires --live-flight-shadow')
    if a.flight_shadow_worker and not a.live_flight_shadow:
        p.error('Flight shadow worker requires --live-flight-shadow')
    if a.imola_motion_diagnostic:
        from scripts.control_ac_session import (
            IMOLA_OWNED_COLLECTION_SOURCE_SHA256, IMOLA_OWNED_MOTION_SOURCE_SHA256,
        )

        short = a.expected_lua_sha256 == IMOLA_OWNED_MOTION_SOURCE_SHA256 and a.frames*a.period <= 20
        approach = (a.expected_lua_sha256 == IMOLA_OWNED_COLLECTION_SOURCE_SHA256
                    and a.period == .1 and 360 <= a.frames <= 400)
        if (a.expected_track!='imola' or not (short or approach)
                or not a.stop_on_complete or a.source_body_recording
                or a.source_timing_diagnostic):
            p.error('Finite non-training Imola motion diagnostic and exact producer required')
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
    native_reader=None
    flight_worker=pending_flight=None
    live_rows=[]
    result=dict(status='failed',actuator_messages_sent=0,ui_actions_sent=0,
        training_admissible=False,student_optimizer_updates=0)
    gc_was_enabled=gc.isenabled()
    if a.memory_only_receive_checks:gc.disable()
    result['finite_camera_cyclic_gc_deferred']=a.memory_only_receive_checks
    try:
        retina=json.loads(Path(a.retina_config).read_text(encoding='utf8'))
        process=None
        flight_shadow=None
        if a.live_frozen_pins:
            from scripts.qualify_exterior_camera_stream import make_frozen_processor
            from flybrain_drive.ac_image_buffer import ExistingNamedMapping
            from flybrain_drive.ac_native_physics import (
                NativePhysicsReader, decode_static_identity_prefix, repeated_snapshot,
            )
            from flybrain_drive.virtual_navigation_body import (
                NavigationBodyEncoder, decode_navigation_body_prefix,
            )
            from plan_frozen_corner_archive import BODY_CONTRACT
            from flybrain_drive.ac_csp_vision import NoFreshCSPFrame

            with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
                identity=decode_static_identity_prefix(mapping.read())
            if identity.pop('track') != a.expected_track:
                raise ValueError('Native track differs from the live diagnostic')
            encoder=NavigationBodyEncoder(BODY_CONTRACT,native_identity=identity)
            process=make_frozen_processor(Path(a.live_frozen_pins),Path(a.live_flyvis_cache),retina,warmup=True)
            if a.live_flight_shadow:
                from live_flight_shadow import LiveFlightShadow

                backend=a.flight_shadow_backend or 'scipy_csr_compact_cpu'
                flight_shadow=LiveFlightShadow(identity,sparse_backend=backend)
                result['live_flight_backend']=backend
                result['live_flight_hashes']=flight_shadow.hashes
                if a.flight_shadow_worker:
                    flight_worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='flight-shadow')
            native_reader=NativePhysicsReader(clock=time.perf_counter)
            result['live_input_hashes']={path:sha256(path) for path in
                                       (a.live_frozen_pins,str(BODY_CONTRACT),__file__)}
        body_path=coordination/'native-body.jsonl'
        body_archive=NativeBodyFileProcess(body_path,diagnostic_ai=a.imola_motion_diagnostic)
        windows=WindowsACWindow().discover()
        if len(windows)!=1:
            raise ValueError('Exactly one original engine required')
        camera=CSPVisionRecorder(windows[0].hwnd,
            retina,
            max_frames=a.frames,period=a.period,max_bytes=2*1024**3,
            image_format='bmp',shared_transport=True,batched_shared_copy=True,
            async_frame_persistence=True,mmf_metadata=True,preserve_producer_time=True,
            expected_source_sha256=a.expected_lua_sha256,expected_track=a.expected_track,
            start_gate=gate,prepared_event=prepared,
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
        next_health=0.
        while not (coordination/'stop').exists():
            if pending_flight is not None and pending_flight[0].done():
                future, event = pending_flight
                event['flight_shadow']=future.result()
                pending_flight=None
            if (coordination/'trigger').exists():
                gate.set()
            state=('failed' if camera.error else 'completed' if camera.count==a.frames else
                'capturing' if camera.triggered else 'prepared')
            now=time.perf_counter()
            if now >= next_health or state in ('failed','completed'):
                body_archive.check()
                write_json(coordination/'health.json',dict(state=state,request_id=camera.request_id,
                    frames=camera.count,error=str(camera.error) if camera.error else None,
                    observed_host_time=now))
                next_health=now+.1
            if camera.error:
                raise RuntimeError('Camera collection failed') from camera.error
            if process is not None and camera.triggered:
                try:
                    frame=camera.candidate_before(time.perf_counter(),max_age=.15)
                except NoFreshCSPFrame:
                    frame=None
                if frame is not None and (not live_rows or frame['capture_id']>live_rows[-1]['frame']):
                    snapshot=repeated_snapshot(native_reader.mapping,clock=time.perf_counter)
                    copied=snapshot.received_monotonic_seconds
                    body=encoder.encode(decode_navigation_body_prefix(snapshot.prefix_bytes,copied),
                                        episode_id=frame['generation'],decision_monotonic_seconds=copied)
                    valid=bool(body.valid[[0,1,16]].all())
                    if valid:
                        prediction=process(camera,frame,speed_mps=float(body.values[16]),tilt=body.values[:2])
                    else:
                        prediction=dict(frame=frame['capture_id'],available=False,line=None,
                                        path_status='body_unavailable')
                    finished=time.perf_counter()
                    event=dict(prediction,body_valid=valid,packet_id=body.packet_id,
                        body_prefix_sha256=hash_bytes(snapshot.prefix_bytes).hexdigest(),
                        body_prefix_base64=base64.b64encode(snapshot.prefix_bytes).decode('ascii'),
                        body_original_copy_s=body.original_received_monotonic_seconds,
                        body_copy_s=copied,received_s=frame['received'],finished_s=finished,
                        speed_mps=float(body.values[16]) if valid else None,
                        tilt_rad=body.values[:2].tolist() if valid else None)
                    if flight_shadow is not None:
                        if valid and finished-frame['received'] <= .15:
                            if pending_flight is not None:
                                event['flight_shadow']=dict(status='worker_busy',finished_s=time.perf_counter(),
                                                            controls_transmitted=False)
                            else:
                                flight_packet=repeated_snapshot(native_reader.mapping,clock=time.perf_counter)
                                args=(flight_shadow,frame,prediction,flight_packet)
                                if flight_worker is None:
                                    event['flight_shadow']=compute_flight_shadow(*args)
                                else:
                                    pending_flight=(flight_worker.submit(compute_flight_shadow,*args),event)
                        else:
                            if flight_worker is None:
                                flight_shadow.reset()
                            event['flight_shadow']=dict(status='invalid_body_or_stale_visual',
                                                        finished_s=time.perf_counter(),controls_transmitted=False)
                    live_rows.append(event)
            if a.stop_on_complete and camera.count==a.frames:
                break
            if time.perf_counter()>deadline:
                raise TimeoutError('Owned trigger/stop lifetime exceeded 540 seconds')
            time.sleep(.002 if process is not None and camera.triggered else .1)
        result['status']='completed_owned_recording' if camera.count==a.frames else 'stopped_owned_recording'
    except Exception as error:
        result.update(error=type(error).__name__,reason=str(error))
    finally:
        if flight_worker is not None:
            flight_worker.shutdown(wait=True)
        if pending_flight is not None:
            future, event = pending_flight
            try:
                event['flight_shadow']=future.result()
            except Exception as error:
                event['flight_shadow']=dict(status='worker_failed',reason=str(error),controls_transmitted=False)
                result.update(status='failed',flight_worker_error=str(error))
        if native_reader is not None:
            native_reader.close()
            from train_vision_teacher_line import jsonl

            trace=run/'live-frozen-readout.jsonl'
            jsonl(trace,live_rows)
            result['live_frozen_readout']=dict(rows=len(live_rows),rows_sha256=sha256(trace),
                valid_body=sum(row['body_valid'] for row in live_rows),
                physical_tick_time_verified=False,controls_transmitted=False)
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
