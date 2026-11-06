"""Expose an owned shadow camera to the existing external-teacher protocol.

No actuator or lifecycle API is present here. The teacher performs its original
paused warmup/owned resume and its own source/actuator checks.
"""
import os,json
import threading
import time
from pathlib import Path
from .evidence import write_json


class TeacherShadowCameraCoordinator:
    def __init__(self,camera,gate,directory,*,window_handle,source_sha256,recording,
                 native_ready_required=False,frames=240,period=.04):
        self.directory=Path(directory).resolve()
        allowed=(Path(os.environ['LOCALAPPDATA'])/'flybrain/camera-coordination').resolve()
        if allowed not in self.directory.parents or not self.directory.is_dir() or any(self.directory.iterdir()):
            raise ValueError('Fresh owned local camera coordination directory required')
        self.camera,self.gate=camera,gate;self.stop=threading.Event();self.error=None
        self.plan=dict(schema='flybrain.owned.source_body_camera.v1',request_id=camera.request_id,
            generation=camera.clock.generation,window_handle=window_handle,recorder_pid=os.getpid(),
            source_sha256=source_sha256,trigger_file=str(self.directory/'trigger'),
            stop_file=str(self.directory/'stop'),health_file=str(self.directory/'health.json'),
            recording=str(recording),prepared_host_time=time.perf_counter(),frames=frames,period=period)
        if native_ready_required:self.plan.update(native_ready_required=True,native_ready_file=str(self.directory/'native-ready.json'))
        self.publish()
        write_json(self.directory/'ready.json',self.plan)
        self.thread=threading.Thread(target=self._serve,name='owned-shadow-camera-health',daemon=True);self.thread.start()

    def publish(self):
        camera=self.camera
        state='failed' if self.error or camera.error else 'completed' if camera.count>=self.plan['frames'] else 'capturing' if camera.triggered else 'prepared'
        write_json(self.directory/'health.json',dict(state=state,request_id=camera.request_id,
            frames=camera.count,error=self.error or (str(camera.error) if camera.error else None),observed_host_time=time.perf_counter()))

    def _serve(self):
        try:
            while not self.stop.is_set():
                native_ready=not self.plan.get('native_ready_required')
                if not native_ready and Path(self.plan['native_ready_file']).exists():
                    ready=json.loads(Path(self.plan['native_ready_file']).read_text(encoding='utf8'))
                    if ready['request_id']!=self.plan['request_id'] or ready['generation']!=self.plan['generation'] or ready['prior_actor_steps']!=0:
                        raise ValueError('Native start barrier belongs to another camera episode')
                    native_ready=True
                if (self.directory/'trigger').exists() and native_ready:self.gate.set()
                self.publish();time.sleep(.05)
        except BaseException as error:self.error=type(error).__name__+': '+str(error)

    def check(self):
        if self.error:raise RuntimeError(self.error)
        if not self.thread.is_alive():raise RuntimeError('Owned camera health publisher ended')

    def close(self,*,failed=False):
        if failed:self.error=self.error or 'Shadow/teacher lifetime ended without complete motion evidence'
        self.stop.set();self.thread.join(2)
        if self.thread.is_alive():raise RuntimeError('Owned camera health publisher did not stop')
        self.publish()


def require_teacher_motion_state(state,*,generation,epoch,request_id,maximum_speed_kmh=40):
    speed=state.get('speed_kmh')
    if (state.get('generation')!=generation or state.get('epoch')!=epoch
            or state.get('paused') is not False or state.get('live') is not True
            or state.get('ai_controlled') is not False or state.get('is_online') is not False
            or state.get('expert_probe_active') is not False
            or state.get('camera_request_id')!=request_id or state.get('camera_active') is not True
            or type(speed) not in (int,float) or not 0<=speed<=maximum_speed_kmh
            or state.get('error') or state.get('pending_receipt_id')):
        raise ValueError('Pinned offline AI-off owned-camera motion within its speed ceiling required')
