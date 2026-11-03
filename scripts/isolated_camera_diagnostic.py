"""Diagnostic-only isolated camera owner, including bounded pixel packet."""
import multiprocessing as mp
import time
import sys
from flybrain_drive.ac_csp_vision import CSPVisionRecorder, NoFreshCSPFrame
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.isolated_pixel_packet import pack_candidate, unpack_candidate
from flybrain_drive.frame_persistence_profile_v3 import BufferedFrameWriterFactory, load_buffered_frame_profile, verify_buffered_persistence_archive


def _owner(pipe, hwnd, config, pin, profile_path, max_frames):
    sys.setswitchinterval(.001)
    camera = None
    try:
        camera = CSPVisionRecorder(hwnd, config, max_frames=max_frames, max_bytes=256*1024**2, period=.04,
            image_format='bmp', shared_transport=True, batched_shared_copy=True,
            async_frame_persistence=True, mmf_metadata=True, source_timing_v1=False,
            expected_source_sha256=pin, frame_writer_factory=BufferedFrameWriterFactory(load_buffered_frame_profile(profile_path)))
        pipe.send(('ok', camera.request_id))
        while True:
            operation, arg = pipe.recv()
            try:
                if operation == 'ready':
                    camera.wait_ready(); value = True
                elif operation == 'status':
                    value = dict(triggered=camera.triggered, alive=camera.thread.is_alive(), error=str(camera.error) if camera.error else None)
                elif operation == 'candidate':
                    value = camera.wait_for_candidate(max_age=.15, timeout=.04)
                elif operation == 'pixels':
                    value = pack_candidate(camera)
                elif operation == 'close':
                    value = camera.close()
                elif operation == 'archive':
                    receipt = camera.archive_to(arg, bundle=True)
                    with CameraArchive(arg) as archive:
                        count = len(verify_buffered_persistence_archive(archive, expected_mmf=True))
                    value = dict(receipt=receipt, count=count)
                elif operation == 'exit':
                    pipe.send(('ok', True)); break
                else:
                    raise ValueError('Unknown owner command')
                pipe.send(('ok', value))
            except NoFreshCSPFrame as exc:
                pipe.send(('no_frame', str(exc)))
            except Exception as exc:
                pipe.send(('error', type(exc).__name__+': '+str(exc)))
    except BaseException as exc:
        try: pipe.send(('fatal', type(exc).__name__+': '+str(exc)))
        except Exception: pass
    finally:
        if camera is not None and camera.thread.is_alive():
            try: camera.close()
            except Exception: pass
        pipe.close()


class IsolatedCamera:
    def __init__(self, hwnd, config, pin, profile_path, *, max_frames=200):
        if type(max_frames) is not int or not 1 <= max_frames <= 240:raise ValueError('Bounded diagnostic frame budget required')
        self.poisoned=False;ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe();self.process=ctx.Process(target=_owner,args=(child,hwnd,config,pin,profile_path,max_frames));self.process.start();child.close()
        self.request_id=self._receive(15);self.thread=self
    def _receive(self, timeout):
        if not self.pipe.poll(timeout):
            self.poisoned=True
            raise TimeoutError('Camera owner response unknown; do not retry request')
        kind,value=self.pipe.recv()
        if kind=='no_frame':raise NoFreshCSPFrame(value)
        if kind!='ok':raise RuntimeError('Camera owner: '+str(value))
        return value
    def call(self,op,arg=None,timeout=10):
        if self.poisoned:raise RuntimeError('Unresolved owner response; no further commands allowed')
        self.pipe.send((op,arg));return self._receive(timeout)
    def wait_ready(self):return self.call('ready')
    @property
    def triggered(self):return self.call('status')['triggered']
    @property
    def error(self):return self.call('status')['error']
    def is_alive(self):return self.process.is_alive() and self.call('status')['alive']
    def wait_for_candidate(self,*,max_age,timeout):
        if max_age!=.15 or timeout!=.04:raise ValueError('Fixed diagnostic bounds required')
        return self.call('candidate')
    def pixel_candidate(self):
        started=time.perf_counter()
        packet=self.call('pixels')
        received=time.perf_counter()
        candidate,group=unpack_candidate(packet)
        self.last_pixel_timing=dict(call_started=started,owner_selected=packet['owner_selected_at'],
            packet_received=received,unpack_finished=time.perf_counter())
        return candidate,group
    def close(self):return self.call('close',timeout=15)
    def archive_to(self,destination,*,bundle):
        if not bundle:raise ValueError('Bundled archive required')
        return self.call('archive',str(destination),timeout=60)
    def shutdown(self):
        if self.process.is_alive():
            self.call('exit');self.process.join(10)
        if self.process.is_alive():raise RuntimeError('Camera owner remains live; do not restart')
        self.pipe.close()
