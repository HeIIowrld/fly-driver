"""Finite GC-deferred CPU owner; exact actor and original freshness contracts."""
import multiprocessing as mp
import multiprocessing.spawn as spawn
import sys,time,threading,gc
from pathlib import Path
from .current_flight_owner import CurrentFlightOwnerProcess,_serve_current_owner
from .evidence import write_json
from .graph import sha256
from .ac_runtime_gc import defer_automatic_cyclic_gc


def _serve_gc_owner(pipe,checkpoint,checkpoint_sha256,directory,flush_denormal):
    events=[]
    def observe(phase,info):events.append(dict(phase=phase,host_time=time.perf_counter(),thread_cpu=time.thread_time(),thread_id=threading.get_ident(),generation=info['generation'],collected=info.get('collected'),uncollectable=info.get('uncollectable')))
    gc.callbacks.append(observe)
    try:
        with defer_automatic_cyclic_gc() as receipt:
            _serve_current_owner(pipe,checkpoint,checkpoint_sha256,directory,flush_denormal)
    finally:
        gc.callbacks.remove(observe)
        write_json(Path(directory)/'owner-gc-events.json',events)
        write_json(Path(directory)/'owner-gc-scope.json',receipt)


class CurrentFlightGCOwnerProcess(CurrentFlightOwnerProcess):
    def __init__(self,*,checkpoint,checkpoint_sha256,directory,python_executable,library_directory,flush_denormal=True):
        self.directory=Path(directory)
        if not self.directory.is_dir() or not Path(python_executable).is_file() or not Path(library_directory,'torch').is_dir() or sha256(checkpoint)!=checkpoint_sha256:raise ValueError('Owned directory and exact current checkpoint/CPU runtime required')
        self.lock=threading.Lock();self.poisoned=False;self.pending_unknown=False;self.stopped=False;self.sequence=0;self.publication_sequence=0;self.events=[];self.stages=[];self.episode=None;self.mode=None
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe();self.process=ctx.Process(target=_serve_gc_owner,args=(child,str(checkpoint),checkpoint_sha256,str(self.directory),flush_denormal));original_executable=spawn.get_executable();original_path=list(sys.path)
        try:mp.set_executable(str(python_executable));sys.path.insert(0,str(library_directory));self.process.start()
        finally:mp.set_executable(original_executable);sys.path[:]=original_path
        child.close()
        try:
            self.ready=self._receive(0,'ready',time.perf_counter()+30)
            if self.ready['status']!='ready' or self.ready['pid']!=self.process.pid or '+cpu' not in self.ready['receipt']['torch_version'] or Path(self.ready['python_executable'])!=Path(python_executable) or self.ready['receipt']['checkpoint_sha256']!=checkpoint_sha256:raise ValueError('Exact current CPU owner readiness differs')
        except Exception:self.poisoned=True;self.pending_unknown=True;self.stop();raise
