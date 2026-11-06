"""One isolated CPU owner for read-only native flight recurrence; no actuator API."""
import multiprocessing as mp
import multiprocessing.spawn as spawn
import sys
from pathlib import Path
from types import SimpleNamespace
from .native_flight_decision_runtime import NativeFlightDecisionRuntime,prepare_fixed_cpu_backend


def _serve(pipe,components):
    runtime=None
    try:
        import torch
        torch.set_num_threads(1)
        runtime=NativeFlightDecisionRuntime(*components)
        warm=prepare_fixed_cpu_backend(runtime.model)
        pipe.send(dict(status='ready',pid=mp.current_process().pid,warmup=warm,torch_version=torch.__version__,torch_file=torch.__file__,python_executable=sys.executable));sequence=0
        while True:
            message=pipe.recv()
            if message['kind']=='stop':pipe.send(dict(status='stopped',sequence=sequence));break
            if message['kind']=='begin':
                if runtime.episode is not None:raise ValueError('No mid-run worker reset')
                runtime.begin_episode(message['episode']);pipe.send(dict(status='begun',episode=runtime.episode));continue
            if message['kind']!='step' or message['sequence']!=sequence+1:raise ValueError('Worker decision order differs')
            decision=runtime.decide(**message['arguments']);sequence+=1
            pipe.send(dict(status='ok',sequence=sequence,decision=decision))
    except BaseException as error:
        try:pipe.send(dict(status='failed',reason=type(error).__name__+': '+str(error),rejection=runtime.last_rejection if runtime else None))
        except Exception:pass
    finally:
        if runtime is not None:runtime.close()
        pipe.close()


class NativeFlightProcessRuntime:
    def __init__(self,*components,python_executable=None,library_directory=None):
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe();self.process=ctx.Process(target=_serve,args=(child,components))
        self.episode=None;self.poisoned=False;self.last_rejection=None;self.sequence=0
        self.model=SimpleNamespace(previous_body_receipt=None)
        if (python_executable is None)!=(library_directory is None):raise ValueError('Explicit interpreter/library pair required')
        old_executable=spawn.get_executable();old_path=list(sys.path)
        try:
            if python_executable is not None:
                if not Path(python_executable).is_file() or not Path(library_directory,'torch').is_dir():raise ValueError('Installed CPU interpreter/library required')
                mp.set_executable(str(python_executable))
                sys.path.insert(0,str(library_directory))
            self.process.start()
        finally:
            mp.set_executable(old_executable);sys.path[:]=old_path
        child.close();self.ready=self._receive(30)
        if python_executable is not None and ('+cpu' not in self.ready.get('torch_version','') or Path(self.ready['python_executable'])!=Path(python_executable)):
            self.poisoned=True;raise RuntimeError('Declared CPU worker runtime differs')
        if self.ready.get('status')!='ready' or self.ready.get('pid')!=self.process.pid:
            self.poisoned=True;raise RuntimeError('Native worker startup identity differs')

    def _receive(self,timeout):
        if not self.pipe.poll(timeout):
            self.poisoned=True;raise TimeoutError('Native worker outcome unknown; no retry')
        try:response=self.pipe.recv()
        except (EOFError,OSError):
            self.poisoned=True;raise RuntimeError('Native worker exited without an outcome')
        if response['status']=='failed':
            self.poisoned=True;self.last_rejection=response.get('rejection');raise RuntimeError(response['reason'])
        return response

    def begin_episode(self,episode):
        if self.poisoned or self.episode is not None:raise RuntimeError('Exactly one explicit worker episode required')
        self.pipe.send(dict(kind='begin',episode=episode));response=self._receive(.05)
        if response!={'status':'begun','episode':episode}:self.poisoned=True;raise RuntimeError('Worker episode response differs')
        self.episode=episode

    def decide(self,**arguments):
        if self.poisoned or not self.process.is_alive():raise RuntimeError('Native worker unavailable')
        self.pipe.send(dict(kind='step',sequence=self.sequence+1,arguments=arguments));response=self._receive(.05)
        if response.get('status')!='ok' or response['sequence']!=self.sequence+1:
            self.poisoned=True;raise RuntimeError('Native worker response sequence differs')
        self.sequence+=1;decision=response['decision'];self.model.previous_body_receipt=decision['body_first_receipt']
        return decision

    def close(self):
        response=None;forced=False
        if not self.poisoned and self.process.is_alive():
            self.pipe.send(dict(kind='stop'));response=self._receive(5)
            if response.get('status')!='stopped' or response['sequence']!=self.sequence:
                self.poisoned=True;raise RuntimeError('Native worker terminal identity differs')
        self.process.join(5)
        if self.process.is_alive():
            # This process owns only a read-only discarded research model, with
            # no simulator/actuator handles. Unknown decisions are never retried.
            self.process.terminate();self.process.join(5);forced=True
        if self.process.is_alive():raise RuntimeError('Owned native worker remains live')
        self.pipe.close()
        return dict(status='discarded_failed_worker' if self.poisoned else 'stopped',exitcode=self.process.exitcode,
            forced_termination=forced,pid=self.process.pid,sequence=self.sequence,response=response)
