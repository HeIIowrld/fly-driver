"""Isolated read-only owner of native sensing, body scheduling and flight recurrence."""
import multiprocessing as mp
import multiprocessing.spawn as spawn
from pathlib import Path
import sys,time
from .ac_runtime_gc import defer_automatic_cyclic_gc
from .evidence import write_json
from .graph import sha256
from .native_flight_body_ticker import NativeFlightBodyTicker
from .native_flight_decision_runtime import NativeFlightDecisionRuntime,prepare_fixed_cpu_backend
from .native_process_rows import NativeProcessRowsSampler
from .online_body_shadow import OnlineBodyShadow
import numpy as np
from .sac_unified_student import read


def _serve_owner(pipe,components,identity,directory,defer_cyclic_gc=False,flush_denormal=False,motion_port=None):
    runtime=sampler=ticker=selector=None;summary=dict(status='failed',vehicle_control_sent=False)
    directory=Path(directory);gc_scope=None;gc_receipt=None
    def finish():
        nonlocal summary,gc_scope
        if ticker is not None and ticker.thread.ident is not None:ticker.stop()
        # Stop/drain the producer before JSON formatting, hashes or NAS writes.
        if sampler is not None and summary.get('sampler_started'):
            try:sampler.stop();summary['sampler_stopped']=True
            except Exception as error:summary['sampler_error']=str(error);summary['sampler_stopped']=False
        else:
            summary.update(sampler_started=False,sampler_stopped=True,native_producer_pid=None)
            for name,key in [('owner-native-copy.json','native_sha256'),('owner-retired-native-copy.json','retired_native_sha256'),('owner-rejected-native-copy.json','rejected_native_sha256')]:
                write_json(directory/name,[]);summary[key]=sha256(directory/name)
            summary.update(rejected_native_records=0,retired_native_records=0)
        if ticker is not None:
            write_json(directory/'owner-ticker-events.json',ticker.events)
            write_json(directory/'owner-wait-events.json',ticker.wait_events)
            summary.update(numerics=getattr(ticker,'numerics',None),ticker_error=ticker.error,outputs=sum(e['status']=='shadow_output' for e in ticker.events),
                ticker_events_sha256=sha256(directory/'owner-ticker-events.json'))
        else:
            write_json(directory/'owner-ticker-events.json',[]);write_json(directory/'owner-wait-events.json',[])
            summary.update(ticker_error=None,outputs=0,ticker_events_sha256=sha256(directory/'owner-ticker-events.json'))
        if sampler is not None and summary.get('sampler_started'):
            write_json(directory/'owner-retired-native-copy.json',sampler.source.retired_rows)
            write_json(directory/'owner-rejected-native-copy.json',sampler.source.rejected_rows)
            summary['rejected_native_records']=len(sampler.source.rejected_rows)
            summary['rejected_native_sha256']=sha256(directory/'owner-rejected-native-copy.json')
            summary['retired_native_records']=len(sampler.source.retired_rows)
            summary['retired_native_sha256']=sha256(directory/'owner-retired-native-copy.json')
            write_json(directory/'owner-native-copy.json',sampler.rows)
            summary['native_sha256']=sha256(directory/'owner-native-copy.json')
            summary['native_producer_pid']=sampler.source.process.pid
        if runtime is not None:runtime.close()
        summary['status']='stopped' if summary.get('sampler_stopped') and summary.get('ticker_error') is None and not summary.get('owner_error') else 'stopped_failed_observation'
        if gc_scope is not None:
            gc_scope.__exit__(None,None,None);gc_scope=None
            summary['gc_scope']=gc_receipt
        write_json(directory/'owner-terminal.json',summary)
        return summary
    try:
        import torch
        torch.set_num_threads(1)
        if motion_port is None:runtime=NativeFlightDecisionRuntime(*components)
        else:
            from .motion_spatial_runtime import MotionSpatialDecisionRuntime
            runtime=MotionSpatialDecisionRuntime(*components,motion_port=motion_port)
        warm=prepare_fixed_cpu_backend(runtime.model)
        summary.update(runtime_kind=type(runtime).__name__,motion_features_required=motion_port is not None)
        pipe.send(dict(status='ready',pid=mp.current_process().pid,torch_version=torch.__version__,torch_file=torch.__file__,python_executable=sys.executable,warmup=warm,runtime_kind=type(runtime).__name__,motion_features_required=motion_port is not None,
            source_contract='Current native21/19 share this owner sampler first receipt; historical view bodies retain camera-alignment sampler receipts'))
        generation=None;publication_sequence=0
        while True:
            message=pipe.recv();kind=message.get('kind')
            if kind=='stop':pipe.send(dict(status='stopped',summary=finish()));break
            if kind=='begin':
                if runtime.episode is not None:raise ValueError('No owner reset inside a run')
                generation=message['generation'];runtime.begin_episode(message['episode'])
                if defer_cyclic_gc:
                    gc_scope=defer_automatic_cyclic_gc();gc_receipt=gc_scope.__enter__()
                sampler=NativeProcessRowsSampler(maximum_samples=20000,drain_on_stop=True)
                ticker=NativeFlightBodyTicker(runtime,native_rows=sampler.rows,native_identity=identity,episode_id=runtime.episode,source_error=lambda:sampler.error,flush_denormal=flush_denormal)
                selector=OnlineBodyShadow(directory='.',request_id=runtime.episode,generation=generation,offset_lower=0.,native_rows=sampler.rows,native_identity=identity)
                summary.update(components_ready_host_time=time.perf_counter(),sampler_started=True,native_start_host_time=time.perf_counter())
                sampler.start();ticker.start();summary['episode_ready_host_time']=time.perf_counter()
                pipe.send(dict(status='begun',episode=runtime.episode));continue
            if kind=='context':
                if selector is None or message['generation']!=generation:raise ValueError('Original owner/generation context required')
                selector._drain_native();bounds=tuple(message['bounds']);gaps=[]
                try:
                    for lower,upper in zip(bounds,message['uppers'],strict=True):
                        j=int(np.searchsorted(selector.history.receipts,lower,side='left')-1)
                        if j<1:raise ValueError('No noninitial prior body receipt')
                        packet=selector.history.samples[j].legacy.packet_id
                        row=next(r for r in sampler.rows[:selector.position] if r['authority']['packet_id']==packet)
                        gap=upper-row['copy']['first_copy_start']
                        if not 0<=gap<=.025:raise ValueError('Original source-to-render bound exceeds25ms')
                        gaps.append(gap)
                    event={};context=selector._select_context(event,camera_receipt=message['camera_received'],bounds=bounds)
                    if context is None or any(v is None for v in context.visual_contexts):raise ValueError('Original prior body context unavailable')
                    pipe.send(dict(status='context',context=context,evidence=event,gaps=gaps))
                except ValueError as error:pipe.send(dict(status='context_unavailable',reason=str(error)))
                continue
            if kind=='status':
                pipe.send(dict(status='status',error=None if ticker is None else ticker.error,
                    outputs=0 if ticker is None else sum(e['status']=='shadow_output' for e in ticker.events)));continue
            if kind!='publish' or ticker is None or message['sequence']!=publication_sequence+1:
                raise ValueError('Owner publication sequence differs')
            try:
                if message['visual'].generation!=generation:raise ValueError('Visual producer generation differs')
                publication=ticker.publish(message['visual'],message['context'])
                publication_sequence+=1;pipe.send(dict(status='published',sequence=publication_sequence,publication=publication))
            except Exception as error:
                pipe.send(dict(status='rejected',reason=type(error).__name__+': '+str(error),error=ticker.error))
    except BaseException as error:
        summary.update(owner_error=type(error).__name__+': '+str(error))
        try:finish()
        except Exception as closing:summary['closing_error']=str(closing);write_json(directory/'owner-terminal.json',summary)
        try:pipe.send(dict(status='failed',reason=summary['owner_error']))
        except Exception:pass
    finally:
        if gc_scope is not None:
            gc_scope.__exit__(None,None,None);gc_scope=None
            summary['gc_scope']=gc_receipt
            write_json(directory/'owner-terminal.json',summary)
        pipe.close()


class NativeFlightBodyOwnerProcess:
    def __init__(self,*components,native_identity,directory,python_executable,library_directory,defer_cyclic_gc=False,flush_denormal=False,motion_port=None):
        if not Path(directory).is_dir() or not Path(python_executable).is_file() or not Path(library_directory,'torch').is_dir():
            raise ValueError('Owned evidence directory and installed CPU runtime required')
        self.directory=Path(directory);self.poisoned=False;self.stopped=False;self.sequence=0;self.events=[];self.episode=None
        ctx=mp.get_context('spawn');self.pipe,child=ctx.Pipe();self.process=ctx.Process(target=_serve_owner,args=(child,components,native_identity,str(self.directory),defer_cyclic_gc,flush_denormal,motion_port))
        original_executable=spawn.get_executable();original_path=list(sys.path)
        try:
            mp.set_executable(str(python_executable));sys.path.insert(0,str(library_directory));self.process.start()
        finally:mp.set_executable(original_executable);sys.path[:]=original_path
        child.close()
        try:
            self.ready=self._receive(30)
            if self.ready.get('status')!='ready' or self.ready['pid']!=self.process.pid or '+cpu' not in self.ready['torch_version'] or Path(self.ready['python_executable'])!=Path(python_executable):
                raise ValueError('Exact CPU owner readiness differs')
        except Exception:
            self.poisoned=True;self.stop();raise

    def _receive(self,timeout):
        if not self.pipe.poll(timeout):self.poisoned=True;raise TimeoutError('Owner response unknown; no retry')
        try:r=self.pipe.recv()
        except (EOFError,OSError):self.poisoned=True;raise RuntimeError('Owner connection ended')
        if r['status']=='failed':self.poisoned=True;raise RuntimeError(r['reason'])
        return r

    def begin_episode(self,episode,generation,*,startup_timeout=5):
        if self.episode is not None or self.poisoned:raise RuntimeError('Exactly one explicit owner episode required')
        if startup_timeout not in (5,30):raise ValueError('Explicit default5s or prepared-before-control30s startup required')
        self.pipe.send(dict(kind='begin',episode=episode,generation=generation));r=self._receive(startup_timeout)
        if r!={'status':'begun','episode':episode}:self.poisoned=True;raise RuntimeError('Owner episode identity differs')
        self.episode=episode

    def status(self):
        if self.poisoned or self.stopped:raise RuntimeError('Owner unavailable')
        self.pipe.send(dict(kind='status'));return self._receive(.05)

    @property
    def error(self):return self.status()['error']

    @property
    def output_count(self):return self.status()['outputs']

    def context_for_camera(self,*,bounds,uppers,camera_received,generation):
        if self.poisoned or self.stopped:raise RuntimeError('Owner unavailable')
        self.pipe.send(dict(kind='context',bounds=bounds,uppers=uppers,camera_received=camera_received,generation=generation))
        return self._receive(.05)

    def publish(self,visual,context):
        if self.poisoned or self.stopped:raise RuntimeError('Owner unavailable')
        self.pipe.send(dict(kind='publish',sequence=self.sequence+1,visual=visual,context=context));r=self._receive(.05)
        if r['status']=='rejected':raise RuntimeError(r['reason'])
        if r.get('status')!='published' or r['sequence']!=self.sequence+1:self.poisoned=True;raise RuntimeError('Owner publication response differs')
        self.sequence+=1;return r['publication']

    def stop(self):
        if self.stopped:return self.terminal
        forced=False;reply=None
        if not self.poisoned:
            try:self.pipe.send(dict(kind='stop'));reply=self._receive(30)
            except Exception:self.poisoned=True
        self.pipe.close();self.process.join(30)
        if self.process.is_alive():self.process.terminate();self.process.join(5);forced=True
        if self.process.is_alive():raise RuntimeError('Owned body process remains live')
        self.stopped=True
        path=self.directory/'owner-ticker-events.json'
        if path.exists():self.events=read(path)
        self.terminal=dict(pid=self.process.pid,exitcode=self.process.exitcode,forced_termination=forced,poisoned=self.poisoned,response=reply)
        return self.terminal
