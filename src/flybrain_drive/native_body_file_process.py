"""Read and archive native body entirely outside the GPU camera process.

Only health/terminal metadata crosses process boundaries. Raw copies preserve
their source brackets; archival completion is never used as a sensory receipt.
"""
import json
import gc
import os
import threading
import multiprocessing as mp
import time
from pathlib import Path

from .ac_image_buffer import ExistingNamedMapping
from .ac_teacher_physics import decode_teacher_physics, timed_teacher_snapshot
from .ac_teacher_status import AI_STATUS_PREFIX
from .async_event_archive import AsyncEventArchive
from .evidence import write_json
from .ac_published_state import read_published_json


def _record(path,stop,status,maximum_samples,recorded):
    mapping=archive=None
    health_stop=threading.Event();health_latest=[None];health_errors=[]
    def publish_health():
        try:
            while not health_stop.wait(.1):
                row=health_latest[0]
                if row is not None:write_json(Path(path).with_suffix('.health.json'),row)
        except BaseException as error:health_errors.append(str(error))
    health_thread=threading.Thread(target=publish_health,name='native-health-metadata',daemon=True)
    original_gc=gc.isenabled()
    count=0;packets=set();last_receipt=last_packet=None
    maximum_gap=maximum_enqueue_age=0.
    result=dict(status='failed',samples=0,synthetic_replay=recorded is not None)
    try:
        archive=AsyncEventArchive(path)
        if recorded is None:
            mapping=ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size)
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            kernel.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD]
            if not kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x8000):raise ctypes.WinError(ctypes.get_last_error())
            result['process_priority']='ABOVE_NORMAL_PRIORITY_CLASS'
        write_json(Path(path).with_suffix('.health.json'),dict(state='recording',
            samples=0,last_original_copy_receipt=None,observed_host_time=time.perf_counter()))
        gc.collect();gc.disable()
        health_thread.start()
        status.send(('ready',))
        while not stop.is_set() and count<maximum_samples:
            if health_errors:raise RuntimeError('Health publisher failed: '+str(health_errors))
            if recorded is None:
                sample,timing=timed_teacher_snapshot(mapping)
            else:
                now=time.perf_counter()
                sample=decode_teacher_physics(recorded[count%len(recorded)],now)
                timing=dict(first_copy_start=now,first_copy_end=now,second_copy_start=now,
                    second_copy_end=now,attempts=1,synthetic_replay=True,
                    engine_atomicity_verified=False,physical_sample_time_verified=False)
            authority=sample.authority_record()
            receipt=authority['received_monotonic_seconds'];packet=authority['packet_id']
            if authority['builtin_ai_controlled'] or last_packet is not None and packet<last_packet:
                raise ValueError('Independent file body authority/packet continuity failed')
            if last_receipt is not None:
                gap=receipt-last_receipt
                maximum_gap=max(maximum_gap,gap)
                if not 0<gap<=.05:
                    result['rejected_copy_brackets']=timing
                    raise RuntimeError(f'Native source copy interval {gap*1000:.3f}ms exceeded 50ms')
            age=time.perf_counter()-receipt
            if not 0<=age<=.05:
                raise RuntimeError('Original native source exceeded 50ms before archive admission')
            maximum_enqueue_age=max(maximum_enqueue_age,age)
            archive.emit(dict(sequence=count,authority=authority,copy=timing))
            count+=1;packets.add(packet);last_receipt,last_packet=receipt,packet
            health_latest[0]=dict(state='recording',samples=count,
                last_original_copy_receipt=receipt,observed_host_time=receipt)
            time.sleep(.005)
        if not stop.is_set():
            raise RuntimeError('Finite native archive sample budget exhausted before owned stop')
        result['status']='completed'
    except BaseException as error:
        result.update(error=type(error).__name__,reason=str(error))
    finally:
        health_stop.set()
        if health_thread.ident is not None:health_thread.join(2)
        if health_thread.is_alive():result.update(status='failed',health_worker_alive=True)
        if original_gc:gc.enable()
        if mapping is not None:
            mapping.close()
        if archive is not None:
            try:
                archive.close()
            except Exception as error:
                result.update(status='failed',archive_error=str(error))
        result.update(samples=count,distinct_packets=len(packets),
            maximum_copy_receipt_gap_ms=maximum_gap*1000,
            maximum_original_copy_to_archive_admission_ms=maximum_enqueue_age*1000,
            archive=archive.statistics() if archive is not None else None)
        write_json(Path(path).with_suffix('.terminal.json'),result)
        status.send(('terminal',result));status.close()


class NativeBodyFileProcess:
    def __init__(self,path,*,maximum_samples=120000,recorded=None):
        if not 1<=maximum_samples<=120000:
            raise ValueError('Finite <=120000 native copy budget required')
        if recorded is not None and (not isinstance(recorded,tuple) or not recorded
                or any(not isinstance(x,bytes) or len(x)!=AI_STATUS_PREFIX.size for x in recorded)):
            raise ValueError('Explicit synthetic 420-byte recorded prefixes required')
        self.path=Path(path);self.failure=None;self.terminal=None
        self.started_host_time=time.perf_counter()
        ctx=mp.get_context('spawn');self.stop=ctx.Event()
        self.status,child=ctx.Pipe(duplex=False)
        self.process=ctx.Process(target=_record,args=(str(path),self.stop,child,maximum_samples,recorded))
        self.process.start();child.close()
        if not self.status.poll(10) or self.status.recv()!=('ready',):
            self.stop.set();self.process.join(5)
            raise RuntimeError('Independent native file process did not become ready')

    def check(self):
        while self.status.poll():
            try:
                message=self.status.recv()
            except EOFError:
                break
            if message[0]=='terminal':
                self.terminal=message[1]
        if self.terminal is not None:
            raise RuntimeError('Native file process ended during collection: '+str(self.terminal))
        if not self.process.is_alive():
            raise RuntimeError('Native file process exited without terminal receipt')
        health_path=self.path.with_suffix('.health.json')
        if health_path.exists():
            health,modified=read_published_json(health_path)
            age=time.perf_counter()-health['observed_host_time']
            if not -2*time.get_clock_info('perf_counter').resolution<=age<=2 or health.get('state')!='recording':
                raise RuntimeError('Native file process health publication stale')
        elif time.perf_counter()-self.started_host_time>1:
            raise RuntimeError('Native file process health publication absent')

    def close(self):
        self.stop.set();self.process.join(5)
        forced=self.process.is_alive()
        if forced:
            self.process.terminate();self.process.join(5)
        terminal_path=self.path.with_suffix('.terminal.json')
        if terminal_path.exists():
            self.terminal=json.loads(terminal_path.read_text(encoding='utf8'))
        result=dict(self.terminal or dict(status='failed',reason='No terminal receipt'))
        if forced or self.process.exitcode!=0:
            result['status']='failed'
        result.update(process_pid=self.process.pid,process_exitcode=self.process.exitcode,
            process_alive=self.process.is_alive(),forced_termination=forced,drain_alive=False)
        self.status.close()
        return result
