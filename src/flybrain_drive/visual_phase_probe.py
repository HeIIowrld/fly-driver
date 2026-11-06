"""Scoped visual wall/CPU/GC/GPU-stream observations; no model or clock changes."""
import gc
import threading
import time
from contextlib import contextmanager


class VisualPhaseProbe:
    def __init__(self, stream, *, synchronize, event_factory=None):
        self.stream, self.synchronize, self.event_factory = stream, synchronize, event_factory
        self.active = False

    def run(self, operation, record, **kwargs):
        if self.active:
            raise RuntimeError('Nested visual instrumentation forbidden')
        self.active = True
        phases, collections, pending, patches = [], [], {}, []
        start = time.perf_counter()
        cpu = time.thread_time()
        process_cpu = time.process_time()
        record.update(started=start, phases=phases, gc_collections=collections,
            automatic_gc_enabled=gc.isenabled(), observer_thread=threading.get_ident(),
            gpu_measurement='Stream event elapsed includes dispatch idle; not isolated kernel execution')

        def gc_event(phase, info):
            key = (threading.get_ident(), info['generation'])
            if phase == 'start':
                pending[key] = time.perf_counter()
            elif key in pending:
                collections.append(dict(started=pending.pop(key), finished=time.perf_counter(),
                    thread=key[0], generation=key[1], collected=info.get('collected'),
                    uncollectable=info.get('uncollectable')))

        def wrap(obj, name, label, gpu=False):
            original = getattr(obj, name)
            own = name in obj.__dict__
            old_own = obj.__dict__.get(name)
            def call(*args, **kw):
                stamp = dict(phase=label, started=time.perf_counter())
                thread = time.thread_time()
                events = None
                if gpu and self.event_factory:
                    events = (self.event_factory(), self.event_factory())
                    events[0].record()
                phases.append(stamp)
                try:
                    return original(*args, **kw)
                finally:
                    if events:
                        events[1].record()
                        stamp['_events'] = events
                    stamp.update(finished=time.perf_counter(), thread_cpu_seconds=time.thread_time()-thread)
            setattr(obj, name, call)
            patches.append((obj, name, own, old_own))

        gc.callbacks.append(gc_event)
        try:
            wrap(self.stream.projector, 'sample', 'retinal_sampling')
            wrap(self.stream.network, 'simulate', 'flyvis_simulate', True)
            wrap(self.stream.pool, 'forward', 'scene_pool', True)
            try:
                value = getattr(self.stream, operation)(**kwargs)
            except Exception as error:
                record['error'] = type(error).__name__+': '+str(error)
                raise
            finally:
                sync = time.perf_counter()
                self.synchronize()
                record['final_synchronize_seconds'] = time.perf_counter()-sync
                for phase in phases:
                    events = phase.pop('_events', None)
                    if events:
                        phase['gpu_stream_elapsed_seconds'] = events[0].elapsed_time(events[1])/1000
            return value
        finally:
            for obj, name, own, old_own in reversed(patches):
                if own:
                    setattr(obj, name, old_own)
                else:
                    delattr(obj, name)
            gc.callbacks.remove(gc_event)
            record.update(finished=time.perf_counter(), thread_cpu_seconds=time.thread_time()-cpu,
                process_cpu_seconds=time.process_time()-process_cpu)
            record['wall_seconds'] = record['finished']-start
            self.active = False
