"""Experimental read-only process sampler; not yet admitted to vehicle control.

Receipt times are host copy times, not engine sample or camera exposure times.
"""
import math
import multiprocessing as mp
import queue
import time

from .ac_image_buffer import ExistingNamedMapping
from .ac_teacher_physics import decode_teacher_physics, timed_teacher_snapshot
from .ac_teacher_status import AI_STATUS_PREFIX


def _worker(stop, output, status, period, max_samples, recorded):
    count = 0
    try:
        mapping = None if recorded is not None else ExistingNamedMapping(
            'Local\\acpmf_physics', AI_STATUS_PREFIX.size)
        try:
            status.send(('ready',))
            while not stop.is_set() and count < max_samples:
                if recorded is None:
                    sample, timing = timed_teacher_snapshot(mapping)
                else:
                    raw = recorded[count % len(recorded)]
                    now = time.perf_counter()
                    sample = decode_teacher_physics(raw, now)
                    timing = {'first_copy_start':now, 'first_copy_end':now,
                        'second_copy_start':now, 'second_copy_end':now,
                        'attempts':1, 'engine_atomicity_verified':False,
                        'physical_sample_time_verified':False, 'synthetic_replay':True}
                row = {'authority': sample.authority_record(), 'copy': timing}
                try:
                    output.put_nowait((count, row))
                except queue.Full:
                    raise RuntimeError('native process queue full; samples not admissible')
                count += 1
                time.sleep(period)
            status.send(('stopped', count))
        finally:
            if mapping is not None:
                mapping.close()
    except BaseException as exc:
        try:
            status.send(('error', type(exc).__name__, str(exc), count))
        except (OSError, EOFError):
            pass
        # Failure is terminal and unread buffered samples are inadmissible.
        # Do not let the child feeder wait forever for a stalled consumer.
        output.cancel_join_thread()
    finally:
        status.close()


class NativeProcessSampler:
    """Bounded queue with explicit sequence, stale and process failure rejection.

    ``recorded`` is an explicit test-only sequence of 420-byte native prefixes.
    The production mapping option is implemented but is not live-validated.
    """
    def __init__(self, *, period=.002, max_samples=20000, capacity=128,
                 max_age=.05, recorded=None, drain_on_stop=False):
        if not (0 < period <= .02 and type(max_samples) is int and max_samples > 0
                and type(capacity) is int and capacity > 0 and 0 < max_age <= .05):
            raise ValueError('bounded sampler configuration required')
        if recorded is not None and (not isinstance(recorded, tuple) or not recorded
                or any(not isinstance(x, bytes) or len(x) != AI_STATUS_PREFIX.size for x in recorded)):
            raise ValueError('explicit finite recorded 420-byte prefixes required')
        if type(drain_on_stop) is not bool:raise ValueError('Explicit retirement-drain option required')
        self.drain_on_stop=drain_on_stop;self.retired_rows=[];self.rejected_rows=[]
        self.period, self.max_age = period, max_age
        self.ctx = mp.get_context('spawn')
        self.stop_event = self.ctx.Event()
        self.output = self.ctx.Queue(maxsize=capacity)
        self.status, child_status = self.ctx.Pipe(duplex=False)
        self.process = self.ctx.Process(target=_worker,
            args=(self.stop_event, self.output, child_status, period, max_samples, recorded))
        self._child_status = child_status
        self.expected = 0
        self.terminal = None
        self.failure = None
        self.started = False

    def _status(self):
        if self.terminal is not None and self.terminal[0] == 'error':
            raise RuntimeError(f'producer failed: {self.terminal[1]}: {self.terminal[2]}')
        while True:
            try:
                if not self.status.poll():
                    break
                message = self.status.recv()
            except (EOFError, BrokenPipeError):
                break
            if message[0] == 'error':
                self.terminal = message
                raise RuntimeError(f'producer failed: {message[1]}: {message[2]}')
            if message[0] == 'stopped':
                self.terminal = message
            elif message[0] != 'ready':
                raise RuntimeError('unknown producer status')

    def start(self, timeout=10):
        if self.started:
            raise RuntimeError('sampler cannot be started twice')
        self.process.start()
        self._child_status.close()
        self.started = True
        if not self.status.poll(timeout):
            raise TimeoutError('producer did not signal readiness')
        message = self.status.recv()
        if message[0] != 'ready':
            raise RuntimeError(f'producer failed before readiness: {message}')

    def poll(self, *, timeout=.1, clock=time.perf_counter):
        if not self.started:
            raise RuntimeError('sampler not started')
        if self.failure is not None:
            raise RuntimeError(self.failure)
        self._status()
        try:
            sequence, row = self.output.get(timeout=timeout)
        except queue.Empty:
            self._status()
            if self.terminal is not None:
                raise RuntimeError(f'producer terminated: {self.terminal}')
            if not self.process.is_alive():
                raise RuntimeError('producer exited without terminal status')
            return None
        self._status()
        if type(sequence) is not int or sequence != self.expected:
            self.failure = 'native sample sequence missing or duplicated'
            raise RuntimeError(self.failure)
        receipt = row['authority']['received_monotonic_seconds']
        now = clock()
        if not math.isfinite(receipt) or not math.isfinite(now) or not 0 <= now-receipt <= self.max_age:
            self.failure = 'native process sample stale or future at consumption'
            self.rejected_rows.append(dict(sequence=sequence,row=row,actor_admitted=False,reason=self.failure,rejected_monotonic_seconds=now))
            raise RuntimeError(self.failure)
        self.expected += 1
        return row

    def stop(self, timeout=5):
        if not self.started:
            return
        self.stop_event.set()
        if self.drain_on_stop:
            # Only after stop, quarantine queued records to let the producer's
            # feeder flush before join. These records never become actor input.
            deadline=time.perf_counter()+timeout
            while self.process.is_alive() and time.perf_counter()<deadline:
                try:
                    sequence,row=self.output.get(timeout=.01)
                    self.retired_rows.append(dict(sequence=sequence,row=row,actor_admitted=False))
                except queue.Empty:pass
                self.process.join(0)
            while True:
                try:
                    sequence,row=self.output.get_nowait()
                    self.retired_rows.append(dict(sequence=sequence,row=row,actor_admitted=False))
                except queue.Empty:break
        else:self.process.join(timeout)
        if self.process.is_alive():
            raise TimeoutError('native sampler process still live')
        try:
            self._status()
            if self.process.exitcode != 0 or self.terminal is None:
                raise RuntimeError('native sampler exited without clean terminal receipt')
        finally:
            self.output.close()
            self.status.close()
