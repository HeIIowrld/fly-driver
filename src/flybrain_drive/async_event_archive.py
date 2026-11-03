"""Bounded immutable event handoff; slow disk writes never block control decisions."""
import json
import queue
import threading
import time
from pathlib import Path


class AsyncEventArchive:
    def __init__(self, path, *, capacity=512, stream=None):
        if not 1 <= capacity <= 4096:
            raise ValueError('Finite event archive queue required')
        self.stream = stream or Path(path).open('x', encoding='utf8', buffering=1024*1024)
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        self.accepted = self.written = self.high_water = 0
        self.maximum_write_seconds = self.maximum_flush_seconds = 0.
        self.maximum_enqueue_seconds = 0.
        self.stop = object()
        self.thread = threading.Thread(target=self._write, name='teacher-event-archive', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Event archive worker failed') from self.error
        if self.closed:
            raise RuntimeError('Event archive is closed')

    def emit(self, event):
        self.check()
        started = time.perf_counter()
        # Freeze and validate on the caller. The writer never observes references
        # to mutable control/sensor dictionaries or invents receipt timestamps.
        payload = json.dumps(event, allow_nan=False)+'\n'
        try:
            self.queue.put_nowait(payload)
        except queue.Full as error:
            raise RuntimeError('Bounded event archive queue is full') from error
        self.accepted += 1
        self.high_water = max(self.high_water, self.queue.qsize())
        self.maximum_enqueue_seconds = max(self.maximum_enqueue_seconds,
            time.perf_counter()-started)

    def _write(self):
        last_flush = time.perf_counter()
        try:
            while True:
                payload = self.queue.get()
                if payload is self.stop:
                    break
                started = time.perf_counter()
                self.stream.write(payload)
                self.maximum_write_seconds = max(self.maximum_write_seconds,
                    time.perf_counter()-started)
                self.written += 1
                if time.perf_counter()-last_flush >= .25:
                    started = time.perf_counter()
                    self.stream.flush()
                    self.maximum_flush_seconds = max(self.maximum_flush_seconds,
                        time.perf_counter()-started)
                    last_flush = time.perf_counter()
            self.stream.flush()
        except BaseException as error:
            self.error = error
        finally:
            try:
                self.stream.close()
            except BaseException as error:
                if self.error is None:
                    self.error = error

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.error is None:
            try:
                self.queue.put(self.stop, timeout=2)
            except queue.Full as error:
                raise RuntimeError('Archive termination backpressure') from error
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError('Owned event archive worker did not terminate')
        if self.error is not None:
            raise RuntimeError('Event archive worker failed during close') from self.error
        if self.accepted != self.written:
            raise RuntimeError('Accepted/written event counts differ')

    def statistics(self):
        return dict(accepted=self.accepted, written=self.written, high_water=self.high_water,
            maximum_enqueue_ms=self.maximum_enqueue_seconds*1000,
            maximum_write_ms=self.maximum_write_seconds*1000,
            maximum_flush_ms=self.maximum_flush_seconds*1000,
            worker_terminated=not self.thread.is_alive(),
            error=str(self.error) if self.error else None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
