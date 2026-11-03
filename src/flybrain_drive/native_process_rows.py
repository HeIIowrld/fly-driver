"""Experimental bounded process sampler adapter for existing shadow readers.

The pump preserves native copy receipts and never creates vehicle authority.
"""
import threading

from .native_process_sampler import NativeProcessSampler


class NativeProcessRowsSampler:
    def __init__(self, *, period=.002, maximum_samples=20000, capacity=128, recorded=None):
        self.source = NativeProcessSampler(period=period, max_samples=maximum_samples,
                                           capacity=capacity, recorded=recorded)
        self.rows = []
        self.error = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name='native-process-pump', daemon=True)

    def _run(self):
        try:
            while True:
                try:
                    row = self.source.poll(timeout=.05)
                except RuntimeError as exc:
                    if self._stopping.is_set() and str(exc).startswith('producer terminated:'):
                        break
                    raise
                if row is not None:
                    self.rows.append(row)
                    self._ready.set()
        except Exception as exc:
            self.error = type(exc).__name__+': '+str(exc)
            self._ready.set()

    def start(self):
        self.source.start()
        self.thread.start()
        if not self._ready.wait(5) or self.error or not self.rows:
            raise RuntimeError('process sampler failed readiness: '+str(self.error))

    def stop(self):
        self._stopping.set()
        self.source.stop_event.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise TimeoutError('native process pump remains live')
        self.source.stop()
        if self.error is not None:
            raise RuntimeError('native process pump failed: '+self.error)
