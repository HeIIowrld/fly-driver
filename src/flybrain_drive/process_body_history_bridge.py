"""Experimental process-sampler to existing read-only body-history bridge.

No camera, actuator, physical-time certification or production admission.
"""
from .native_process_sampler import NativeProcessSampler
from .online_body_shadow import OnlineBodyShadow
from .unified_host_guard import UnifiedHostGuard


class ProcessBodyHistoryBridge:
    """Latch the existing guard on process, ordering or original-copy failure."""

    def __init__(self, sampler, guard, *, native_identity, episode_id):
        if not isinstance(sampler, NativeProcessSampler) or not isinstance(guard, UnifiedHostGuard):
            raise ValueError('explicit process sampler and unified guard required')
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError('explicit episode identity required')
        self.sampler = sampler
        self.guard = guard
        self.rows = []
        self.shadow = OnlineBodyShadow(directory='.', request_id=episode_id, generation=1,
            offset_lower=0., native_rows=self.rows, native_identity=native_identity)

    def collect(self, *, timeout=.1):
        if self.guard.failed:
            raise RuntimeError('guard failed; explicit restart required')
        try:
            row = self.sampler.poll(timeout=timeout)
            if row is None:
                return None
            self.rows.append(row)
            self.shadow._drain_native()  # Preserves original bytes, packet and host copy time.
            return row
        except Exception:
            self.guard.failed = True
            raise
