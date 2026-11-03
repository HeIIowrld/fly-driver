"""Bounded original native receipts for per-view and decision-time body contexts.

This is a host-receipt ordering utility, not a sensor synchronization assertion.
No thread, simulator access, dataset qualification, or actuator is created here.
"""

import copy
import math
import threading
from bisect import bisect_left, bisect_right

from .ac_native_physics import NativePhysicsSample
from .temporal_body_context import per_view_temporal_body_context
from .virtual_navigation_body import (
    NativeNavigationBodySample, NavigationBodyEncoder, decode_navigation_body_prefix,
)


class PerViewBodyHistory:
    """Keep first receipts and encode distinct view and decision body observations."""

    def __init__(self, *, episode_id, native_identity, capacity=4096):
        if (not isinstance(episode_id, str) or not episode_id
                or type(capacity) is not int or not 2 <= capacity <= 100000):
            raise ValueError('Explicit episode and bounded receipt capacity required')
        self.episode_id, self.capacity = episode_id, capacity
        self.encoders = [NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                                               native_identity=native_identity) for _ in range(5)]
        self.samples, self.receipts = [], []
        self.first_packet, self.last_sample, self.last_ingest_time = None, None, None
        self.last_decision, self.last_view_bounds = None, None
        self.lock = threading.Lock()

    def ingest(self, sample):
        if not isinstance(sample, NativeNavigationBodySample):
            raise ValueError('Original decoded navigation-body sample required')
        s = sample.legacy
        if (not math.isfinite(s.received_monotonic_seconds)
                or s.received_monotonic_seconds < 0):
            raise ValueError('Finite original host receipt time required')
        with self.lock:
            if self.last_ingest_time is not None and s.received_monotonic_seconds <= self.last_ingest_time:
                raise ValueError('Native copies must have strictly increasing receipt clocks')
            if self.last_sample is not None:
                old = self.last_sample.legacy
                if s.packet_id < old.packet_id:
                    raise ValueError('Native packet reset requires a new episode')
                if s.packet_id == old.packet_id:
                    if s.prefix_sha256 != old.prefix_sha256:
                        raise ValueError('Repeated packet changed bytes')
                    self.last_ingest_time = s.received_monotonic_seconds
                    return False
            else:
                self.first_packet = s.packet_id
            self.last_sample = sample
            self.last_ingest_time = s.received_monotonic_seconds
            self.samples.append(sample)
            self.receipts.append(s.received_monotonic_seconds)
            if len(self.samples) > self.capacity:
                del self.samples[0]
                del self.receipts[0]
            return True

    def ingest_native_physics(self, sample):
        """Adapt the collector's original native bytes and receipt without re-dating."""
        if not isinstance(sample, NativePhysicsSample):
            raise ValueError('Original native physics receipt required')
        decoded = decode_navigation_body_prefix(sample.prefix_bytes,
                                                sample.received_monotonic_seconds)
        if decoded.legacy.packet_id != sample.packet_id:
            raise ValueError('Native packet identity differs from original bytes')
        return self.ingest(decoded)

    def build(self, *, render_start_lowers, camera_receipt_time, decision_time):
        """Return context or None if no provably noninitial current receipt is available."""
        if (not isinstance(render_start_lowers, tuple) or len(render_start_lowers) != 4
                or any(isinstance(t, bool) or not isinstance(t, (int, float))
                       or not math.isfinite(t) or t < 0 for t in
                       (*render_start_lowers, camera_receipt_time, decision_time))
                or not all(t <= camera_receipt_time <= decision_time for t in render_start_lowers)):
            raise ValueError('Four finite ordered original view and decision clocks required')
        with self.lock:
            if (self.last_decision is not None and decision_time <= self.last_decision
                    or self.last_view_bounds is not None and any(
                        new <= old for new, old in zip(render_start_lowers, self.last_view_bounds,
                                                       strict=True))):
                raise ValueError('Advancing decisions and per-view render clocks required')
            current_index = bisect_right(self.receipts, decision_time)-1
            if (current_index < 0 or self.samples[current_index].legacy.packet_id == self.first_packet):
                return None
            view_indices = [bisect_left(self.receipts, lower)-1 for lower in render_start_lowers]
            # Transactional encoders: malformed input cannot partly advance an episode.
            encoders = copy.deepcopy(self.encoders)
            now = encoders[4].encode(self.samples[current_index], episode_id=self.episode_id,
                                     decision_monotonic_seconds=decision_time)
            history = []
            for i, index in enumerate(view_indices):
                if index < 0 or self.samples[index].legacy.packet_id == self.first_packet:
                    history.append(None)
                else:
                    history.append(encoders[i].encode(self.samples[index], episode_id=self.episode_id,
                        decision_monotonic_seconds=camera_receipt_time))
            context = per_view_temporal_body_context(tuple(history), now,
                render_start_lowers=render_start_lowers, camera_receipt_time=camera_receipt_time,
                decision_time=decision_time)
            self.encoders = encoders
            self.last_decision, self.last_view_bounds = float(decision_time), render_start_lowers
            return context
