"""Experimental host-fresh hold of a geometry response, for offline evaluation only."""
import math

import numpy as np

from .ac_native_physics import NativePhysicsSample, decode_physics_prefix
from .flight_geometry_feedback import CausalGeometryVisualPort
from .visual_self_motion import StationaryMotionReference


class FreshHoldGeometryPort(CausalGeometryVisualPort):
    """Hold one computed visual response until its original receipt is 50 ms old.

    This is an engineering hypothesis, not a new observation, biological claim,
    exposure-time guarantee, or qualification for vehicle control.
    """

    def reset_episode(self):
        super().reset_episode()
        self.reference = StationaryMotionReference(mean_frames=3)
        self._held = None
        self._held_receipt = None

    def observe(self, motion, view_valid, native, *, capture_id, decision_time, camera_receipt_time):
        if capture_id != self.last_capture:
            self._held = self._held_receipt = None
            ready = super().observe(motion, view_valid, native, capture_id=capture_id,
                                    decision_time=decision_time, camera_receipt_time=camera_receipt_time)
            if ready:
                self._held = self.current.detach().clone()
                self._held_receipt = float(camera_receipt_time)
            return ready
        self.current = None
        if (type(capture_id) is not int or capture_id < 0 or type(decision_time) not in (int, float)
                or not math.isfinite(decision_time) or self.last_decision_time is None
                or decision_time <= self.last_decision_time):
            raise ValueError('Advancing decision clock and original capture identity required')
        self.last_decision_time = float(decision_time)
        if not isinstance(view_valid, np.ndarray) or view_valid.shape != (4,) or view_valid.dtype != np.bool_:
            raise ValueError('Four explicit boolean view validity values required')
        if not view_valid.all():
            self._held = self._held_receipt = None
            self.reason = 'camera_unavailable'
            return False
        if type(camera_receipt_time) not in (int, float) or not math.isfinite(camera_receipt_time):
            raise ValueError('Finite original camera receipt time required')
        if self._held is None:
            self.reason = 'repeated_camera_capture'
            return False
        if camera_receipt_time != self._held_receipt:
            raise ValueError('Repeated capture cannot change its original receipt')
        if not 0 <= decision_time - self._held_receipt <= .05:
            self._held = self._held_receipt = None
            self.reason = 'camera_stale'
            return False
        if not isinstance(native, NativePhysicsSample):
            self._held = self._held_receipt = None
            self.reason = 'body_unavailable'
            return False
        if decode_physics_prefix(native.prefix_bytes, native.received_monotonic_seconds) != native:
            raise ValueError('Body values must match their immutable original prefix bytes')
        if self.last_native is not None:
            if native.packet_id < self.last_native.packet_id:
                raise ValueError('Native packet sequence reset; explicit episode reset required')
            if native.packet_id == self.last_native.packet_id:
                if native.prefix_bytes != self.last_native.prefix_bytes:
                    raise ValueError('Repeated native packet has different original bytes')
                native = self.last_native
        self.last_native = native
        if not 0 <= decision_time - native.received_monotonic_seconds <= .05:
            self._held = self._held_receipt = None
            self.reason = 'body_stale'
            return False
        self.current = self._held.clone()
        self.reason = 'fresh_original_capture_held'
        return True


class FreshEvolvingGeometryPort(FreshHoldGeometryPort):
    """Recompute from current neural activity while the same image remains fresh.

    The caller must advance the original motion-vision state with that held image;
    repeating a stale activity tensor is not evidence of such an advance.
    """

    def observe(self, motion, view_valid, native, *, capture_id, decision_time, camera_receipt_time):
        ready = super().observe(motion, view_valid, native, capture_id=capture_id,
                                decision_time=decision_time, camera_receipt_time=camera_receipt_time)
        if ready and self.reason == 'fresh_original_capture_held':
            feature = self.pool.features(motion)
            if feature.shape != (6,) or not np.isfinite(feature).all():
                self.current = None
                raise ValueError('Finite current four-view neural activity required')
            vector = np.zeros(6, dtype=np.float32)
            vector[2] = feature[2] - self.reference.value[2]
            self.current = self.projection.weight.new_tensor(vector).unsqueeze(0)
            self.reason = 'fresh_capture_evolving_motion'
        return ready
