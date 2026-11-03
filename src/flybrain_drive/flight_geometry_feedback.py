"""Experimental causal geometry-to-flight visual ports; not an actor qualification."""
import math

import numpy as np
import torch
from torch import nn

from .ac_native_physics import NativePhysicsSample, decode_physics_prefix
from .visual_self_motion import StationaryMotionReference, VisualSelfMotionPool


class CausalGeometryVisualPort(nn.Module):
    """Add a learned visual-port correction after an evidenced stationary reference.

    Motion columns and geometry are engineering surrogates. Camera exposure timing,
    physical calibration and use as a vehicle actor require independent qualification.
    The caller must reset this boundary whenever the episode or source generation changes.
    """
    def __init__(self, base, pool, port_count):
        super().__init__()
        if not isinstance(base, nn.Module) or not isinstance(pool, VisualSelfMotionPool):
            raise ValueError('Original visual context and registered geometry pool required')
        if type(port_count) is not int or port_count < 1:
            raise ValueError('Explicit visual port count required')
        self.base = base
        self.pool = pool
        self.projection = nn.Linear(6, port_count, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.reset_episode()

    def reset_episode(self):
        self.reference = StationaryMotionReference(mean_frames=11)
        self.last_capture = -1
        self.last_decision_time = None
        self.last_native = None
        self.current = None
        self.reason = 'reference_unavailable'

    def observe(self, motion, view_valid, native, *, capture_id, decision_time, camera_receipt_time):
        """Update with one original observation; missing or stale data disables correction."""
        self.current = None
        if (type(capture_id) is not int or capture_id < 0 or capture_id < self.last_capture
                or type(decision_time) not in (float, int) or not math.isfinite(decision_time)
                or self.last_decision_time is not None and decision_time <= self.last_decision_time):
            raise ValueError('Nondecreasing capture ID and advancing finite decision clock required')
        self.last_decision_time = float(decision_time)
        if capture_id == self.last_capture:
            self.reason = 'repeated_camera_capture'
            return False
        self.last_capture = capture_id
        if (not isinstance(view_valid, np.ndarray) or view_valid.shape != (4,)
                or view_valid.dtype != np.bool_):
            raise ValueError('Four explicit boolean view validity values required')
        if not view_valid.all():
            if self.reference.value is None:
                self.reference.reset()
            self.reason = 'camera_unavailable'
            return False
        if (type(camera_receipt_time) not in (float, int)
                or not math.isfinite(camera_receipt_time)):
            raise ValueError('Finite original camera receipt time required')
        camera_age = decision_time - camera_receipt_time
        if not 0 <= camera_age <= .05:
            if self.reference.value is None:
                self.reference.reset()
            self.reason = 'camera_future' if camera_age < 0 else 'camera_stale'
            return False
        if not isinstance(native, NativePhysicsSample):
            if self.reference.value is None:
                self.reference.reset()
            self.reason = 'body_unavailable'
            return False
        try:
            verified = decode_physics_prefix(native.prefix_bytes, native.received_monotonic_seconds)
        except (ValueError, TypeError):
            verified = None
        if verified != native:
            raise ValueError('Body values must match their immutable original prefix bytes')
        if self.last_native is not None:
            if native.packet_id < self.last_native.packet_id:
                raise ValueError('Native packet sequence reset; explicit episode reset required')
            if native.packet_id == self.last_native.packet_id:
                if native.prefix_bytes != self.last_native.prefix_bytes:
                    raise ValueError('Repeated native packet has different original bytes')
                # Re-reading a shared packet cannot make its body values younger.
                native = self.last_native
        self.last_native = native
        age = decision_time-native.received_monotonic_seconds
        if not 0 <= age <= .05:
            if self.reference.value is None:
                self.reference.reset()
            self.reason = 'body_stale'
            return False
        feature = self.pool.features(motion)
        if feature.shape != (6,):
            raise ValueError('One original four-view motion response required')
        if self.reference.value is None:
            self.reference.observe(feature, native, capture_id=capture_id, decision_time=decision_time)
            self.reason = 'reference_unavailable'
            return False  # Training withheld all first eleven observations, including readiness.
        vector = np.zeros(6, dtype=np.float32)
        vector[2] = feature[2]-self.reference.value[2]
        self.current = torch.as_tensor(vector, device=self.projection.weight.device,
                                       dtype=self.projection.weight.dtype).unsqueeze(0)
        self.reason = 'ready'
        return True

    def forward(self, traces, body, valid):
        original = self.base(traces, body, valid)
        if self.current is None:
            return original
        if self.current.device != self.projection.weight.device:
            raise ValueError('Geometry input/device changed; explicit observation required')
        # One observation authorizes one decision only.  If the caller omits the
        # next observe(), never silently hold a stale visual correction.
        current = self.current
        self.current = None
        self.reason = 'observation_consumed'
        return original+.5*torch.tanh(self.projection(current))
