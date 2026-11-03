"""Privileged AC Gym teacher reconstruction. Never a fly actor input adapter.

Numerical conventions follow the reviewed MIT-licensed AssettoCorsaGym revision
a90db0a8b164164e11412857b396bf6874462976 (legacy) and
ef0fa502bdd257fc2affad9361c02b3dbdfe1632 (current); retained source/license are in
assets/external/ac-autonomy-research-20261027/dasGringuen__assetto_corsa_gym.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

CHANNELS = ("speed", "gap", "LastFF", "RPM", "accelX", "accelY", "actualGear",
            "angular_velocity_y", "local_velocity_x", "local_velocity_y",
            "SlipAngle_fl", "SlipAngle_fr", "SlipAngle_rl", "SlipAngle_rr")
PROFILES = {"legacy_20241008": 10., "current_2025": 20.}


def channel_scales(profile):
    if profile not in PROFILES:
        raise ValueError("An explicit reviewed teacher observation profile is required")
    return np.array([80, 10, 1, 10000, 5, 5, 8, np.pi, 80, PROFILES[profile], 25, 25, 25, 25])


def basic_teacher_state(state, profile):
    sensors = np.asarray(state["sensors"], dtype=float)
    current = np.array([state[key] for key in CHANNELS], dtype=float)
    if sensors.shape != (11,) or not np.isfinite(sensors).all() or not np.isfinite(current).all():
        raise ValueError("Teacher requires 14 finite state channels and 11 map rays")
    return np.r_[current / channel_scales(profile), sensors / 200.]


def encode_teacher_observation(state, history, curvature, *, profile, allow_initial_padding=False):
    """Pinned order, including privileged rays/curvature and measured-control history.

    A sampled record may omit its earlier reset/warm-up history: by default require
    three preceding records rather than pretending that unknown history was zero.
    """
    current = basic_teacher_state(state, profile)
    curve = np.asarray(curvature, dtype=float)
    if curve.shape != (12,) or not np.isfinite(curve).all() or state["out_of_track"] not in (0, 1):
        raise ValueError("Teacher curvature and out-of-track flag required")
    if len(history) < 3:
        if not allow_initial_padding:
            raise ValueError("Three preceding states required; initial history is unknown")
        controls, previous = np.zeros(9), np.zeros(75)
    else:
        recent = history[-3:]
        controls = np.r_[[r["steerAngle"] / 450. for r in recent],
                         [r["accStatus"] for r in recent], [r["brakeStatus"] for r in recent]]
        previous = np.concatenate([basic_teacher_state(r, profile) for r in recent])
    latest = [state["steerAngle"] / 450., state["accStatus"], state["brakeStatus"]]
    result = np.r_[current, float(state["out_of_track"]), curve / .1, controls, latest, previous]
    if result.shape != (125,) or not np.isfinite(result).all():
        raise ValueError("Teacher observation must have 125 finite entries")
    return result


class TeacherReferenceCurve:
    """Match the pinned spline/interpolation/look-ahead recipe, including its end crop."""

    def __init__(self, csv_path):
        from scipy.interpolate import UnivariateSpline, interp1d

        table = np.genfromtxt(csv_path, delimiter=",", names=True)
        x, y = np.asarray(table["pos_x"]), np.asarray(table["pos_y"])
        if len(x) < 6 or not np.isfinite([x, y]).all():
            raise ValueError("Finite reference geometry with at least six points required")
        distance = np.r_[0., np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        indices = np.flatnonzero(np.diff(distance) <= 0)
        stop = int(indices[0]) if len(indices) else -1
        if len(distance[:stop]) < 5:
            raise ValueError("Insufficient strictly increasing reference distance")
        if "curvature" in table.dtype.names:
            curve = np.asarray(table["curvature"])
        else:
            t = np.arange(len(x))
            weight = np.ones_like(x) / np.sqrt(.1)
            fx = UnivariateSpline(t, x, k=4, w=weight)
            fy = UnivariateSpline(t, y, k=4, w=weight)
            dx, dy = fx.derivative()(t), fy.derivative()(t)
            denominator = (dx*dx + dy*dy)**1.5
            if np.any(denominator <= 0):
                raise ValueError("Reference spline has zero tangent")
            curve = (dx * fy.derivative(2)(t) - dy * fx.derivative(2)(t)) / denominator
        if not np.isfinite(curve).all():
            raise ValueError("Reference curvature is not finite")
        grid = np.arange(0, distance[:stop][-1] + 3)
        # Explicitly preserve the upstream distance-channel interpolation too.
        self.distance = interp1d(distance[:stop], distance[:stop],
                                 fill_value="extrapolate")(grid)
        self.curvature = interp1d(distance[:stop], curve[:stop], fill_value="extrapolate")(grid)

    def segment(self, distance):
        if not np.isfinite(distance) or distance < 0 or distance - self.distance[-1] > 50:
            raise ValueError("Teacher look-ahead distance outside reference domain")
        selected = np.flatnonzero((self.distance >= distance) & (self.distance <= distance+300))
        if distance + 300 > self.distance[-1]:
            patch = distance + 300 - self.distance[-1]
            selected = np.r_[selected, np.flatnonzero((self.distance >= 0)
                                                       & (self.distance <= patch))]
        if len(selected) < 12:
            raise ValueError("Insufficient reference samples; no invented curvature padding")
        return self.curvature[selected][::len(selected)//12][:12]


class PrivilegedSACTeacher(nn.Module):
    def __init__(self, checkpoint, expected_sha256):
        super().__init__()
        path = Path(checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Teacher checkpoint hash mismatch")
        layers = []
        previous = 125
        for width in (256, 256, 256):
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 6))
        self.net = nn.Sequential(*layers)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(state, dict) or any(not isinstance(v, torch.Tensor)
                or not torch.isfinite(v).all() for v in state.values())):
            raise ValueError("Finite plain tensor teacher checkpoint required")
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def predict(self, observations):
        value = np.asarray(observations)
        if value.ndim != 2 or value.shape[1] != 125 or not np.isfinite(value).all():
            raise ValueError("Teacher input must be finite batch x 125")
        # Sensor observations are read-only; do not alias their NumPy backing buffer.
        raw = self.net(torch.tensor(value, dtype=torch.float32))
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite teacher inference")
        means, log_std = raw.chunk(2, dim=-1)
        return {"deterministic_policy_action": means.tanh().numpy(),
                "log_std": log_std.clamp(-20, 2).numpy()}


def relative_request(delta, previous, maximum_delta, lower, upper, *, product_dtype="float64"):
    """Offline request codec only; does not send or acknowledge applied controls."""
    arrays = [np.asarray(value, dtype=float) for value in
              (delta, previous, maximum_delta, lower, upper)]
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("Explicit finite three-channel action profile required")
    action, prior, scale, lo, hi = arrays
    if (np.any(np.abs(action) > 1) or np.any(scale <= 0) or np.any(lo >= hi)
            or np.any(prior < lo) or np.any(prior > hi)
            or product_dtype not in ("float32", "float64")):
        raise ValueError("Invalid relative action/request domain")
    product = action.astype(product_dtype) * scale.astype(product_dtype)
    return np.clip(prior + product, lo, hi)


def legacy_request_to_normalized_controls(request):
    """Pinned car_control request convention; independent pedals, never applied feedback.

    Source Controls.set_controls maps each pedal command with (command + 1) / 2.
    The native vJoy profile's axis direction and actual response still need measurement.
    """
    value = np.asarray(request, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
        raise ValueError('Bounded three-channel legacy absolute request required')
    result = value.copy()
    result[1:] = (result[1:] + 1) / 2
    return result
