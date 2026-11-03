"""Prototype actuator adapter; positive curvature/yaw means left turn."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleConfig:
    wheelbase: float = 1.6
    track: float = 1.1
    wheel_radius: float = 0.22
    chassis_mass: float = 210.0
    suspension_stiffness: float = 20000.0
    suspension_damping: float = 1000.0
    max_speed: float = 8.0
    max_curvature: float = 0.3
    max_torque: float = 160.0
    max_steer: float = 0.5
    max_residual_yaw_moment: float = 80.0
    max_torque_rate: float = 600.0
    calibration_status: str = "illustrative_unvalidated"

    def __post_init__(self):
        numeric = [v for v in vars(self).values() if isinstance(v, (int, float))]
        if not np.all(np.isfinite(numeric)) or min(numeric) <= 0:
            raise ValueError("Vehicle parameters must be finite and positive")
        if self.track * self.max_curvature / 2 >= 1:
            raise ValueError("Curvature exceeds the Ackermann geometry domain")


class IntentController:
    """No route truth. Encoder speed, gyro yaw and wheel speeds are the only feedback."""

    def __init__(self, config=None, actuation_mode="legacy"):
        self.config = config or VehicleConfig()
        if actuation_mode not in ("legacy", "off", "fixed", "residual"):
            raise ValueError("Unknown actuation_mode")
        self.actuation_mode = actuation_mode
        self.action_size = 2 if actuation_mode == "legacy" else 3
        self.steer = np.zeros(2)
        self.previous_torque = np.zeros(2)

    def reset(self):
        self.steer[:] = 0
        self.previous_torque[:] = 0

    def command(self, action, speed, yaw_rate, wheel_speed, dt, valid=True):
        c = self.config
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_size,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Intent must be {self.action_size} finite normalized values")
        action = np.clip(action, -1, 1)
        v_ref = (action[0] + 1) * c.max_speed / 2 if valid else 0.0
        curvature = action[1] * c.max_curvature if valid else 0.0
        desired = np.arctan(
            c.wheelbase
            * curvature
            / np.array([1 - c.track * curvature / 2, 1 + c.track * curvature / 2])
        )
        desired = np.clip(desired, -c.max_steer, c.max_steer)
        self.steer += np.clip(desired - self.steer, -1.5 * dt, 1.5 * dt)
        base = np.clip(45 * (v_ref - speed), 0, c.max_torque)
        differential = np.clip(35 * (speed * curvature - yaw_rate), -40, 40)
        residual_moment = 0.0
        if self.actuation_mode == "off":
            differential = 0.0
        if self.actuation_mode == "residual" and valid:
            residual_moment = action[2] * c.max_residual_yaw_moment
            differential += residual_moment * c.wheel_radius / c.track
        if self.actuation_mode == "legacy":
            rear_torque = np.clip([base - differential, base + differential], 0, c.max_torque)
        else:
            # Preserve a feasible common torque, then allocate differential within
            # individual motor torque and slew limits. No negative drive/regen assumed.
            lower = np.maximum(0, self.previous_torque - c.max_torque_rate * dt)
            upper = np.minimum(c.max_torque, self.previous_torque + c.max_torque_rate * dt)
            common = np.clip(base, np.mean(lower), np.mean(upper))
            difference_low = max(common - upper[0], lower[1] - common)
            difference_high = min(common - lower[0], upper[1] - common)
            allocated = np.clip(differential, difference_low, difference_high)
            rear_torque = np.array([common - allocated, common + allocated])
        if not valid:
            rear_torque[:] = 0
        self.previous_torque[:] = rear_torque
        # Separate wheel brakes, opposing measured spin. No ideal stationary brake claim.
        brake_level = np.clip(60 * (speed - v_ref), 0, 100)
        if v_ref < 0.1:
            brake_level = 100
        brakes = -brake_level * np.tanh(np.asarray(wheel_speed) / 0.5)
        return np.r_[self.steer, rear_torque, brakes], {
            "v_ref": float(v_ref),
            "curvature_ref": float(curvature),
            "rear_torque": rear_torque.tolist(),
            "actuation_mode": self.actuation_mode,
            "residual_yaw_moment_requested": float(residual_moment),
            "yaw_moment_requested": float(c.track / c.wheel_radius * differential),
            "yaw_moment_from_commanded_torques": float(
                c.track / (2 * c.wheel_radius) * (rear_torque[1] - rear_torque[0])
            ),
            "torque_constraint_active": bool(
                not np.allclose(rear_torque, [base - differential, base + differential])
            ),
            "speed_measured": float(speed),
            "yaw_rate_measured": float(yaw_rate),
            "yaw_rate_ref": float(speed * curvature),
            "valid": bool(valid),
        }
