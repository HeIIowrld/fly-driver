"""Explicit vJoy SDK requests, separate from native AC applied-control feedback.

No DLL is loaded or device acquired on import. UI input injection is not used.
Axis binding and inversion require an independently checked AC controller profile.
"""

import ctypes
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .graph import sha256

AXIS_MEMBERS = dict(
    zip(
        range(0x30, 0x39),
        (
            "wAxisX",
            "wAxisY",
            "wAxisZ",
            "wAxisXRot",
            "wAxisYRot",
            "wAxisZRot",
            "wSlider",
            "wDial",
            "wWheel",
        ),
        strict=True,
    )
)


class JoystickFrame(ctypes.Structure):
    """SDK JOYSTICK_POSITION_V2, Windows BYTE/LONG/DWORD and four-byte alignment."""

    _pack_ = 4
    _fields_ = (
        [("bDevice", ctypes.c_uint8)]
        + [
            (name, ctypes.c_int32)
            for name in (
                "wThrottle",
                "wRudder",
                "wAileron",
                "wAxisX",
                "wAxisY",
                "wAxisZ",
                "wAxisXRot",
                "wAxisYRot",
                "wAxisZRot",
                "wSlider",
                "wDial",
                "wWheel",
                "wAxisVX",
                "wAxisVY",
                "wAxisVZ",
                "wAxisVBRX",
                "wAxisVBRY",
                "wAxisVBRZ",
                "lButtons",
            )
        ]
        + [(name, ctypes.c_uint32) for name in ("bHats", "bHatsEx1", "bHatsEx2", "bHatsEx3")]
        + [(name, ctypes.c_int32) for name in ("lButtonsEx1", "lButtonsEx2", "lButtonsEx3")]
    )


@dataclass(frozen=True)
class AxisBinding:
    channel: str
    hid_usage: int
    inverted: bool

    def __post_init__(self):
        if (
            self.channel not in {"steering", "throttle", "brake"}
            or type(self.hid_usage) is not int
            or self.hid_usage not in AXIS_MEMBERS
            or type(self.inverted) is not bool
        ):
            raise ValueError(
                "Explicit normalized channel, supported HID axis and inversion required"
            )

    def encode(self, value, minimum, maximum):
        low = -1 if self.channel == "steering" else 0
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not low <= value <= 1
            or not -(2**31) <= minimum < maximum < 2**31
        ):
            raise ValueError("Finite bounded request and measured int32 axis range required")
        unit = (value - low) / (1 - low)
        if self.inverted:
            unit = 1 - unit
        return round(minimum + unit * (maximum - minimum))


def load_vjoy_sdk(path, expected_sha256):
    """Load one explicitly supplied SDK DLL after installation approval; never search PATH."""
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError("64-bit Windows vJoy SDK runtime required")
    path = Path(path).resolve(strict=True)
    if sha256(path) != expected_sha256:
        raise ValueError("vJoy DLL hash differs from reviewed deployment")
    sdk = ctypes.CDLL(str(path))  # The public SDK declares __cdecl, not WINAPI.
    u32, i32 = ctypes.c_uint32, ctypes.c_int32
    signatures = {
        "vJoyEnabled": ([], i32),
        "GetvJoyVersion": ([], ctypes.c_int16),
        "GetVJDStatus": ([u32], i32),
        "GetVJDButtonNumber": ([u32], i32),
        "AcquireVJD": ([u32], i32),
        "RelinquishVJD": ([u32], None),
        "GetVJDAxisExist": ([u32, u32], i32),
        "GetVJDAxisMin": ([u32, u32, ctypes.POINTER(i32)], i32),
        "GetVJDAxisMax": ([u32, u32, ctypes.POINTER(i32)], i32),
        "UpdateVJD": ([u32, ctypes.POINTER(JoystickFrame)], i32),
    }
    for name, (arguments, result) in signatures.items():
        method = getattr(sdk, name)
        method.argtypes, method.restype = arguments, result
    return sdk


class VJoySession:
    def __init__(self, sdk, device_id, bindings):
        self.bindings = tuple(bindings)
        if (
            type(device_id) is not int
            or not 1 <= device_id <= 16
            or len(self.bindings) != 3
            or any(not isinstance(b, AxisBinding) for b in self.bindings)
            or {b.channel for b in self.bindings} != {"steering", "throttle", "brake"}
            or len({b.hid_usage for b in self.bindings}) != 3
        ):
            raise ValueError("One explicit device and three distinct axis bindings required")
        self.sdk, self.device_id, self.owned, self.sequence = sdk, device_id, False, 0
        self.submit_attempts = 0
        self.ranges = {}

    def open(self):
        if self.owned:
            raise ValueError("Already owns this device")
        if not self.sdk.vJoyEnabled() or self.sdk.GetVJDStatus(self.device_id) != 1:
            raise RuntimeError("vJoy device must be enabled and FREE; never steal ownership")
        self.ranges = {}
        for axis in AXIS_MEMBERS:
            if not self.sdk.GetVJDAxisExist(self.device_id, axis):
                continue
            minimum, maximum = ctypes.c_int32(), ctypes.c_int32()
            if (
                not self.sdk.GetVJDAxisMin(self.device_id, axis, ctypes.byref(minimum))
                or not self.sdk.GetVJDAxisMax(self.device_id, axis, ctypes.byref(maximum))
                or minimum.value >= maximum.value
            ):
                raise RuntimeError("Cannot query valid device axis range")
            self.ranges[axis] = (minimum.value, maximum.value)
        if any(b.hid_usage not in self.ranges for b in self.bindings):
            raise RuntimeError("Configured control axis is missing")
        if not self.sdk.AcquireVJD(self.device_id):
            raise RuntimeError("Could not acquire explicitly selected virtual device")
        self.owned = True
        try:
            if self.sdk.GetVJDStatus(self.device_id) != 0:
                raise RuntimeError("Driver did not confirm ownership")
            return self.submit(0.0, 0.0, 1.0)
        except Exception:
            self.sdk.RelinquishVJD(self.device_id)
            self.owned = False
            raise

    def submit(self, steering, throttle, brake, *, pressed_buttons=()):
        if not self.owned:
            raise RuntimeError("Explicit device ownership required before actuation")
        if self.sdk.GetVJDStatus(self.device_id) != 0:
            raise RuntimeError("Driver ownership was lost; no request sent")
        if (not isinstance(pressed_buttons, (tuple, list))
                or any(type(b) is not int or not 0 <= b < 128 for b in pressed_buttons)
                or len(set(pressed_buttons)) != len(pressed_buttons)):
            raise ValueError('Explicit distinct zero-based joystick buttons required')
        if pressed_buttons:
            count = self.sdk.GetVJDButtonNumber(self.device_id)
            if not 1 <= count <= 128 or max(pressed_buttons) >= count:
                raise ValueError('Requested button not present on this vJoy device')
        values = dict(steering=steering, throttle=throttle, brake=brake)
        encoded = {
            b.hid_usage: b.encode(values[b.channel], *self.ranges[b.hid_usage])
            for b in self.bindings
        }
        frame = JoystickFrame()
        frame.bDevice = self.device_id
        for axis, (low, high) in self.ranges.items():
            setattr(frame, AXIS_MEMBERS[axis], encoded.get(axis, round((low + high) / 2)))
        for name in ("bHats", "bHatsEx1", "bHatsEx2", "bHatsEx3"):
            setattr(frame, name, 0xFFFFFFFF)
        for bank, name in enumerate(('lButtons', 'lButtonsEx1', 'lButtonsEx2', 'lButtonsEx3')):
            mask = sum(1 << (b % 32) for b in pressed_buttons if b // 32 == bank)
            setattr(frame, name, ctypes.c_int32(mask).value)
        requested = time.monotonic()
        self.submit_attempts += 1
        perf_before = time.perf_counter()
        if not self.sdk.UpdateVJD(self.device_id, ctypes.byref(frame)):
            raise RuntimeError("Driver rejected full joystick state request")
        perf_after = time.perf_counter()
        self.sequence += 1
        return {
            "request_sequence": self.sequence,
            "requested_monotonic_seconds": requested,
            "driver_call_started_perf_counter_seconds": perf_before,
            "driver_call_finished_perf_counter_seconds": perf_after,
            "requested_control": values,
            "driver_axis_values": encoded,
            "driver_pressed_buttons_zero_based": sorted(pressed_buttons),
            "driver_accepted": True,
            "game_applied": False,
        }

    def close(self):
        if self.owned:
            try:
                return self.submit(0.0, 0.0, 1.0)
            finally:
                self.sdk.RelinquishVJD(self.device_id)
                self.owned = False
