"""Read-only AC window acquisition. Capture times are not render/simulation times."""

import ctypes
import hashlib
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WindowState:
    hwnd: int
    pid: int
    executable: str
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self):
        if (type(self.hwnd) is not int or self.hwnd <= 0 or type(self.pid) is not int
                or self.pid <= 0
                or any(type(value) is not int for value in
                       (self.left, self.top, self.width, self.height))
                or min(self.width, self.height) < 2
                or self.width * self.height > 16_777_216):
            raise ValueError("Explicit window identity and bounded client geometry required")
        if Path(self.executable).name.lower() not in ("acs.exe", "acs_x86.exe"):
            raise ValueError("Only original AC engine windows may be acquired")


@dataclass(frozen=True)
class CaptureSample:
    capture_id: int
    started_monotonic_seconds: float
    finished_monotonic_seconds: float
    window: WindowState
    rgb: np.ndarray
    acquisition_kind: str = "synthetic_fixture"

    def __post_init__(self):
        if (not isinstance(self.window, WindowState)
                or self.acquisition_kind not in ("synthetic_fixture", "mss_screen_client_area")
                or type(self.capture_id) is not int or self.capture_id < 0
                or not np.isfinite([self.started_monotonic_seconds,
                                    self.finished_monotonic_seconds]).all()
                or not 0 <= self.started_monotonic_seconds <= self.finished_monotonic_seconds):
            raise ValueError("Capture identity and ordered acquisition interval required")
        pixels = np.asarray(self.rgb)
        if pixels.dtype != np.uint8 or pixels.shape != (self.window.height, self.window.width, 3):
            raise ValueError("Acquired RGB must match client geometry")
        pixels = pixels.copy()
        pixels.setflags(write=False)
        object.__setattr__(self, "rgb", pixels)

    def metadata(self):
        return {"capture_id": self.capture_id,
                "acquisition_kind": self.acquisition_kind,
                "started_monotonic_seconds": self.started_monotonic_seconds,
                "finished_monotonic_seconds": self.finished_monotonic_seconds,
                "window": asdict(self.window), "rgb_sha256":
                hashlib.sha256(self.rgb.tobytes()).hexdigest(),
                "clock": "capture_process_time.monotonic",
                "engine_frame_id": None, "simulation_seconds": None,
                "render_freshness_verified": False, "occlusion_verified": False,
                "camera_calibration": None, "actor_stream_created": False}


class WindowCapture:
    """Injectable acquisition boundary; reject identity/geometry changes around the grab."""

    def __init__(self, probe, grab, clock=time.monotonic, *, acquisition_kind="synthetic_fixture"):
        self.probe, self.grab, self.clock = probe, grab, clock
        self.acquisition_kind = acquisition_kind
        self.next_capture_id = 0

    def capture(self):
        before = self.probe()
        started = self.clock()
        rgb = self.grab(before)
        finished = self.clock()
        if before != self.probe():
            raise ValueError("Window changed during acquisition; discard ambiguous pixels")
        result = CaptureSample(self.next_capture_id, started, finished, before, rgb,
                               self.acquisition_kind)
        self.next_capture_id += 1
        return result


class WindowsACWindow:
    """Verify PID/executable/foreground/client area, without moving or focusing a window."""

    def __init__(self):
        if os.name != "nt":
            raise OSError("Native AC window discovery requires Windows")
        from ctypes import wintypes

        self.w = wintypes
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name in ("IsWindowVisible", "IsIconic"):
            function = getattr(self.user, name)
            function.argtypes, function.restype = [wintypes.HWND], wintypes.BOOL
        self.user.GetForegroundWindow.restype = wintypes.HWND
        self.user.GetWindowThreadProcessId.argtypes = [wintypes.HWND,
                                                       ctypes.POINTER(wintypes.DWORD)]
        self.user.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.user.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user.GetClientRect.restype = wintypes.BOOL
        self.user.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user.ClientToScreen.restype = wintypes.BOOL
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
            wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL

    def probe(self, hwnd, *, require_foreground=True):
        if type(hwnd) is not int or hwnd <= 0:
            raise ValueError("Explicit positive HWND required")
        pid = self.w.DWORD()
        if not self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)):
            raise ValueError("Window no longer exists")
        handle = self.kernel.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if not handle:
            raise OSError("Cannot verify window producer process")
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            count = self.w.DWORD(len(buffer))
            if not self.kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(count)):
                raise OSError("Cannot verify producer executable")
            executable = buffer.value
        finally:
            self.kernel.CloseHandle(handle)
        if Path(executable).name.lower() not in ("acs.exe", "acs_x86.exe"):
            raise ValueError("Selected window does not belong to original AC engine")
        if not self.user.IsWindowVisible(hwnd) or self.user.IsIconic(hwnd):
            raise ValueError("AC window is hidden or minimized")
        if require_foreground and self.user.GetForegroundWindow() != hwnd:
            raise ValueError("AC window must already be foreground; no focus action is performed")
        rect, point = self.w.RECT(), self.w.POINT(0, 0)
        if (not self.user.GetClientRect(hwnd, ctypes.byref(rect))
                or not self.user.ClientToScreen(hwnd, ctypes.byref(point))):
            raise OSError("Cannot read client geometry")
        return WindowState(hwnd, pid.value, executable, point.x, point.y,
                           rect.right - rect.left, rect.bottom - rect.top)

    def discover(self):
        states = []
        callback_type = ctypes.WINFUNCTYPE(self.w.BOOL, self.w.HWND, self.w.LPARAM)

        @callback_type
        def visit(hwnd, _):
            try:
                states.append(self.probe(hwnd, require_foreground=False))
            except (OSError, ValueError):
                pass
            return True

        self.user.EnumWindows.argtypes = [callback_type, self.w.LPARAM]
        self.user.EnumWindows.restype = self.w.BOOL
        if not self.user.EnumWindows(visit, 0):
            raise OSError("Cannot enumerate game windows")
        return states


class MSSACCapture:
    """Optional MSS client-area grab; never falls back to a monitor or another window."""

    def __init__(self, hwnd):
        self.native = WindowsACWindow()
        self.native.probe(hwnd)  # No capture library/opening until AC process is verified.
        from mss import MSS

        self.sct = MSS()
        self.capture_boundary = WindowCapture(lambda: self.native.probe(hwnd), self._grab,
                                              acquisition_kind="mss_screen_client_area")

    def _grab(self, state):
        area = {name: getattr(state, name) for name in ("left", "top", "width", "height")}
        bounds = self.sct.monitors[0]
        if (state.left < bounds["left"] or state.top < bounds["top"]
                or state.left + state.width > bounds["left"] + bounds["width"]
                or state.top + state.height > bounds["top"] + bounds["height"]):
            raise ValueError("Client area is partially outside the desktop")
        shot = self.sct.grab(area)
        return np.frombuffer(shot.rgb, dtype=np.uint8).reshape(state.height, state.width, 3)

    def capture(self):
        return self.capture_boundary.capture()

    def close(self):
        self.sct.close()
