"""Exact-HWND read-only Windows Graphics Capture; no desktop fallback or app input."""

import hashlib
import queue
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from .ac_capture import WindowsACWindow, WindowState


@dataclass(frozen=True)
class WGCFrameSample:
    capture_id: int
    started_monotonic_seconds: float
    finished_monotonic_seconds: float
    timespan_native: int
    window: WindowState
    rgb: np.ndarray

    def __post_init__(self):
        pixels = np.asarray(self.rgb)
        if (
            not isinstance(self.window, WindowState)
            or type(self.capture_id) is not int
            or self.capture_id < 0
            or type(self.timespan_native) is not int
            or self.timespan_native < 0
            or not np.isfinite(
                [self.started_monotonic_seconds, self.finished_monotonic_seconds]
            ).all()
            or not 0 <= self.started_monotonic_seconds <= self.finished_monotonic_seconds
            or pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or min(pixels.shape[:2]) < 2
            or pixels.shape[0] * pixels.shape[1] > 16_777_216
        ):
            raise ValueError("Explicit WGC window identity, native time and bounded RGB required")
        pixels = pixels.copy()
        pixels.setflags(write=False)
        object.__setattr__(self, "rgb", pixels)

    def metadata(self):
        return {
            "capture_id": self.capture_id,
            "window": asdict(self.window),
            "started_monotonic_seconds": self.started_monotonic_seconds,
            "finished_monotonic_seconds": self.finished_monotonic_seconds,
            "capture_timespan_native": self.timespan_native,
            "capture_timespan_units_verified": False,
            "acquisition_kind": "wgc_exact_hwnd_whole_window",
            "pixel_region": "whole_window; client/camera crop uncalibrated",
            "pixel_width": self.rgb.shape[1],
            "pixel_height": self.rgb.shape[0],
            "rgb_sha256": hashlib.sha256(self.rgb.tobytes()).hexdigest(),
            "engine_frame_id": None,
            "simulation_seconds": None,
            "render_freshness_verified": False,
            "occlusion_verified": False,
            "actor_stream_created": False,
        }


class WGCACCapture:
    """Capture only a verified AC engine HWND. A resize/producer change stops acquisition."""

    def __init__(self, hwnd, *, native=None, capture_factory=None):
        self.native = WindowsACWindow() if native is None else native
        self.hwnd = hwnd
        self.identity = self.native.probe(hwnd, require_foreground=False)
        if capture_factory is None:
            from .ac_windows_session import require_capture_session

            require_capture_session(self.identity.pid)
            from windows_capture import WindowsCapture

            capture_factory = WindowsCapture
        self.frames = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.error = None
        self.last_native_time = None
        self.next_capture_id = 0
        self.frames_arrived = 0
        self.frames_discarded_before_bound = 0
        self.capture_api = capture_factory(
            cursor_capture=False,
            draw_border=None,
            secondary_window=False,
            minimum_update_interval=40,
            monitor_index=None,
            window_name=None,
            window_hwnd=hwnd,
        )

        @self.capture_api.event
        def on_frame_arrived(frame, capture_control):
            try:
                self.frames_arrived += 1
                started = time.monotonic()
                before = self.native.probe(hwnd, require_foreground=False)
                pixels = np.asarray(frame.frame_buffer)
                timestamp = frame.timespan
                if (
                    before != self.identity
                    or pixels.dtype != np.uint8
                    or pixels.shape != (frame.height, frame.width, 4)
                    or type(timestamp) is not int
                    or timestamp < 0
                    or self.last_native_time is not None
                    and timestamp <= self.last_native_time
                ):
                    raise ValueError("WGC identity, geometry, pixels or native time changed/reset")
                # Copy within callback: the native zero-copy view must not outlive its owner.
                rgb = pixels[..., 2::-1].copy()
                after = self.native.probe(hwnd, require_foreground=False)
                if before != after:
                    raise ValueError("AC window changed during native frame copy")
                finished = time.monotonic()
                self.last_native_time = timestamp
                item = (started, finished, timestamp, before, rgb)
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(item)
            except Exception as error:
                self.error = error
                capture_control.stop()
                self.closed.set()

        @self.capture_api.event
        def on_closed():
            self.closed.set()

        self.control = self.capture_api.start_free_threaded()

    def capture(self, *, timeout=3.0, not_before=0.0):
        if not np.isfinite([timeout, not_before]).all() or not 0 < timeout <= 30 or not_before < 0:
            raise ValueError("Finite capture deadline and same-host receipt lower bound required")
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("No eligible WGC frame before capture deadline")
            if self.error is not None:
                raise self.error
            if self.closed.is_set():
                raise RuntimeError("WGC source closed; no replacement window selected")
            if self.control.is_finished():
                self.control.wait()
                raise RuntimeError("WGC worker finished without an eligible frame")
            try:
                item = self.frames.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("No eligible WGC frame before capture deadline") from None
                continue
            started, finished, timestamp, window, rgb = item
            if started < not_before:
                self.frames_discarded_before_bound += 1
                continue
            if window != self.native.probe(self.hwnd, require_foreground=False):
                raise ValueError("Window changed before accepting captured pixels")
            sample = WGCFrameSample(self.next_capture_id, started, finished, timestamp, window, rgb)
            self.next_capture_id += 1
            return sample

    def close(self):
        self.control.stop()
        self.control.wait()

    def diagnostics(self):
        return {
            "frames_arrived": self.frames_arrived,
            "frames_discarded_before_receipt_bound": self.frames_discarded_before_bound,
            "frames_returned": self.next_capture_id,
            "last_capture_timespan_native": self.last_native_time,
            "worker_finished": self.control.is_finished(),
        }
