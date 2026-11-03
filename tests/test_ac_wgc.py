"""Synthetic exact-window capture boundary; no native Windows input or desktop acquisition."""

from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_capture import WindowState
from flybrain_drive.ac_wgc import WGCACCapture


class Native:
    state = WindowState(10, 42, "C:/Games/acs.exe", 10, 20, 8, 6)

    def probe(self, hwnd, *, require_foreground):
        assert hwnd == 10 and not require_foreground
        return self.state


class Factory:
    def __init__(self, **kwargs):
        assert kwargs["window_hwnd"] == 10
        assert kwargs["window_name"] is None and kwargs["monitor_index"] is None
        assert kwargs["cursor_capture"] is False
        self.callbacks, self.stopped = {}, False

    def event(self, method):
        self.callbacks[method.__name__] = method
        return method

    def start_free_threaded(self):
        return self

    def stop(self):
        self.stopped = True

    def wait(self):
        return None

    def is_finished(self):
        return self.stopped

    def emit(self, frame):
        self.callbacks["on_frame_arrived"](frame, self)


def frame(timestamp):
    pixels = np.full((8, 10, 4), [1, 2, 3, 255], np.uint8)
    return SimpleNamespace(frame_buffer=pixels, width=10, height=8, timespan=timestamp)


def test_wgc_copies_only_exact_target_bgra_and_preserves_unknown_exposure_clock():
    capture = WGCACCapture(10, native=Native(), capture_factory=Factory)
    original = frame(100)
    capture.capture_api.emit(original)
    original.frame_buffer.fill(0)
    sample = capture.capture(timeout=.02)
    np.testing.assert_array_equal(sample.rgb[0, 0], [3, 2, 1])
    assert sample.rgb.shape == (8, 10, 3) and not sample.rgb.flags.writeable
    metadata = sample.metadata()
    assert metadata["window"]["width"] == 8 and metadata["pixel_width"] == 10
    assert metadata["simulation_seconds"] is None and not metadata["render_freshness_verified"]
    assert metadata["capture_timespan_native"] == 100
    capture.close()
    assert capture.control.stopped


def test_wgc_rejects_producer_changes_and_reused_timestamps():
    native = Native()
    capture = WGCACCapture(10, native=native, capture_factory=Factory)
    capture.capture_api.emit(frame(100))
    capture.capture(timeout=.02)
    capture.capture_api.emit(frame(100))
    with pytest.raises(ValueError, match="native time"):
        capture.capture(timeout=.02)
    changed = WGCACCapture(10, native=native, capture_factory=Factory)
    native.state = WindowState(10, 43, "C:/Games/acs.exe", 10, 20, 8, 6)
    changed.capture_api.emit(frame(101))
    with pytest.raises(ValueError, match="identity"):
        changed.capture(timeout=.02)


def test_wgc_has_a_finite_deadline_and_no_old_frame_after_receipt_lower_bound():
    capture = WGCACCapture(10, native=Native(), capture_factory=Factory)
    with pytest.raises(TimeoutError, match="deadline"):
        capture.capture(timeout=.01)
    capture.capture_api.emit(frame(100))
    with pytest.raises(TimeoutError):
        capture.capture(timeout=.01, not_before=1e12)


def test_continuously_available_ineligible_frames_cannot_extend_deadline(monkeypatch):
    capture = WGCACCapture(10, native=Native(), capture_factory=Factory)
    ticks = count(10.0, .003)
    monkeypatch.setattr('flybrain_drive.ac_wgc.time.monotonic', lambda: next(ticks))
    # A busy producer can keep the queue nonempty; the deadline must still apply.
    monkeypatch.setattr(capture.frames, 'get', lambda **kwargs: (
        1.0, 1.0, 100, Native.state, np.zeros((8, 10, 3), dtype=np.uint8)))
    with pytest.raises(TimeoutError, match='deadline'):
        capture.capture(timeout=.01, not_before=20)
    assert capture.frames_discarded_before_bound > 0
