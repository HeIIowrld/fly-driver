import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_capture import (
    CaptureSample,
    MSSACCapture,
    WindowCapture,
    WindowsACWindow,
    WindowState,
)
from flybrain_drive.evidence import begin_run
from scripts.record_ac_capture import record_frames


def state(**overrides):
    return WindowState(**dict(dict(hwnd=17, pid=42, executable="/synthetic/acs.exe",
                                   left=1, top=2, width=4, height=2), **overrides))


def synthetic_capture(probe=None):
    readings = iter([1., 1.01, 1.02, 1.03, 1.04, 1.05])
    return WindowCapture(probe or state, lambda _: np.zeros((2, 4, 3), dtype=np.uint8),
                         lambda: next(readings))


def test_identical_pixels_are_distinct_acquisitions_without_invented_engine_exposures():
    capture = synthetic_capture()
    first, second = capture.capture(), capture.capture()
    assert (first.capture_id, second.capture_id) == (0, 1)
    assert first.metadata()["rgb_sha256"] == second.metadata()["rgb_sha256"]
    assert first.finished_monotonic_seconds <= second.started_monotonic_seconds
    for sample in (first, second):
        metadata = sample.metadata()
        assert metadata["engine_frame_id"] is None and metadata["simulation_seconds"] is None
        assert not metadata["render_freshness_verified"]
        assert not metadata["actor_stream_created"] and not sample.rgb.flags.writeable
        assert metadata["acquisition_kind"] == "synthetic_fixture"


@pytest.mark.parametrize("change", [dict(hwnd=18), dict(pid=99), dict(left=2), dict(width=8)])
def test_window_movement_or_process_or_geometry_change_discards_ambiguous_capture(change):
    states = iter([state(), state(**change)])
    capture = synthetic_capture(probe=lambda: next(states))
    with pytest.raises(ValueError, match="Window changed"):
        capture.capture()
    assert capture.next_capture_id == 0


def test_foreground_loss_raises_without_advancing_capture_sequence():
    calls = []

    def probe():
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("AC is no longer foreground")
        return state()

    capture = synthetic_capture(probe=probe)
    with pytest.raises(ValueError, match="foreground"):
        capture.capture()
    assert capture.next_capture_id == 0


def test_capture_shape_and_clock_validation_and_pixel_ownership():
    pixels = np.zeros((2, 4, 3), dtype=np.uint8)
    sample = CaptureSample(0, 1, 2, state(), pixels)
    pixels[:] = 100
    assert sample.rgb.max() == 0
    with pytest.raises(ValueError, match="interval"):
        CaptureSample(0, 2, 1, state(), pixels)
    with pytest.raises(ValueError, match="RGB"):
        CaptureSample(0, 1, 2, state(), pixels[..., :2])
    with pytest.raises(ValueError, match="original AC"):
        state(executable="/synthetic/explorer.exe")


def test_raw_recording_preserves_black_pixels_hashes_and_unknown_clocks(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.record_ac_capture.time.sleep", lambda _: None)
    result = record_frames(synthetic_capture(), tmp_path, frames=3, rate=25, max_bytes=10000)
    rows = [json.loads(line) for line in (tmp_path / "captures.jsonl").read_text().splitlines()]
    assert result["frames"] == 3 and not result["body_synchronization_verified"]
    assert len(rows) == 3
    for index, row in enumerate(rows):
        assert row["capture_id"] == index and row["engine_frame_id"] is None
        path = tmp_path / row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["file_sha256"]
        pixels = np.load(path, allow_pickle=False)
        assert pixels.shape == (2, 4, 3) and pixels.max() == 0
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["rgb_sha256"]


def test_budget_exhaustion_preserves_completed_frames(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.record_ac_capture.time.sleep", lambda _: None)
    one_size = 24 + 128  # Current NumPy uint8 HWC NPY header plus payload.
    budget = one_size + 24 + 255
    with pytest.raises(ValueError, match="budget"):
        record_frames(synthetic_capture(), tmp_path, frames=3, rate=25, max_bytes=budget)
    rows = (tmp_path / "captures.jsonl").read_text().splitlines()
    assert len(rows) == 1 and (tmp_path / "rgb-000000.npy").exists()
    assert not (tmp_path / "rgb-000001.npy").exists()


def test_mss_adapter_preserves_rgb_and_checks_desktop_bounds_without_a_desktop_grab():
    pytest.importorskip("mss")
    from mss.screenshot import ScreenShot

    window = state(width=2)
    monitor = dict(left=window.left, top=window.top, width=2, height=2)
    shot = ScreenShot(bytearray([1, 2, 3, 255, 0, 0, 0, 0] * 2), monitor)
    calls = []

    def grab(area):
        calls.append(area)
        return shot

    capture = MSSACCapture.__new__(MSSACCapture)
    capture.sct = SimpleNamespace(monitors=[dict(left=0, top=0, width=10, height=10)], grab=grab)
    rgb = capture._grab(window)
    np.testing.assert_array_equal(rgb[0], [[3, 2, 1], [0, 0, 0]])
    assert calls == [monitor]
    with pytest.raises(ValueError, match="outside"):
        capture._grab(state(left=-10))
    assert calls == [monitor]


def test_evidence_includes_capture_entrypoint_and_embedded_app_sources(tmp_path):
    run = begin_run(tmp_path / "run", {"scope": "synthetic provenance test"})
    sources = json.loads((run / "manifest.json").read_text(encoding="utf8"))["source_sha256"]
    assert "scripts/record_ac_capture.py" in sources
    assert "integrations/ac/flybrain_telemetry/flybrain_telemetry.py" in sources


@pytest.mark.skipif(os.name != "nt", reason="Native Win32 process verification")
def test_native_probe_rejects_a_real_non_game_window_without_acquiring_pixels():
    import ctypes
    from ctypes import wintypes

    user = ctypes.WinDLL("user32", use_last_error=True)
    user.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
        wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p]
    user.CreateWindowExW.restype = wintypes.HWND
    user.DestroyWindow.argtypes = [wintypes.HWND]
    user.DestroyWindow.restype = wintypes.BOOL
    hwnd = user.CreateWindowExW(0, "STATIC", "Flybrain owned hidden test fixture", 0,
                                0, 0, 4, 4, None, None, None, None)
    assert hwnd
    try:
        with pytest.raises(ValueError, match="does not belong"):
            WindowsACWindow().probe(hwnd)
    finally:
        assert user.DestroyWindow(hwnd)


def test_native_probe_refuses_invalid_identity():
    if os.name == "nt":
        with pytest.raises(ValueError, match="HWND"):
            WindowsACWindow().probe(0)
    else:
        with pytest.raises(OSError, match="Windows"):
            WindowsACWindow()
