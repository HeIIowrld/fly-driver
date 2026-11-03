"""Asynchronous image evidence and causal host association; no live native capture."""

import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_capture import WindowState
from flybrain_drive.ac_teacher_vision import TeacherVisionRecorder
from flybrain_drive.ac_wgc import WGCFrameSample
from flybrain_drive.graph import sha256


class Capture:
    def __init__(self):
        self.queue = queue.Queue()
        self.native = SimpleNamespace(user=SimpleNamespace(GetForegroundWindow=lambda: 0))
        self.closed = False

    def capture(self, *, timeout):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError('No frame') from None

    def close(self):
        self.closed = True


def wait_written(recorder, count):
    deadline = time.monotonic() + 2
    while recorder.count < count and recorder.error is None and time.monotonic() < deadline:
        time.sleep(.005)
    assert recorder.count == count


def test_only_committed_causal_frame_is_returned_with_hash_and_no_clock_claim(tmp_path):
    capture = Capture()
    recorder = TeacherVisionRecorder(tmp_path / 'vision', 10, capture_factory=lambda _: capture)
    window = WindowState(10, 42, 'C:/Games/acs.exe', 0, 0, 8, 6)
    try:
        for index in range(2):
            capture.queue.put(WGCFrameSample(index, 1. + index, 1.01 + index, 100 + index,
                window, np.full((6, 8, 3), index, dtype=np.uint8)))
        wait_written(recorder, 2)
        candidate = recorder.candidate_before(1.1)
        assert candidate['capture_id'] == 0  # Newer frame is in the future of this receipt.
        assert candidate['file_sha256'] == sha256(tmp_path / candidate['file'])
        assert not candidate['training_admissible']
        assert not candidate['exposure_synchronization_verified']
        with pytest.raises(ValueError, match='No committed preceding image'):
            recorder.candidate_before(.9)
        with pytest.raises(ValueError, match='No committed preceding image'):
            recorder.candidate_before(2.5)  # Stale frame is not silently reused.
    finally:
        result = recorder.close()
    assert result['frames'] == 2 and not result['worker_alive'] and capture.closed
    receipt = recorder.archive_to(tmp_path / 'archive')
    assert receipt['all_hashes_verified'] and receipt['source_retained']
    assert len(receipt['files']) == 4
    assert sha256(tmp_path / 'archive/rgb-000000.npy') == candidate['file_sha256']
    with pytest.raises(FileExistsError):
        recorder.archive_to(tmp_path / 'archive')


def test_storage_budget_failure_is_reported_to_controller(tmp_path):
    capture = Capture()
    recorder = TeacherVisionRecorder(tmp_path / 'vision', 10, max_bytes=1024,
                                     capture_factory=lambda _: capture)
    window = WindowState(10, 42, 'C:/Games/acs.exe', 0, 0, 32, 32)
    capture.queue.put(WGCFrameSample(0, 1., 1.01, 100, window,
                                   np.zeros((32, 32, 3), dtype=np.uint8)))
    deadline = time.monotonic() + 2
    while recorder.error is None and time.monotonic() < deadline:
        time.sleep(.005)
    with pytest.raises(RuntimeError, match='recorder failed'):
        recorder.candidate_before(1.1)
    result = recorder.close()
    assert result['status'] == 'failed' and result['frames'] == 0 and capture.closed
