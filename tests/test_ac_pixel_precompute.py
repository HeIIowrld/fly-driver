import threading

import numpy as np
import pytest

from flybrain_drive.ac_pixel_precompute import PixelPrecomputeQueue


def test_slow_optional_cache_never_blocks_capture_or_exceeds_memory_budget():
    release = threading.Event()
    entered = threading.Event()
    pixels = np.zeros((180, 320, 3), np.uint8)
    pixels.setflags(write=False)
    calls = []

    def blocked(source):
        entered.set()
        assert release.wait(5)
        calls.append(source)

    queue = PixelPrecomputeQueue(blocked)
    try:
        for _ in range(12):
            queue.submit(pixels)
        assert entered.wait(1)
        assert queue.submitted == 8 and queue.skipped == 4
        assert queue.peak_pending == 8
    finally:
        release.set()
        queue.close()
    assert queue.completed == 8 and len(calls) == 8
    assert all(source is pixels for source in calls)


def test_cache_callback_error_is_not_silently_admitted():
    def failure(pixels):
        raise RuntimeError('pixel transduction failed')

    queue = PixelPrecomputeQueue(failure)
    pixels = np.zeros((2, 2, 3), np.uint8)
    pixels.setflags(write=False)
    queue.submit(pixels)
    with pytest.raises(RuntimeError, match='transduction failed'):
        queue.close()


def test_mutable_images_and_unreviewed_worker_budgets_are_rejected():
    with pytest.raises(ValueError, match='budget'):
        PixelPrecomputeQueue(lambda x: None, workers=16)
    queue = PixelPrecomputeQueue(lambda x: None)
    try:
        with pytest.raises(ValueError, match='immutable'):
            queue.submit(np.zeros((2, 2, 3), np.uint8))
    finally:
        queue.close()
