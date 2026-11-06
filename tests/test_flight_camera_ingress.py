import hashlib
import queue
import time

import numpy as np
import pytest

from flybrain_drive.ac_csp_vision import NoFreshCSPFrame
from flybrain_drive.flight_camera_ingress import FlightCameraIngress
from flybrain_drive.flyvis_camera import VIEW_ORDER


class Camera:
    def __init__(self):
        self.frames = queue.Queue()

    def put(self, capture_id, age):
        received = time.perf_counter()
        pixels = np.ones((180, 320, 3), dtype=np.uint8)
        render = received - age
        sha = hashlib.sha256(pixels.tobytes()).hexdigest()
        identity = dict(capture_id=capture_id, generation="test", received=received)
        samples = {view: dict(identity, render_started_lower=render, rgb=pixels) for view in VIEW_ORDER}
        frame = dict(identity, source_kind="csp_four_camera", ideal_line_shown=False,
                     aid_probe_only=False, views=[dict(view_id=view, render_started_lower=render,
                                                     rgb_sha256=sha) for view in VIEW_ORDER])
        self.frames.put((frame, samples))
        return pixels

    def wait_for_candidate(self, *, max_age, timeout):
        assert max_age == .025
        try:
            frame, self.samples = self.frames.get(timeout=timeout)
        except queue.Empty as error:
            raise NoFreshCSPFrame("No test group") from error
        return frame, time.perf_counter(), 0.

    def samples_for(self, frame):
        return self.samples


def test_selected_pixels_are_original_and_frozen():
    camera = Camera()
    ingress = FlightCameraIngress(camera)
    try:
        pixels = camera.put(0, .002)
        frame, group = ingress.receive()
        pixels[:] = 2
        assert frame["received"] == group["received"] <= group["selected_at"]
        assert group["pixels"]["front"][0, 0, 0] == 1
    finally:
        ingress.close()


def test_ingress_rejects_original_selection_age():
    camera = Camera()
    ingress = FlightCameraIngress(camera)
    try:
        camera.put(0, .026)
        with pytest.raises(RuntimeError, match="ingress failed") as error:
            ingress.receive()
        assert "conservative age" in str(error.value.__cause__)
    finally:
        ingress.close()
