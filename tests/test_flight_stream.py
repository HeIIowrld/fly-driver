import copy

import numpy as np
import pytest
import torch

from flybrain_drive.flight_stream import CameraSequence, FlightStream, held_input_batches
from flybrain_drive.flyvis_camera import integration_intervals


def inputs():
    names = ['front', 'left', 'right', 'rear']
    camera = dict(generation='one', capture_id=0, received=10.01,
                  ideal_line_shown=False, aid_probe_only=False,
                  views=[dict(view_id=n, render_started_lower=10.) for n in names])
    pixels = {n: np.zeros((180, 320, 3), np.uint8) for n in names}
    return camera, pixels


def test_reused_capture_retains_source_clock_and_does_not_commit_a_preview():
    camera, pixels = inputs()
    order = CameraSequence()
    elapsed, signature = order.preview(10.02, camera, pixels)
    assert elapsed is None and order.clock is None
    order.commit(10.02, camera, signature)
    elapsed, reused = order.preview(10.06, camera, pixels)
    assert elapsed == pytest.approx(.04) and reused == signature
    assert order.clock == 10.02
    assert camera['received'] == 10.01


@pytest.mark.parametrize('change', ['pixels', 'received', 'render_time'])
def test_reused_capture_cannot_be_repaired_by_changing_bytes_or_source_times(change):
    camera, pixels = inputs()
    order = CameraSequence()
    _, signature = order.preview(10.02, camera, pixels)
    order.commit(10.02, camera, signature)
    if change == 'pixels':
        pixels['front'][0, 0, 0] = 1
    elif change == 'received':
        camera['received'] = 10.03
    else:
        camera['views'][0]['render_started_lower'] = 10.02
    with pytest.raises(ValueError, match='Repeated capture'):
        order.preview(10.06, camera, pixels)
    assert order.clock == 10.02


@pytest.mark.parametrize('change', ['generation', 'clock_reverse', 'gap', 'missing_view', 'aid'])
def test_neural_source_failures_never_implicitly_start_a_new_episode(change):
    camera, pixels = inputs()
    order = CameraSequence()
    _, signature = order.preview(10.02, camera, pixels)
    order.commit(10.02, camera, signature)
    camera = copy.deepcopy(camera)
    clock = 10.06
    if change == 'generation':
        camera['generation'] = 'two'
    elif change == 'clock_reverse':
        clock = 10.01
    elif change == 'gap':
        clock = 10.12
    elif change == 'missing_view':
        pixels.pop('rear')
    else:
        camera.pop('ideal_line_shown')
    with pytest.raises(ValueError):
        order.preview(clock, camera, pixels)
    assert order.clock == 10.02 and order.generation == 'one' and order.capture == 0


def test_stale_or_future_camera_is_rejected_before_neural_state_work():
    camera, pixels = inputs()
    for clock in [9.99, 10.2, float('nan'), True]:
        with pytest.raises(ValueError):
            CameraSequence().preview(clock, camera, pixels)


def test_failed_coupled_stream_cannot_resume_or_reset_itself():
    # No author network/GPU is needed to prove that invalid sources fail first.
    stream = object.__new__(FlightStream)
    stream.failed, stream.order = False, CameraSequence()
    camera, pixels = inputs()
    camera['aid_probe_only'] = True
    with pytest.raises(ValueError, match='non-diagnostic'):
        stream.advance(clock=10.02, camera=camera, pixels=pixels, native_sample=None)
    assert stream.failed and stream.order.clock is None
    camera['aid_probe_only'] = False
    with pytest.raises(RuntimeError, match='cannot resume'):
        stream.advance(clock=10.02, camera=camera, pixels=pixels, native_sample=None)


def test_author_default_device_side_effect_is_restored_even_on_load_failure(monkeypatch):
    stream = object.__new__(FlightStream)
    previous = torch.get_default_device()

    def failed_load(self, config):
        torch.set_default_device('meta')
        raise RuntimeError('Author initialization failed')

    monkeypatch.setattr(FlightStream, '_initialize_vision', failed_load)
    with pytest.raises(RuntimeError, match='initialization failed'):
        stream._load_vision({})
    assert torch.get_default_device() == previous


@pytest.mark.parametrize('elapsed', [.0001, .005, .01, .0399, .04, .04000001, .0799])
def test_batching_preserves_exact_original_internal_step_sequence(elapsed):
    batches = held_input_batches(elapsed)
    reconstructed = [dt for dt, count in batches for _ in range(count)]
    assert reconstructed == integration_intervals(elapsed)
    assert sum(dt * count for dt, count in batches) == pytest.approx(elapsed, abs=1e-12)
