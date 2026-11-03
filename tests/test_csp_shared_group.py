import json

import numpy as np
import pytest

from flybrain_drive.ac_csp_vision import copy_shared_group


def frame():
    return dict(index=0, request_id='owned', generation='session', views=[
        dict(view_id=name, file=f'0000-{name}.bmp', image_writer='host_lossless_bmp',
             shared_texture_handle=index+1)
        for index, name in enumerate(('front', 'left', 'right', 'rear'))])


def test_ack_only_after_all_copies_and_cpu_images_survive_source_reuse(tmp_path):
    class GPU:
        calls = 0
        source = np.zeros((180, 320, 3), dtype=np.uint8)

        def read(self, handle):
            assert not list(tmp_path.iterdir())  # Neither image I/O nor early ack.
            self.calls += 1
            self.source.fill(handle)
            return self.source.copy(), dict(started_at=handle, finished_at=handle+.1)

    gpu = GPU()
    pixels, receipts, ack = copy_shared_group(gpu, frame(), tmp_path, host_clock=lambda: 5.)
    assert gpu.calls == 4 and len(receipts) == 4
    assert json.loads((tmp_path/'ack-0000.json').read_text()) == ack
    assert ack['copied_before_ack'] >= max(r['finished_at'] for r in receipts)
    gpu.source.fill(255)  # Producer is allowed to reuse its textures after ack.
    for value, image in enumerate(pixels.values(), start=1):
        assert np.all(image == value) and not image.flags.writeable
    assert len(list(tmp_path.iterdir())) == 1


def test_incomplete_group_never_releases_source(tmp_path):
    class GPU:
        calls = 0

        def read(self, handle):
            self.calls += 1
            if handle == 3:
                raise RuntimeError('copy failed')
            return np.zeros((180, 320, 3), dtype=np.uint8), {}

    with pytest.raises(RuntimeError, match='copy failed'):
        copy_shared_group(GPU(), frame(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_batched_group_ack_only_after_all_images_are_owned(tmp_path):
    class GPU:
        def read_group(self, handles):
            assert handles == [1, 2, 3, 4]
            assert not list(tmp_path.iterdir())
            return [(np.full((180, 320, 3), h, dtype=np.uint8),
                     dict(started_at=1., finished_at=2.)) for h in handles]
    pixels, receipts, ack = copy_shared_group(GPU(), frame(), tmp_path,
        host_clock=lambda: 3., batched=True)
    assert len(pixels) == len(receipts) == 4
    assert ack['copied_before_ack'] == 3
    assert all(not image.flags.writeable for image in pixels.values())


def test_batch_failure_cannot_publish_an_ack(tmp_path):
    class GPU:
        def read_group(self, _handles):
            raise TimeoutError('group failed')
    with pytest.raises(TimeoutError):
        copy_shared_group(GPU(), frame(), tmp_path, batched=True)
    assert not list(tmp_path.iterdir())


def test_bad_pixel_layout_never_releases_source(tmp_path):
    class GPU:
        def read(self, handle):
            return np.zeros((180, 319, 3), dtype=np.uint8), {}

    with pytest.raises(ValueError, match='shape and dtype'):
        copy_shared_group(GPU(), frame(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_failed_group_never_publishes_mmf_ack_either(tmp_path):
    acknowledgments = []

    class GPU:
        def read_group(self, _handles):
            raise TimeoutError('failed immutable copy')

    with pytest.raises(TimeoutError):
        copy_shared_group(GPU(), frame(), tmp_path, batched=True,
                          ack_writer=acknowledgments.append)
    assert acknowledgments == [] and not list(tmp_path.iterdir())
