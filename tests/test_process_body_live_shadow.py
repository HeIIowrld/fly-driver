import numpy as np
import pytest
import torch

from flybrain_drive.process_body_live_shadow import ProcessBodyAdapter


class Worker:
    def __init__(self):
        self.received = []

    def update(self, **kwargs):
        self.received.append(kwargs)
        return {'intent': np.array([[.1, .2]], dtype=np.float32),
                'info': {'previous_held_body_receipt': 9.0}}


def kwargs():
    return dict(when=10., body=object(), goals=torch.tensor([[0., .5]]),
                visual=torch.zeros(1, 6), visual_valid=torch.zeros(1, 6, dtype=torch.bool),
                scene_valid=torch.ones(1, 4, dtype=torch.bool),
                new_scene=np.zeros((1, 576), np.float32), new_visual_context=object())


def test_process_adapter_preserves_ordered_data_and_intent():
    worker = Worker()
    adapter = ProcessBodyAdapter(worker)
    args = kwargs()
    intent, muscles, info = adapter.update(**args)
    torch.testing.assert_close(intent, torch.tensor([[.1, .2]]))
    assert muscles is None and adapter.previous_time == 10.
    assert worker.received[0]['body'] is args['body']
    assert worker.received[0]['context'] is args['new_visual_context']
    assert info['previous_held_body_receipt'] == 9.


def test_process_adapter_rejects_changed_sensor_contract():
    worker = Worker()
    adapter = ProcessBodyAdapter(worker)
    args = kwargs()
    args['scene_valid'][0, 1] = False
    with pytest.raises(ValueError, match='sensor masks'):
        adapter.update(**args)
    assert not worker.received and adapter.previous_time is None
