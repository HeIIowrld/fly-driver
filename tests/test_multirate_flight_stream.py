import numpy as np
import pytest
import torch

from flybrain_drive.flight_visual_body_temporal_context import PerViewTemporalContextOpponentFlightStream
from flybrain_drive.multirate_flight_stream import MultirateFlightStream
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def obs(packet, receipt, when):
    return NavigationBodyObservation('run', packet, receipt, when, np.zeros(21, np.float32),
                                     np.ones(21, bool), 'contract')


class Stub(PerViewTemporalContextOpponentFlightStream):
    def __init__(self):
        self.calls = []

    def decision(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1, 2), torch.zeros(1, 6)


def test_body_updates_reuse_visual_history_without_redating_and_reject_stale():
    core = Stub()
    stream = MultirateFlightStream(core)
    old = obs(1, 1., 1.02)
    current = obs(2, 1.025, 1.03)
    frame = per_view_temporal_body_context((old,)*4, current,
        render_start_lowers=(1.01,)*4, camera_receipt_time=1.02, decision_time=1.03)
    kwargs = dict(goals=torch.tensor([[0., .5]]), visual=torch.zeros(1, 6),
                  visual_valid=torch.zeros(1, 6, dtype=torch.bool),
                  scene_valid=torch.ones(1, 4, dtype=torch.bool))
    _, _, first = stream.update(when=1.03, body=current, new_scene=torch.ones(1, 576),
                                 new_visual_context=frame, **kwargs)
    assert first['previous_held_body_receipt'] is None and first['visual_replaced']
    _, _, next_event = stream.update(when=1.05, body=obs(3, 1.049, 1.05), **kwargs)
    assert next_event['previous_held_body_receipt'] == 1.025
    assert abs(next_event['previous_held_body_age_at_decision']-.025) < 1e-10
    assert not next_event['visual_replaced']
    assert core.calls[-1]['body_context'].visual_contexts[0] is old
    assert core.calls[-1]['body_context'].camera_receipt_time == 1.02
    assert torch.equal(core.calls[-1]['scene_features'], torch.ones(1, 576))
    with pytest.raises(ValueError, match='fresh'):
        stream.update(when=1.12, body=obs(4, 1.06, 1.12), **kwargs)
    assert len(core.calls) == 2 and stream.previous_time == 1.05
    with pytest.raises(ValueError, match='nonstale'):
        stream.update(when=1.17, body=obs(5, 1.169, 1.17), **kwargs)
    assert len(core.calls) == 2
