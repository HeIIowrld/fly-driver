import torch

from flybrain_drive.flight_stop_narrow_adapter import NarrowStopVectorAdapter
from flybrain_drive.flight_stop_vector_adapter import StopVectorAdapter


def test_narrow_gate_preserves_stop_signal_and_suppresses_moving_goals():
    narrow, original = NarrowStopVectorAdapter(2), StopVectorAdapter(2)
    with torch.no_grad():
        narrow.stop_delta[:] = torch.tensor([1., -1.])
        original.stop_delta.copy_(narrow.stop_delta)
    target = torch.tensor([0., .5, 1.])
    actual, baseline = narrow(target), original(target)
    assert torch.equal(actual[0], baseline[0])
    assert torch.all(actual[1:].abs() < baseline[1:].abs() * .01)
    assert torch.all(actual.abs() < 2)


def test_narrow_gate_starts_neutral():
    assert torch.equal(NarrowStopVectorAdapter(3)(torch.tensor([0., 1.])), torch.zeros(2, 3))
