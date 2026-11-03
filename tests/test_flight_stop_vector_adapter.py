import pytest
import torch

from flybrain_drive.flight_stop_vector_adapter import StopVectorAdapter


def test_neutral_stop_signal_and_separate_bounded_port_gradients():
    adapter = StopVectorAdapter(2)
    target = torch.tensor([0., 5.])
    initial = adapter(target)
    assert torch.equal(initial, torch.zeros(2, 2))
    initial[0, 0].backward()
    assert adapter.stop_delta.grad[0] != 0 and adapter.stop_delta.grad[1] == 0
    with torch.no_grad():
        adapter.stop_delta[:] = torch.tensor([2., -2.])
    signal = adapter(target)
    assert signal[0, 0] > 0 and signal[0, 1] < 0
    assert torch.all(signal.abs() < 2)
    assert torch.all(signal[1].abs() < signal[0].abs())


def test_invalid_target_or_port_count_rejected():
    with pytest.raises(ValueError):
        StopVectorAdapter(0)
    with pytest.raises(ValueError):
        StopVectorAdapter(2)(torch.tensor([-1.]))
