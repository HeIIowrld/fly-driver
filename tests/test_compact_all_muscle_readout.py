import numpy as np
import pytest
import torch
from torch import nn

from flybrain_drive.compact_all_muscle_readout import CompactAllMuscleReadout, all_muscle_features
from flybrain_drive.compact_flight_pedal_readout import CompactFlightPedalReadout


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.register_buffer("training_reference", torch.zeros(6))
        self.base.register_buffer("training_scale", torch.ones(6))

    def forward(self, muscles):
        return muscles[:, :3]


class Phase(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = Parent()
        self.pedal_projection = nn.Linear(6, 3, bias=False)
        self.offset = nn.Linear(6, 2, bias=False)
        nn.init.zeros_(self.pedal_projection.weight)
        nn.init.zeros_(self.offset.weight)

    def features(self, muscles):
        return muscles


def test_all_muscle_residual_distinguishes_power_and_preserves_outside_output():
    compact = CompactFlightPedalReadout(
        Phase(), torch.zeros((1, 4)), torch.ones(1), torch.tensor([[.2], [-.3]]))
    center = torch.zeros((1, 6))
    coeff = torch.tensor([[-1.], [1.]])
    head = CompactAllMuscleReadout(compact, center, torch.ones(1), coeff)
    inputs = torch.tensor([[.2, 0., 0., 0., 0., 0.], [.2, 0., 0., 0., 1., 0.]])
    previous = compact(inputs)
    updated = head(inputs)
    assert torch.equal(updated[:, :1], previous[:, :1])
    assert torch.equal(updated[1], previous[1])
    assert updated[0, 1] < previous[0, 1] and updated[0, 2] > previous[0, 2]
    basis = all_muscle_features(inputs.numpy(), center.numpy(), np.ones(1))
    assert basis[0, 0] > 0 and basis[1, 0] == 0
    with pytest.raises(ValueError, match="six-muscle"):
        CompactAllMuscleReadout(compact, center[:, :4], torch.ones(1), coeff)
