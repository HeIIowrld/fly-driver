from types import SimpleNamespace

import pytest
import torch

from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow


def stub():
    backbone = SimpleNamespace(neurons=3, spec={'dynamics': {'dt': .005}},
                               assert_body_boundary_protected=lambda: None)
    circuit = SimpleNamespace(backbone=backbone,
                              _advance_elapsed_tensor=lambda drive, state, dt: state + drive * dt,
                              _read_state_tensor=lambda state: state[:, :1])
    return SimpleNamespace(circuit=circuit)


def test_valid_advance_and_fractional_tail():
    drive = torch.tensor([[.1, .2, .3]])
    state = torch.tensor([[.1, .2, .3]])
    output, result = ExperimentalVisualBodyAirflow.advance(stub(), drive, state, .012)
    torch.testing.assert_close(result, state + drive * .012)
    torch.testing.assert_close(output, result[:, :1])


@pytest.mark.parametrize('which,value', [
    ('state', float('nan')), ('state', float('inf')), ('state', -.01),
    ('state', 1.01), ('drive', float('nan')), ('drive', float('inf')),
])
def test_invalid_values_still_rejected(which, value):
    state, drive = torch.full((1, 3), .5), torch.zeros((1, 3))
    (state if which == 'state' else drive)[0, 1] = value
    with pytest.raises(ValueError):
        ExperimentalVisualBodyAirflow.advance(stub(), drive, state, .01)


def test_shape_and_elapsed_still_rejected():
    with pytest.raises(ValueError):
        ExperimentalVisualBodyAirflow.advance(stub(), torch.zeros((1, 2)), torch.zeros((1, 3)), .01)
    with pytest.raises(ValueError):
        ExperimentalVisualBodyAirflow.advance(stub(), torch.zeros((1, 3)), torch.zeros((1, 3)), .081)
