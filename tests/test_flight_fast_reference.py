import pytest
import torch

from flybrain_drive.flight_circuit import FlightMotorReadout
from flybrain_drive.flight_fast_reference import relative_motor_reference


def test_reference_preserves_original_motor_reference_and_matches_actual_neural_readout():
    groups = {name: {'L': [str(2*n)], 'R': [str(2*n+1)]}
              for n, name in enumerate(('b1', 'b2', 'power'))}
    readout = FlightMotorReadout([str(n) for n in range(6)], groups, power_reference=.1)
    muscles = torch.tensor([[.02, .03, .04, .06, .2, .2],
                            [.03, .09, .05, .1, .2, .2]])
    reference = readout.lateral_reference.clone()
    references, shifts = relative_motor_reference(readout, muscles)
    assert torch.equal(references[0], reference)
    raw, _ = readout.preactivation(muscles)
    with torch.no_grad():
        for n in range(2):
            readout.lateral_reference.copy_(references[n])
            actual, _ = readout(muscles[n:n+1])
            expected = torch.tanh(raw[n, 0]-shifts[n])
            assert actual[0, 0] == pytest.approx(float(expected), abs=1e-7)
            assert actual[0, 1] == pytest.approx(float(raw[n, 1].tanh()), abs=1e-7)
        readout.lateral_reference.copy_(reference)
