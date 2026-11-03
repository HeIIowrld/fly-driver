import pytest
import torch
from test_flight_fast_haltere import fixture

from flybrain_drive.flight_fast_graph import (
    interval_parameters,
    nominal_fast_step,
    prepared_interval_step,
)


@pytest.mark.parametrize('coupling,tau',
                         [(0., .05), (0., .005), (.1, .005), (.3, .005), (1., .005)])
def test_prepared_nominal_kernel_matches_verified_rate_solver_and_keeps_input_state(
        tmp_path, coupling, tau):
    _, raw = fixture(tmp_path)
    state = torch.tensor([[.1, .2, .3, .4], [.3, .2, .1, .4]])
    drive = torch.tensor([[.2, .3, 0., 0.]]).expand(2, -1)
    original = state.clone()
    g = torch.tensor(coupling).reshape(1, 1)
    alpha = -torch.expm1(torch.tensor(-.001)/torch.tensor(tau))
    actual = nominal_fast_step(raw, drive, state, g, alpha, baseline=coupling == 0 and tau == .05)
    expected = raw.advance(drive, state, .01, coupling=coupling, fast_tau=tau)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)
    assert torch.equal(state, original)


@pytest.mark.parametrize('elapsed', [.000013, .001, .0010001, .00317, .009999, .01])
@pytest.mark.parametrize('coupling,tau',
                         [(0., .05), (0., .005), (.1, .005), (.3, .005), (1., .005)])
def test_fractional_prepared_math_matches_original_solver_without_clock_quantization(
        tmp_path, elapsed, coupling, tau):
    _, raw = fixture(tmp_path)
    state = torch.tensor([[.1, .2, .3, .4], [.3, .2, .1, .4]])
    drive = torch.tensor([[.2, .3, 0., 0.]]).expand(2, -1)
    original = state.clone()
    count, microdt, ratio = interval_parameters(raw, elapsed)
    constants = torch.tensor([microdt, ratio], dtype=state.dtype)
    alpha = -torch.expm1(constants[0]/torch.tensor(tau).reshape(1, 1))
    net = torch.sparse.mm(raw.chemical_matrix, state.T).T+drive
    actual = prepared_interval_step(raw, net, state, torch.tensor(coupling).reshape(1, 1),
        alpha, baseline=coupling == 0 and tau == .05, count=count,
        slow_ratio=constants[1] if abs(ratio-1) > 1e-7 else None)
    expected = raw.advance(drive, state, elapsed, coupling=coupling, fast_tau=tau)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)
    assert torch.equal(state, original)
    assert -microdt*count == pytest.approx(elapsed, abs=1e-18)


@pytest.mark.parametrize('elapsed', [0., -.001, .011, float('nan'), float('inf')])
def test_fractional_parameters_reject_invalid_neural_time(tmp_path, elapsed):
    _, raw = fixture(tmp_path)
    with pytest.raises(ValueError, match='positive subinterval'):
        interval_parameters(raw, elapsed)
