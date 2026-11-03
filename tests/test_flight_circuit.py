from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive.flight_circuit import FlightCircuit, FlightMotorReadout, FlightScenePool
from scripts.train_flight_initial import substeps


def test_retinotopic_scene_preserves_camera_channel_and_position():
    uv = np.array(
        [(q, r) for q in range(-15, 16) for r in range(max(-15, -15 - q), min(15, 15 - q) + 1)]
    )
    pixels = np.column_stack([0.5 + uv[:, 1] / 32, 0.5 + (uv[:, 0] + uv[:, 1] / 2) / 32])
    pool = FlightScenePool(pixels, dict(rows=4, columns=4, sigma_image_uv=0.18))
    left = int(np.argmin(pixels[:, 0]))
    right = int(np.argmax(pixels[:, 0]))
    scene = torch.zeros(2, 4, 9, 721)
    scene[0, 1, 4, left] = 1.0
    scene[1, 1, 4, right] = 1.0
    result = pool(scene).reshape(2, 4, 9, 16)
    assert result[:, 1, 4].sum() > 0
    assert result[:, 0].sum() == 0 and result[:, 2:].sum() == 0
    assert result[0, 1, 4].argmax() % 4 == 0
    assert result[1, 1, 4].argmax() % 4 == 3
    with pytest.raises(ValueError):
        pool(scene[:, :3])


def test_motor_readout_separates_bilateral_difference_from_common_power():
    ids = ["b1L", "b1R", "b2L", "b2R", "pL", "pR"]
    groups = dict(
        b1=dict(L=["b1L"], R=["b1R"]), b2=dict(L=["b2L"], R=["b2R"]), power=dict(L=["pL"], R=["pR"])
    )
    readout = FlightMotorReadout(ids, groups, power_reference=0.2)
    output = torch.tensor(
        [
            [0.3, 0.3, 0.1, 0.1, 0.2, 0.2],
            [0.3, 0.4, 0.1, 0.1, 0.1, 0.3],
            [0.3, 0.3, 0.1, 0.1, 0.3, 0.3],
            [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    intent, muscles = readout(output)
    assert intent[0, 0] == 0 and intent[0, 1] == 0
    assert intent[1, 0] > 0 and abs(intent[1, 1]) < 1e-6
    assert intent[2, 1] > 0 and intent[3, 1] < 0
    assert muscles.shape == (4, 6)


def test_neutral_calibration_zeroes_intents_without_changing_motor_activity():
    ids = ["b1L", "b1R", "b2L", "b2R", "pL", "pR"]
    groups = dict(
        b1=dict(L=["b1L"], R=["b1R"]), b2=dict(L=["b2L"], R=["b2R"]), power=dict(L=["pL"], R=["pR"])
    )
    readout = FlightMotorReadout(ids, groups, power_reference=0.05)
    outputs = torch.tensor([[0.002, 0.003, 0.001, 0.0015, 0.004, 0.005]])
    _, muscles = readout(outputs)
    readout.calibrate_reference(muscles[0], turn_scale=0.001, forward_scale=0.0045)
    intent, after = readout(outputs)
    torch.testing.assert_close(intent, torch.zeros_like(intent), atol=1e-6, rtol=0)
    torch.testing.assert_close(after, muscles)
    logits, _ = readout.preactivation(outputs)
    torch.testing.assert_close(logits.tanh(), intent)


def test_preactivation_supervision_retains_gradient_when_bounded_output_saturates():
    ids = ["b1L", "b1R", "b2L", "b2R", "pL", "pR"]
    groups = dict(
        b1=dict(L=["b1L"], R=["b1R"]), b2=dict(L=["b2L"], R=["b2R"]), power=dict(L=["pL"], R=["pR"])
    )
    readout = FlightMotorReadout(ids, groups, power_reference=0.0)
    with torch.no_grad():
        readout.forward_log_gain.fill_(10.0)
    outputs = torch.full((1, 6), 0.1, requires_grad=True)
    intent, _ = readout(outputs)
    assert intent[0, 1] == 1.0
    logits, _ = readout.preactivation(outputs)
    logits.square().sum().backward()
    assert torch.isfinite(outputs.grad).all() and outputs.grad[:, 4:].abs().sum() > 0


def test_original_fractional_intervals_are_preserved_without_gap_clipping():
    for elapsed in (0.037123, 0.0658938999986276):
        ticks = substeps(elapsed)
        assert max(ticks) <= 0.01 and abs(sum(ticks) - elapsed) < 1e-12
    with pytest.raises(ValueError):
        substeps(0.081)


def test_fractional_rate_step_matches_declared_exponential_relaxation():
    alpha = float(np.float32(-np.expm1(-0.01 / 0.05)))

    class Rate:
        def _advance_tensor(self, drive, state):
            next_state = state + alpha * (torch.tanh(drive) - state)
            return next_state, next_state

    proxy = SimpleNamespace(spec=dict(dynamics=dict(dt=0.01, tau=0.05)), circuit=Rate())
    state, drive = torch.tensor([[0.2]]), torch.tensor([[0.4]])
    actual = FlightCircuit._advance_elapsed_tensor(proxy, drive, state, 0.003)
    expected = state + (-np.expm1(-0.003 / 0.05)) * (torch.tanh(drive) - state)
    torch.testing.assert_close(actual, expected)
