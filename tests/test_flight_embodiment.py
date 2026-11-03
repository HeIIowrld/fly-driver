"""Synthetic decoder contract tests, never original-graph/vehicle qualification."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.flight_embodiment import (
    EMBODIMENT_CONTRACT,
    REFERENCE_SCHEMA,
    SOURCE_ROLES,
    AnchorProvenance,
    EmbodimentReference,
    HashedSource,
    decode_embodiment,
)


def reference(dtype="float64", **changes):
    """Deliberately synthetic sources; no claim these hashes certify real files."""
    provenance = AnchorProvenance(
        sources=tuple(
            HashedSource(role, "synthetic-test/" + role, "a" * 64) for role in sorted(SOURCE_ROLES)
        ),
        tensor_dtype=dtype,
        numerical_scope="cpu_repeat",
        original_clock_sha256="b" * 64,
        original_body_mask_sha256="c" * 64,
        quiet_end_state_sha256="d" * 64,
        anchor_state_sha256="e" * 64,
        quiet_duration_seconds=1.0,
        step_speed_error_m_s=-100 / 3.6,
        anchor_elapsed_seconds=3.6801013999502175,
        fixed_time_registered_before_trajectory=True,
        quiet_to_step_state_continuous=True,
        original_graph_body_readout_frozen=True,
        all_original_samples_retained=True,
        anchor_power_observed=True,
        optimizer_updates_before_anchor=0,
        teacher_labels_used=False,
    )
    return replace(
        EmbodimentReference(
            schema=REFERENCE_SCHEMA,
            quiet_power=0.01,
            brake_anchor_power=0.002,
            original_readout_power_reference=0.01,
            quiet_raw_neural_intent=(0.0, 0.0),
            common_power_numerical_uncertainty=0.0,
            maximum_decoded_brake_parity_error=0.0,
            quiet_decoded_brake_error=0.0,
            provenance=provenance,
        ),
        **changes,
    )


def inputs(powers, dtype=torch.float64):
    p = torch.tensor(powers, dtype=dtype)
    summary = torch.full((*p.shape, 6), 0.1, dtype=dtype)
    summary[..., 4] = summary[..., 5] = p
    return summary


def test_preserves_original_intent_legacy_canonical_and_every_nonbrake_axis():
    intent = torch.tensor([[0.3, 0.2], [-0.4, -0.7], [0.0, 0.0]], dtype=torch.float64)
    muscles = inputs([0.02, 0.006, 0.01])
    original_intent, original_muscles = intent.clone(), muscles.clone()
    result = decode_embodiment(intent, muscles, reference=reference())
    expected_v1 = np.stack([canonical_controls(row) for row in intent.numpy()])
    assert np.array_equal(result.canonical_v1.numpy(), expected_v1)
    assert torch.equal(result.canonical_v2[..., :2], result.canonical_v1[..., :2])
    assert result.raw_neural_intent is intent
    assert torch.equal(intent, original_intent) and torch.equal(muscles, original_muscles)
    assert result.encoded_request is None and result.native_applied is None
    assert result.contract == EMBODIMENT_CONTRACT


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quiet_anchor_midpoint_and_saturation_are_explicit(dtype):
    ref = reference(str(dtype).split(".")[-1])
    muscles = inputs([0.01, 0.002, 0.006, 0.001, 0.02], dtype)
    result = decode_embodiment(torch.zeros((5, 2), dtype=dtype), muscles, reference=ref)
    assert result.canonical_v2[0].tolist() == [0.0, 0.0, 0.0]
    assert result.canonical_v2[1, 2].item() == 1.0
    assert torch.allclose(
        result.canonical_v2[:, 2], torch.tensor([0.0, 1.0, 0.5, 1.0, 0.0], dtype=dtype)
    )
    assert result.span.dtype == dtype and result.span.device == muscles.device
    assert torch.allclose(result.span, torch.tensor(0.008, dtype=dtype))


def test_only_common_power_controls_new_brake_not_intent_or_other_mn_groups():
    muscles = inputs([0.006, 0.006])
    muscles[1, :4] = torch.tensor([0.7, 0.8, 0.9, 1.0])
    intent = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float64)
    result = decode_embodiment(intent, muscles, reference=reference())
    assert result.canonical_v2[0, 2] == result.canonical_v2[1, 2]
    asymmetric = muscles.clone()
    asymmetric[0, 4:6] = torch.tensor([0.002, 0.01])
    other = decode_embodiment(intent, asymmetric, reference=reference())
    assert torch.allclose(result.canonical_v2[:, 2], other.canonical_v2[:, 2])


def test_interior_brake_gradient_reaches_power_mns_only():
    intent = torch.tensor([[0.2, -0.2]], dtype=torch.float64, requires_grad=True)
    muscles = inputs([0.006]).requires_grad_()
    result = decode_embodiment(intent, muscles, reference=reference())
    intent_grad, mn_grad = torch.autograd.grad(result.canonical_v2[:, 2].sum(), (intent, muscles))
    assert torch.equal(intent_grad, torch.zeros_like(intent))
    assert torch.equal(mn_grad[..., :4], torch.zeros_like(mn_grad[..., :4]))
    assert torch.allclose(mn_grad[..., 4:], torch.full((1, 2), -62.5, dtype=torch.float64))


def test_normal_clamp_has_no_straight_through_gradient_outside_span():
    muscles = inputs([0.001, 0.02]).requires_grad_()
    result = decode_embodiment(
        torch.zeros((2, 2), dtype=torch.float64), muscles, reference=reference()
    )
    (gradient,) = torch.autograd.grad(result.canonical_v2[:, 2].sum(), (muscles,))
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_autograd_gradcheck_with_original_throttle_and_new_brake():
    intent = torch.tensor([[0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    muscles = inputs([0.006]).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda i, m: decode_embodiment(i, m, reference=reference()).canonical_v2, (intent, muscles)
    )


def test_coactivation_is_preserved_and_reported_without_sample_or_loss_mask():
    intent = torch.tensor([[0.0, 0.4], [0.0, -0.4]], dtype=torch.float64)
    result = decode_embodiment(intent, inputs([0.006, 0.006]), reference=reference())
    assert result.pedal_coactivation.tolist() == [True, False]
    assert result.canonical_v2.shape == (2, 3)
    assert result.canonical_v2[0, 1].item() == 0.4
    assert result.canonical_v2[0, 2].item() == pytest.approx(0.5)
    assert "independent pedal coactivation" in result.contract.limitation


@pytest.mark.parametrize(
    "changes",
    [
        {"brake_anchor_power": None},
        {"brake_anchor_power": -0.001},
        {"brake_anchor_power": 0.01},
        {"brake_anchor_power": float("nan")},
        {"brake_anchor_power": 0.0099},
        {"quiet_power": 1e-7, "original_readout_power_reference": 1e-7, "brake_anchor_power": 0.0},
        {"common_power_numerical_uncertainty": 0.0001},
        {"maximum_decoded_brake_parity_error": 0.00010001},
        {"quiet_decoded_brake_error": 1e-12},
        {"quiet_raw_neural_intent": (0.0, 1e-12)},
        {"quiet_raw_neural_intent": (False, False)},
        {"original_readout_power_reference": 0.011},
        {"schema": "legacy"},
    ],
)
def test_rejects_invalid_missing_weak_or_inconsistent_reference(changes):
    with pytest.raises(ValueError):
        reference(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"quiet_duration_seconds": 0.5},
        {"step_speed_error_m_s": 0.0},
        {"anchor_elapsed_seconds": 0.0},
        {"fixed_time_registered_before_trajectory": False},
        {"quiet_to_step_state_continuous": False},
        {"original_graph_body_readout_frozen": False},
        {"all_original_samples_retained": False},
        {"anchor_power_observed": False},
        {"optimizer_updates_before_anchor": 1},
        {"teacher_labels_used": True},
        {"original_clock_sha256": "unverified"},
        {"sources": ()},
        {"numerical_scope": "CPU_also_proves_GPU"},
    ],
)
def test_rejects_unregistered_nonoriginal_or_optimization_selected_anchor(changes):
    with pytest.raises(ValueError):
        replace(reference().provenance, **changes)


def test_reference_and_sources_are_immutable_and_zero_anchor_is_not_implicit():
    ref = reference()
    with pytest.raises(FrozenInstanceError):
        ref.brake_anchor_power = 0.0
    with pytest.raises(FrozenInstanceError):
        ref.provenance.sources[0].path = "other"
    with pytest.raises(TypeError):
        EmbodimentReference(schema=REFERENCE_SCHEMA, quiet_power=0.01)
    assert reference(brake_anchor_power=0.0).brake_anchor_power == 0.0
    # This synthetic declared zero is accepted structurally, never established reachable.
    with pytest.raises(ValueError):
        HashedSource("graph_spec", "source", "not-a-SHA256")


@pytest.mark.parametrize("case", ["nan", "bounds", "dtype", "shape", "device", "empty"])
def test_input_contract_rejects_invalid_or_mismatched_samples(case):
    intent, muscles = torch.zeros((1, 2), dtype=torch.float64), inputs([0.006])
    if case == "nan":
        muscles[0, 4] = float("nan")
    elif case == "bounds":
        muscles[0, 4] = -1.0
    elif case == "dtype":
        muscles = muscles.float()
    elif case == "shape":
        muscles = muscles[:, :5]
    elif case == "device":
        muscles = torch.empty((1, 6), device="meta", dtype=torch.float64)
    elif case == "empty":
        intent, muscles = intent[:0], muscles[:0]
    with pytest.raises(ValueError):
        decode_embodiment(intent, muscles, reference=reference())


def test_leading_history_dimensions_preserve_all_samples():
    intent = torch.zeros((2, 3, 2), dtype=torch.float64)
    result = decode_embodiment(intent, inputs([[0.01, 0.006, 0.002]] * 2), reference=reference())
    assert result.canonical_v1.shape == result.canonical_v2.shape == (2, 3, 3)
    assert result.common_power.shape == (2, 3)
