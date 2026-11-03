"""Explicit task goals and complete-history contracts for offline flight learning."""

import numpy as np

GOAL_CONTRACT = {
    "schema": "flybrain.flight.speed_goal.v1",
    "fields": ["turn_bias", "desired_speed_fraction"],
    "speed_scale_kmh": 30.0,
    "turn_bias": "zero means follow the visually inferred road; no teacher steering input",
    "desired_speed_fraction": "declared task target/30kmh, not measured speed or throttle",
    "biological_claim": "engineering task projection into DNg02 boundary, not measured tuning",
    "checkpoint_compatibility": "new task semantics; no implicit loading of intent-goal v3 weights",
}


def speed_goals(speed_kmh, count):
    if (isinstance(speed_kmh, bool) or not np.isscalar(speed_kmh)
            or not np.isfinite(speed_kmh) or not 0 <= speed_kmh <= 30
            or type(count) is not int or count < 1):
        raise ValueError("Finite0..30kmh declared task and positive observation count required")
    goals = np.zeros((count, 2), dtype=np.float32)
    goals[:, 1] = speed_kmh / GOAL_CONTRACT["speed_scale_kmh"]
    return goals


def complete_history(clocks, expected_clocks):
    clocks, expected = np.asarray(clocks), np.asarray(expected_clocks)
    if (clocks.ndim != 1 or len(clocks) < 2 or not np.isfinite(clocks).all()
            or not np.array_equal(clocks, expected)
            or np.any(np.diff(clocks) <= 0) or np.any(np.diff(clocks) > .08)):
        raise ValueError("Complete original increasing history, no gap over80ms or retiming")


def signed_intent_targets(actions, valid, phases):
    actions, valid, phases = np.asarray(actions), np.asarray(valid), np.asarray(phases)
    if (actions.ndim != 2 or actions.shape[1] != 3 or actions.shape != valid.shape
            or valid.dtype != bool or phases.shape != (len(actions),)
            or not np.isfinite(actions).all() or np.any(np.abs(actions[:, 0]) > 1)
            or np.any((actions[:, 1:] < 0) | (actions[:, 1:] > 1))):
        raise ValueError("Original finite canonical three-head labels and boolean masks required")
    prefix = np.isin(phases, ["student_request", "handoff_wait_observation"])
    if valid[prefix].any():
        raise ValueError("Student and handoff state history must never receive teacher targets")
    if np.any(valid[:, 1] & valid[:, 2] & (actions[:, 1] > 0) & (actions[:, 2] > 0)):
        raise ValueError("Signed-drive codec cannot express simultaneous positive pedals")
    targets = np.column_stack([-actions[:, 0], actions[:, 1] - actions[:, 2]])
    masks = np.column_stack([valid[:, 0], valid[:, 1] & valid[:, 2]])
    return targets.astype(np.float32), masks


def teacher_speed(contract):
    required = dict(kind="engine_ai_future_applied_control_receipt_v1",
                    actor_steering_device_scale=.6, ai_steer_multiplier=.9,
                    target_receipt_horizon_seconds=.04, maximum_target_lateness_seconds=.025,
                    clock="host_monotonic_QPC_time.perf_counter")
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("One reviewed applied-control receipt contract required")
    speed = contract.get("ai_speed_hint_kmh")
    if type(speed) not in (int, float) or speed not in (15, 30):
        raise ValueError("Explicit15/30kmh engine-AI task required")
    return speed


def bounded_intent_loss(preactivation, target, valid, scale):
    """Proper logistic loss for tanh means, including exact full-pedal endpoints."""
    import torch
    import torch.nn.functional as functional

    if (preactivation.shape != target.shape or target.shape != valid.shape
            or target.shape[-1] != 2 or valid.dtype != torch.bool
            or scale.shape != (2,) or not torch.isfinite(target).all()
            or torch.any(target.abs() > 1) or not torch.isfinite(scale).all()
            or torch.any(scale <= 0) or not valid.any()):
        raise ValueError("Finite bounded targets, positive scales and nonempty masks required")
    # tanh(z) = 2sigmoid(2z)-1. This is a proper mean loss, not an altered neuron gradient.
    losses = functional.binary_cross_entropy_with_logits(
        2 * preactivation, (target + 1) / 2, reduction="none") / scale.square()
    return losses[valid].mean()


def turn_groups(target, valid, *, threshold=.01):
    """Label-only diagnostic/training strata: left0/right1/near-neutral2, missing-1.

    Never an actor input. Exact threshold endpoints belong to near-neutral; no
    labels are clipped, duplicated or removed to balance a training objective.
    """
    target, valid = np.asarray(target), np.asarray(valid)
    if (target.ndim != 1 or valid.shape != target.shape or valid.dtype != bool
            or not np.isfinite(target).all() or np.any(np.abs(target) > 1)
            or isinstance(threshold, bool) or not np.isfinite(threshold)
            or not 0 < threshold < 1):
        raise ValueError("Finite bounded turns, boolean mask and explicit threshold required")
    groups = np.full(target.shape, -1, dtype=np.int8)
    groups[valid & (target > threshold)] = 0
    groups[valid & (target < -threshold)] = 1
    groups[valid & (np.abs(target) <= threshold)] = 2
    return groups


def step_connected_loss(loss, optimizer, parameters):
    """A detached causal-boundary target must not trigger a fake momentum-only update."""
    import torch

    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training objective")
    if not loss.requires_grad:
        return False
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    return True
