import numpy as np
import pytest

from flybrain_drive.flight_learning import (
    bounded_intent_loss,
    complete_history,
    signed_intent_targets,
    speed_goals,
    teacher_speed,
)


def test_target_speed_is_a_task_not_the_executed_throttle():
    np.testing.assert_array_equal(speed_goals(15, 3), [[0, .5]] * 3)
    np.testing.assert_array_equal(speed_goals(30, 2), [[0, 1]] * 2)
    for speed in (float("nan"), -1, 31, True):
        with pytest.raises(ValueError):
            speed_goals(speed, 2)


def test_gap_slicing_retiming_and_duplicate_clocks_do_not_repair_history():
    original = np.array([0., .04, .13, .17])
    for candidate in (original, original[2:], np.arange(4) * .04, [0, .04, .04, .08]):
        with pytest.raises(ValueError):
            complete_history(candidate, original)
    complete_history([0, .04, .08], [0, .04, .08])


def test_prefix_preserved_without_teacher_labels_and_signs_are_explicit():
    actions = np.array([[.4, .2, 0], [-.3, 0, .5], [.2, .3, 0]])
    valid = np.array([[False] * 3, [True] * 3, [True, False, True]])
    phases = ["student_request", "expert_observation", "expert_observation"]
    targets, masks = signed_intent_targets(actions, valid, phases)
    np.testing.assert_allclose(targets, [[-.4, .2], [.3, -.5], [-.2, .3]])
    np.testing.assert_array_equal(masks, [[False, False], [True, True], [True, False]])
    valid[0] = True
    with pytest.raises(ValueError, match="handoff"):
        signed_intent_targets(actions, valid, phases)


def test_two_intents_reject_simultaneous_pedals_instead_of_collapsing_them():
    with pytest.raises(ValueError, match="simultaneous"):
        signed_intent_targets([[0, .5, .2]], np.ones((1, 3), bool), ["expert_observation"])


def test_desired_command_labels_cannot_masquerade_as_applied_receipts():
    with pytest.raises(ValueError, match="receipt contract"):
        teacher_speed(dict(kind="geometric_recovery_fixed_task", maximum_speed_kmh=30))


def test_full_pedals_have_finite_loss_and_exact_proper_loss_gradient():
    import torch

    z = torch.tensor([[-20., 20.], [.4, -.7]], requires_grad=True)
    target = torch.tensor([[1., -1.], [.2, -.3]])
    scale = torch.tensor([.2, .1])
    mask = torch.tensor([[True, True], [False, True]])
    loss = bounded_intent_loss(z, target, mask, scale)
    assert torch.isfinite(loss)
    loss.backward()
    expected = (z.detach().tanh() - target) / scale.square() / mask.sum()
    expected[~mask] = 0
    torch.testing.assert_close(z.grad, expected)
    assert z.grad[0, 0] < 0 < z.grad[0, 1]
def test_turn_strata_preserve_missing_labels_and_exact_neutral_endpoints():
    import numpy as np

    from flybrain_drive.flight_learning import turn_groups

    target = np.array([-.1, -.01, 0., .01, .1, .9])
    valid = np.array([True, True, True, True, True, False])
    np.testing.assert_array_equal(turn_groups(target, valid), [1, 2, 2, 2, 0, -1])
    np.testing.assert_array_equal(target, [-.1, -.01, 0., .01, .1, .9])


