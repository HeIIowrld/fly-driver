import copy

import numpy as np
import pytest
import torch

from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, ManeuverCurrents
from flybrain_drive.flight_maneuver_learning import (
    TrainableManeuverCurrents,
    _fingerprint,
    applied_axis_imitation_loss,
    applied_axis_supervision,
    canonical_actuator_tensors,
    validate_motor_training_episode,
)
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from scripts.qualify_trainable_maneuver_boundary import tensors_bitwise_equal

KWARGS = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)


def test_zero_init_matches_old_currents_exactly_at_60_100_and_both_turns():
    old = ManeuverCurrents([-1, 1, -1, 1], 4)
    new = TrainableManeuverCurrents([-1, 1, -1, 1], 4)
    body = torch.zeros(12, 21)
    body[:, 16] = torch.linspace(0, 100/3.6, 12)
    valid = torch.zeros_like(body, dtype=torch.bool)
    valid[:, [0, 1, 4, 16]] = True
    command = torch.tensor([[yaw, speed] for yaw in (-.5, 0., .5)
                            for speed in (0., 20/3.6, 60/3.6, 100/3.6)])
    goals = torch.tensor(racing_goal(100, 12))
    for lesion in (None, 'desired_yaw', 'speed_error', 'yaw_feedback'):
        for a, b in zip(old(command, body, valid, goals, lesion=lesion, **KWARGS),
                        new(command, body, valid, goals, lesion=lesion, **KWARGS), strict=True):
            assert torch.equal(a, b)
    with torch.no_grad():
        new.turn_delta[:] = torch.tensor([-100., 100.])
        new.power_delta.fill_(100.)
        new.speed_feedforward_delta.fill_(-100.)
    c = new.coefficients()
    assert torch.allclose(c['turn'], torch.tensor([.05, .8]))
    assert torch.allclose(c['power'], torch.tensor(8.))
    assert c['speed_feedforward'] == -.5
    command[:, 0] = command[:, 1] = body[:, 16] = 0
    steering, power = new(command, body, valid, goals, **KWARGS)
    assert not steering.any()
    assert torch.equal(power, old(command, body, valid, goals, **KWARGS)[1])


def test_codec_feasibility_gradient_is_explicit_and_hard_encoded_gap_has_no_fake_gradient():
    intent = torch.tensor([[.1, -.016], [0., -.005], [0., .025]], requires_grad=True)
    result = canonical_actuator_tensors(intent)
    torch.testing.assert_close(result['encoded'][:, 2], torch.tensor([.02, 0., 0.]))
    result['feasibility_loss'].backward()
    # .016brake moves toward.02; .005brake moves toward0; feasible.025gas stays.
    assert intent.grad[0, 1] > 0 and intent.grad[1, 1] < 0 and intent.grad[2, 1] == 0
    intent = torch.tensor([[0., -.016]], requires_grad=True)
    canonical_actuator_tensors(intent)['encoded'][:, 2].sum().backward()
    assert intent.grad[0, 1] == 0


def test_subfloor_teacher_axes_need_explicit_target_projection_and_coactivation_is_masked():
    native = torch.tensor([[0., 0., .016], [0., .005, 0.], [0., .03, .04], [.3, 0., 0.]])
    valid = torch.ones_like(native, dtype=torch.bool)
    raw, mask = native.clone(), valid.clone()
    reject = applied_axis_supervision(native, valid, steering_gain=1.40624985)
    assert reject['supervision_valid'].tolist() == [[True, True, False], [True, False, True],
                                                   [True, False, False], [False, True, True]]
    assert not reject['projection_applied'].any()
    assert reject['canonical_lower'][0, 2] == reject['canonical_upper'][0, 2] == 0
    projected = applied_axis_supervision(native, valid, steering_gain=1.40624985,
                                         projection_policy='nearest_reachable')
    torch.testing.assert_close(projected['target_for_supervision'][:, 2],
                               torch.tensor([.02, 0., .04, 0.]))
    assert projected['projection_applied'][0, 2] and projected['projection_applied'][1, 1]
    assert projected['supervision_valid'][0, 2] and projected['supervision_valid'][1, 1]
    assert not projected['supervision_valid'][2, 1:].any()
    assert torch.equal(native, raw) and torch.equal(valid, mask)


def test_floor_and_zero_are_set_valued_labels_not_arbitrary_neural_intent_points():
    native = torch.tensor([[0., 0., .02], [0., 0., 0.]])
    supervision = applied_axis_supervision(native, torch.ones_like(native, dtype=torch.bool),
                                           steering_gain=1.40624985)
    one = torch.tensor([[0., -.012], [0., -.003]], requires_grad=True)
    two = torch.tensor([[0., -.019], [0., -.009]], requires_grad=True)
    assert applied_axis_imitation_loss(one, supervision) == 0
    assert applied_axis_imitation_loss(two, supervision) == 0
    at_tie = torch.tensor([[0., -.01], [0., -.02]], requires_grad=True)
    loss = applied_axis_imitation_loss(at_tie, supervision)
    assert loss > 0
    loss.backward()
    assert at_tie.grad[0, 1] > 0 and at_tie.grad[1, 1] < 0


def episode():
    clock = np.array([1., 1.04, 1.08])
    body_valid = np.ones((3, 21), bool)
    arrays = dict(decision_clock=clock, native_receipt_clock=clock-.001,
        native_packet=np.arange(1, 4), control_native_receipt_clock=clock-.001,
        control_native_packet=np.arange(1, 4), body=np.zeros((3, 21)), body_valid=body_valid,
        physical_command=np.zeros((3, 2)), racing_goals=racing_goal(100, 3),
        native_control=np.zeros((3, 3)), native_control_valid=np.ones((3, 3), bool))
    source = dict(id='fixture', role='train', observations=3,
        original_clock_fingerprint=_fingerprint(clock),
        original_body_mask_fingerprint=_fingerprint(body_valid))
    metadata = dict(schema='flybrain.flight.motor_training_episode.v1', id='fixture', role='train',
        supervision_kind='observed_teacher_applied_axis_imitation',
        command_source='declared_external_task_and_teacher_yaw_reference',
        native_axis_calibration_verified=True, native_target_is_optimal_control=False)
    return metadata, source, arrays


def test_training_gate_keeps_full_original_clocks_masks_same_native_packet_and_source_role():
    meta, source, arrays = episode()
    assert validate_motor_training_episode(meta, source, arrays)['observations'] == 3
    for fault in ('development', 'dropped', 'retimed', 'changed_body_mask', 'future_body',
                  'different_packet', 'invented_optimal_label', 'above_ceiling'):
        m, s, a = copy.deepcopy((meta, source, arrays))
        if fault == 'development':
            s['role'] = 'development'
        elif fault == 'dropped':
            a = {k: v[:2] for k, v in a.items()}
        elif fault == 'retimed':
            a['decision_clock'][1] += .001
        elif fault == 'changed_body_mask':
            a['body_valid'][0, 12] = False
        elif fault == 'future_body':
            a['native_receipt_clock'][1] += .1
        elif fault == 'different_packet':
            a['control_native_packet'][0] += 1
        elif fault == 'above_ceiling':
            a['physical_command'][0, 1] = 101/3.6
        else:
            m['native_target_is_optimal_control'] = True
        with pytest.raises(ValueError):
            validate_motor_training_episode(m, s, a)


def test_frozen_sparse_graph_check_inspects_storage_without_unsupported_equal_or_densifying():
    original = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1., -2.], (2, 2)).coalesce()
    assert tensors_bitwise_equal(original, original.clone())
    changed = torch.sparse_coo_tensor([[0, 1], [1, 0]], [1., -2.1], (2, 2)).coalesce()
    assert not tensors_bitwise_equal(original, changed)
    csr = original.to_sparse_csr()
    assert tensors_bitwise_equal(csr, csr.clone())
    assert not tensors_bitwise_equal(csr, changed.to_sparse_csr())
    assert not tensors_bitwise_equal(original, csr)
