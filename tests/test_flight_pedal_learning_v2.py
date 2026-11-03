"""CPU pure-component tests; no dataset arrays/model checkpoint/graph/optimizer/GPU."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from flybrain_drive.actuator_codec import project_pedals
from flybrain_drive.flight_embodiment import decode_embodiment
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT
from flybrain_drive.flight_maneuver_learning import TrainableManeuverCurrents
from flybrain_drive.flight_pedal_learning_v2 import (
    CPU_READOUT_SHA256,
    CPU_REFERENCE_SHA256,
    PedalInputScope,
    RegisteredPedalReference,
    bind_registered_reference,
    install_registered_private_offsets,
    native_pedal_supervision,
    pedal_only_parameters,
    v2_pedal_imitation_loss,
    v2_pedal_tensors,
    validate_pedal_inputs,
)
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT

REFERENCE = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/'
    '20261030-t1734-cpu-mn-anchor-qualification/reference.json')
READOUT = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/'
    '20261030-t1741-matched-gpu-mn-qualification/cpu-readout-state.json')


@pytest.fixture
def binding():
    return bind_registered_reference(REFERENCE.read_bytes(), READOUT.read_bytes())


def muscles(binding, fractions):
    p0, anchor = binding.reference.quiet_power, binding.reference.brake_anchor_power
    power = torch.tensor([p0-f*(p0-anchor) for f in fractions], dtype=torch.float32)
    base = torch.zeros(len(fractions), 4)
    return torch.cat((base, power[:, None].expand(-1, 2)), -1)


def test_exact_registered_bytes_prevent_reanchor_or_private_reference_substitution(binding):
    assert binding.reference.quiet_power == .0014305992517620325
    assert binding.reference.brake_anchor_power == .00016035125008784235
    with pytest.raises(ValueError, match='SHA'):
        bind_registered_reference(REFERENCE.read_bytes()+b' ', READOUT.read_bytes())
    with pytest.raises(ValueError, match='correspondence'):
        RegisteredPedalReference(replace(binding.reference,
            brake_anchor_power=binding.reference.brake_anchor_power*.9),
            binding.cpu_reference_bytes, binding.cpu_readout_bytes)


def test_forward_bitwise_matches_frozen_v2_and_original_float64_codec(binding):
    mn = muscles(binding, [0., 1., .5, -.1, 1.1, .02, .01])
    intent = torch.tensor([[.1, x] for x in (0., -.2, -.3, .02, .015, .01, .019)],
                          dtype=torch.float32)
    old = decode_embodiment(intent, mn, reference=binding.reference)
    new = v2_pedal_tensors(intent, mn, binding=binding)
    expected = np.array([project_pedals(row)['encoded_request']
                         for row in old.canonical_v2.detach().numpy()])
    assert torch.equal(new['canonical'], old.canonical_v2)
    assert np.array_equal(new['encoded'].numpy(), expected)
    assert new['encoded'].dtype == torch.float64
    assert new['canonical'][0, 2] == 0 and new['canonical'][1, 2] == 1
    assert new['original_intent'] is intent and new['original_motor_summary'] is mn


def test_full_brake_kept_and_coactivation_only_masks_pedal_cells_without_rows(binding):
    native = torch.tensor([[0., 1.], [0., 0.], [.2, .3], [.4, 0.]], dtype=torch.float32)
    valid = torch.ones_like(native, dtype=torch.bool)
    before_native, before_valid = native.clone(), valid.clone()
    labels = native_pedal_supervision(native, valid)
    assert labels['canonical_lower'][0, 1] == labels['canonical_upper'][0, 1] == 1
    assert labels['supervision_valid'].tolist() == [[True, True], [True, True],
                                                   [False, False], [True, True]]
    assert labels['native_target'] is native
    assert torch.equal(native, before_native) and torch.equal(valid, before_valid)
    prediction = v2_pedal_tensors(torch.zeros(4, 2), muscles(binding, [.5]*4), binding=binding)
    loss = v2_pedal_imitation_loss(prediction, labels)
    assert loss['supervised_cells'].tolist() == [3, 3]
    assert loss['steering_loss_applied'] is False and loss['state_frames_dropped'] == 0


def test_subfloor_policy_is_explicit_and_zero_floor_preimages_are_distinct():
    native = torch.tensor([[0., .02], [.005, 0.], [.015, 0.], [.1, 0.]])
    valid = torch.ones_like(native, dtype=torch.bool)
    labels = native_pedal_supervision(native, valid)
    assert labels['canonical_upper'][0, 0] == pytest.approx(.01)
    assert labels['canonical_lower'][0, 1] == pytest.approx(.0101)
    assert not labels['supervision_valid'][1, 0] and not labels['supervision_valid'][2, 0]
    projected = native_pedal_supervision(native, valid, projection_policy='nearest_reachable')
    assert projected['target_for_supervision'][1, 0] == 0
    assert projected['target_for_supervision'][2, 0] == pytest.approx(.02)
    assert torch.equal(projected['native_target'], native)
    with pytest.raises(ValueError):
        native_pedal_supervision(native, valid, floor_preimage_margin=.005)


def test_local_interior_loss_gradient_reaches_power_mns_without_steering_loss(binding):
    intent = torch.tensor([[.2, -.1]], dtype=torch.float32, requires_grad=True)
    mn = muscles(binding, [.5]).requires_grad_()
    labels = native_pedal_supervision(torch.tensor([[0., 1.]]),
                                     torch.ones(1, 2, dtype=torch.bool))
    output = v2_pedal_tensors(intent, mn, binding=binding)
    loss = v2_pedal_imitation_loss(output, labels)['loss']
    gi, gm = torch.autograd.grad(loss, (intent, mn))
    assert torch.isfinite(gi).all() and torch.isfinite(gm).all()
    assert torch.equal(gi, torch.zeros_like(gi))  # throttlezero, no steering objective
    assert torch.equal(gm[:, :4], torch.zeros_like(gm[:, :4]))
    assert (gm[:, 4:] != 0).all()


def test_normal_clamp_and_hard_codec_have_no_straight_through_gradient(binding):
    mn = muscles(binding, [-.5, 1.05]).requires_grad_()
    intent = torch.tensor([[0., .005], [0., .015]], requires_grad=True)
    output = v2_pedal_tensors(intent, mn, binding=binding)
    gi, gm = torch.autograd.grad(output['encoded'][:, 1:].sum(), (intent, mn))
    assert torch.equal(gi, torch.zeros_like(gi))
    assert torch.equal(gm, torch.zeros_like(gm))


class FrozenReadout(nn.Module):
    def __init__(self, state):
        super().__init__()
        for key, value in state.items():
            if key in ('turn_log_gain', 'forward_log_gain', 'steering_mix'):
                self.register_parameter(key, nn.Parameter(torch.tensor(value, dtype=torch.float32),
                                                         requires_grad=False))
            else:
                dtype = torch.bool if type(value) is bool else (
                    torch.long if key.startswith(('b1_', 'b2_', 'power_L', 'power_R'))
                    else torch.float32)
                self.register_buffer(key, torch.tensor(value, dtype=dtype))


def test_only_private_cpu_offsets_install_after_exact_quiet_no_gain_or_state_reset(binding):
    readout = FrozenReadout(binding.readout_state)
    protected = {k: v.clone() for k, v in readout.state_dict().items()
                 if k not in ('power_reference', 'lateral_reference', 'reference_calibrated')}
    readout.power_reference.fill_(.01)
    readout.lateral_reference.add_(.03)
    readout.reference_calibrated.fill_(False)
    with pytest.raises(ValueError, match='quiet state'):
        install_registered_private_offsets(readout, binding, quiet_state_sha256='0'*64)
    receipt = install_registered_private_offsets(readout, binding,
        quiet_state_sha256=binding.reference.provenance.quiet_end_state_sha256)
    assert set(receipt['changed_neutral_keys']) == {
        'power_reference', 'lateral_reference', 'reference_calibrated'}
    assert receipt['state_advanced_or_reset'] is False
    assert all(torch.equal(readout.state_dict()[k], v) for k, v in protected.items())


def test_private_cpu_quiet_hash_cannot_admit_another_device(binding):
    readout = FrozenReadout(binding.readout_state).to('meta')
    with pytest.raises(ValueError, match='CPU-only'):
        install_registered_private_offsets(readout, binding,
            quiet_state_sha256=binding.reference.provenance.quiet_end_state_sha256)


def test_no_fabricated_yaw_or_full_steering_calibration_required():
    scope = PedalInputScope('prospectively_declared_zero_component', 'a'*64, 'b'*64)
    body, valid = torch.zeros(2, 21), torch.zeros(2, 21, dtype=torch.bool)
    valid[:, [4, 16]] = True
    command = torch.tensor([[0., 60/3.6], [0., 0.]])
    goals = torch.tensor([[0., 1.], [0., 1.]])
    original = valid.clone()
    result = validate_pedal_inputs(command, body, valid, goals, scope=scope,
        original_source_role='train', source_manifest_sha256='c'*64)
    assert result['fit_allowed'] is False and torch.equal(valid, original)
    with pytest.raises(ValueError):
        validate_pedal_inputs(command+.1, body, valid, goals, scope=scope,
            original_source_role='train', source_manifest_sha256='c'*64)
    with pytest.raises(ValueError):
        validate_pedal_inputs(command, body, valid, goals, scope=scope,
            original_source_role='diagnostic', source_manifest_sha256='c'*64)
    with pytest.raises(ValueError):
        PedalInputScope('separately_qualified_teacher_yaw_reference', 'a'*64, 'b'*64)


def test_two_dng_currents_have_finite_loss_gradients_in_synthetic_mn_chain_only(binding):
    # This bridge is a synthetic composition test, NOT original measured graph.
    model = nn.Module()
    model.command = TrainableManeuverCurrents([1, -1, 1, -1], 4)
    model.command.turn_delta.requires_grad_(False)
    selected = pedal_only_parameters(model)
    body = torch.zeros(1, 21)
    body[:, 16] = 20.
    valid = torch.zeros_like(body, dtype=torch.bool)
    valid[:, [4, 16]] = True
    command, goals = torch.tensor([[0., 10.]]), torch.tensor([[0., 1.]])
    _, drive = model.command(command, body, valid, goals,
        command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
    p0, anchor = binding.reference.quiet_power, binding.reference.brake_anchor_power
    power = .5*(p0+anchor)+p0*(drive.mean(1)-.01)
    mn = torch.cat((torch.zeros(1, 4), power[:, None].expand(-1, 2)), -1)
    gain = torch.tensor(binding.readout_state['forward_log_gain']).exp()
    raw = torch.stack((torch.zeros_like(power), (gain*(power-p0)).tanh()), -1)
    output = v2_pedal_tensors(raw, mn, binding=binding)
    labels = native_pedal_supervision(torch.tensor([[0., 1.]]),
                                     torch.ones(1, 2, dtype=torch.bool))
    gradients = torch.autograd.grad(v2_pedal_imitation_loss(output, labels)['loss'], selected)
    assert all(torch.isfinite(g).all() and g.abs().item() > 0 for g in gradients)
    assert model.command.turn_delta.grad is None
    model.command.turn_delta.requires_grad_(True)
    with pytest.raises(ValueError, match='Only2DNg'):
        pedal_only_parameters(model)


def test_stationary_zero_task_has_zero_dng_derivative_but_keeps_full_brake_loss(binding):
    model = nn.Module()
    model.command = TrainableManeuverCurrents([1, -1, 1, -1], 4)
    model.command.turn_delta.requires_grad_(False)
    body, command = torch.zeros(1, 21), torch.zeros(1, 2)
    valid = torch.zeros_like(body, dtype=torch.bool)
    valid[:, [4, 16]] = True
    _, drive = model.command(command, body, valid, torch.tensor([[0., 1.]]),
        command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
    gradients = torch.autograd.grad(drive.sum(), pedal_only_parameters(model))
    assert all(torch.equal(g, torch.zeros_like(g)) for g in gradients)
    # Exact private quiet is brake0; StopCounter's teacher brake1 is retained.
    # This does not claim every history-dependent stationary graph state is quiet.
    prediction = v2_pedal_tensors(torch.zeros(1, 2), muscles(binding, [0.]), binding=binding)
    labels = native_pedal_supervision(torch.tensor([[0., 1.]]),
                                     torch.ones(1, 2, dtype=torch.bool))
    loss = v2_pedal_imitation_loss(prediction, labels)
    assert prediction['canonical'][0, 2] == 0
    assert loss['supervised_cells'].tolist() == [1, 1]
    assert loss['loss'].item() == .5


def test_contract_keeps_full_objective_and_external_fit_requirements():
    plan = json.loads(Path('configs/flight-pedal-v2-component-training-contract-20261030.json')
                      .read_text(encoding='utf8'))
    assert plan['fit_allowed'] is False and plan['steering_loss'] is False
    assert plan['reference']['cpu_reference_sha256'] == CPU_REFERENCE_SHA256
    assert plan['reference']['cpu_readout_state_sha256'] == CPU_READOUT_SHA256
    assert len(plan['required_before_actual_fit']) >= 8
