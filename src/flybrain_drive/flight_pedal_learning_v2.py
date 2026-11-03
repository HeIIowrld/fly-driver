"""Pure scoped pedal supervision through the unchanged qualified v2 decoder.

No model/data loading, optimizer, simulator, GPU initialization or fit permission.
Caller must independently qualify whole new TRAIN histories and same-state
intent/MN provenance. A valid scope/reference is not dataset admission.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import torch

from .flight_embodiment import (
    AnchorProvenance,
    EmbodimentReference,
    HashedSource,
    decode_embodiment,
)

CPU_REFERENCE_SHA256 = '97fc57dff725808150972f4555a2b83d33a2fd096b90c30427cb668f89130175'
CPU_READOUT_SHA256 = '1647d180c7e66b1f98ac6902b729342cbacf5f598cf98580d71d192ae88dd032'
SOURCE_CHECKPOINT_SHA256 = '01a7a68372b76c68336f0403da48702a7f2f6f181f3e2567aecc6e9bd0e43116'
NEUTRAL_KEYS = frozenset({'power_reference', 'lateral_reference', 'reference_calibrated'})
YAW_POLICIES = frozenset({'prospectively_declared_zero_component',
                          'separately_qualified_teacher_yaw_reference'})


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _reference(data):
    provenance = data['provenance']
    return EmbodimentReference(**{**data,
        'quiet_raw_neural_intent': tuple(data['quiet_raw_neural_intent']),
        'provenance': AnchorProvenance(**{**provenance,
            'sources': tuple(HashedSource(**s) for s in provenance['sources'])})})


@dataclass(frozen=True)
class RegisteredPedalReference:
    """Exact CPU private reference bytes; external9-artifact proof still required."""

    reference: EmbodimentReference
    cpu_reference_bytes: bytes
    cpu_readout_bytes: bytes

    def __post_init__(self):
        if (type(self.cpu_reference_bytes) is not bytes or type(self.cpu_readout_bytes) is not bytes
                or hashlib.sha256(self.cpu_reference_bytes).hexdigest() != CPU_REFERENCE_SHA256
                or hashlib.sha256(self.cpu_readout_bytes).hexdigest() != CPU_READOUT_SHA256):
            raise ValueError('Exact registered CPU private reference/readout bytes required')
        data, readout = json.loads(self.cpu_reference_bytes), json.loads(self.cpu_readout_bytes)
        if (not isinstance(self.reference, EmbodimentReference)
                or self.reference != _reference(data)
                or self.reference.quiet_power != data['quiet_power']
                or self.reference.brake_anchor_power != data['brake_anchor_power']
                or self.reference.original_readout_power_reference != data['quiet_power']
                or self.reference.provenance.tensor_dtype != 'float32'
                or self.reference.provenance.quiet_duration_seconds != 1.
                or readout.get('cpu_reference_sha256') != CPU_REFERENCE_SHA256
                or readout.get('source_checkpoint_sha256') != SOURCE_CHECKPOINT_SHA256
                or readout['readout_state']['power_reference'] != self.reference.quiet_power
                or readout['readout_state']['reference_calibrated'] is not True):
            raise ValueError('Fixed CPU quiet/anchor/readout correspondence differs')

    @property
    def readout_state(self):
        return json.loads(self.cpu_readout_bytes)['readout_state']


def bind_registered_reference(cpu_reference_bytes, cpu_readout_bytes):
    """Pure bytes-only binding; does not read files or reconstruct a new anchor."""
    if (type(cpu_reference_bytes) is not bytes or type(cpu_readout_bytes) is not bytes
            or hashlib.sha256(cpu_reference_bytes).hexdigest() != CPU_REFERENCE_SHA256
            or hashlib.sha256(cpu_readout_bytes).hexdigest() != CPU_READOUT_SHA256):
        raise ValueError('Exact original CPU reference/readout SHA required; no reanchor')
    data = json.loads(cpu_reference_bytes)
    reference = _reference(data)
    return RegisteredPedalReference(reference, cpu_reference_bytes, cpu_readout_bytes)


def install_registered_private_offsets(readout, binding, *, quiet_state_sha256):
    """CPU-only: install3neutral buffers after original quiet; no recalibration.

    The external initializer must compute the registered quiet-state fingerprint
    using its original hasher. This function never advances or receives a state.
    The CPU hash never certifies an actual-device GPU quiet state.
    """
    if (not isinstance(binding, RegisteredPedalReference)
            or quiet_state_sha256 != binding.reference.provenance.quiet_end_state_sha256
            or any(p.requires_grad for p in readout.parameters())):
        raise ValueError('Exact retained quiet state and frozen readout required')
    actual, expected = readout.state_dict(), binding.readout_state
    if any(value.device.type != 'cpu' for value in actual.values()):
        raise ValueError('CPU-only private offsets; actual-device quiet proof required elsewhere')
    if set(actual) != set(expected) or not NEUTRAL_KEYS <= dict(readout.named_buffers()).keys():
        raise ValueError('Original complete readout state and neutral buffers required')
    targets = {k: torch.as_tensor(expected[k], dtype=v.dtype, device=v.device)
               for k, v in actual.items()}
    if any(v.shape != targets[k].shape for k, v in actual.items()):
        raise ValueError('Private readout shapes differ')
    for key, value in actual.items():
        if key not in NEUTRAL_KEYS and not torch.equal(value, targets[key]):
            raise ValueError('Protected readout gain/mix/port differs: '+key)
    changed = []
    with torch.no_grad():
        for key in NEUTRAL_KEYS:
            if not torch.equal(actual[key], targets[key]):
                actual[key].copy_(targets[key])
                changed.append(key)
    return dict(changed_neutral_keys=sorted(changed), cpu_readout_sha256=CPU_READOUT_SHA256,
        retained_quiet_state_sha256=quiet_state_sha256, quiet_steps=100,
        quiet_step_seconds=.01, state_advanced_or_reset=False, fit_allowed=False)


@dataclass(frozen=True)
class PedalInputScope:
    """Explicit declarations, no fabricated yaw or steering calibration."""

    yaw_policy: str
    command_declaration_sha256: str
    native_pedal_calibration_sha256: str
    teacher_yaw_reference_sha256: str | None = None

    def __post_init__(self):
        if (self.yaw_policy not in YAW_POLICIES or not _sha(self.command_declaration_sha256)
                or not _sha(self.native_pedal_calibration_sha256)
                or self.yaw_policy == 'separately_qualified_teacher_yaw_reference'
                and not _sha(self.teacher_yaw_reference_sha256)
                or self.yaw_policy == 'prospectively_declared_zero_component'
                and self.teacher_yaw_reference_sha256 is not None):
            raise ValueError('Explicit yaw and pedal-only calibration provenance required')


def validate_pedal_inputs(command, body, body_valid, racing_goals, *, scope,
                          original_source_role, source_manifest_sha256):
    if (not isinstance(scope, PedalInputScope) or original_source_role != 'train'
            or not _sha(source_manifest_sha256)):
        raise ValueError('New original TRAIN and explicit scoped command provenance required')
    n = len(body)
    if (n < 1 or body.shape != (n, 21) or command.shape != (n, 2)
            or body_valid.shape != body.shape or body_valid.dtype != torch.bool
            or racing_goals.shape != (n, 2)
            or not all(x.device == body.device for x in (command, body_valid, racing_goals))
            or not all(x.dtype == body.dtype for x in (command, racing_goals))
            or body.dtype != torch.float32
            or not all(torch.isfinite(x).all() for x in (body, command, racing_goals))
            or not body_valid[:, [4, 16]].all() or torch.any(body[:, 16] < 0)
            or torch.any(command[:, 0].abs() > 2) or torch.any(command[:, 1] < 0)
            or torch.any(racing_goals[:, 0] != 0)
            or torch.any((racing_goals[:, 1] <= 0) | (racing_goals[:, 1] > 3.5))
            or torch.any(command[:, 1] > racing_goals[:, 1]*100/3.6+1e-5)
            or scope.yaw_policy == 'prospectively_declared_zero_component'
            and torch.any(command[:, 0] != 0)):
        raise ValueError('Original body21/masks and explicit physical yaw/speed required')
    return dict(scope='conditional_pedal_component_only', steering_loss=False,
        full_manifest_and_clock_authority_verification_still_required=True,
        yaw_policy=scope.yaw_policy, source_role=original_source_role, fit_allowed=False)


def pedal_only_parameters(model):
    """Validate flags and return exactly2existing parameters, without changing them."""
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    names = ('command.power_delta', 'command.speed_feedforward_delta')
    if set(active) != set(names) or any(active[name].numel() != 1 for name in names):
        raise ValueError('Only2DNg scalars may require gradients; DNa/backbone/readout frozen')
    return tuple(active[name] for name in names)


def v2_pedal_tensors(intent, motor_summary, *, binding, minimum_positive=.02):
    """Exact frozen decoder plus hard codec; no residual/STE/new anchor."""
    if not isinstance(binding, RegisteredPedalReference) or minimum_positive != .02:
        raise ValueError('Registered fixed reference and reviewed pedal floor .02 required')
    decoded = decode_embodiment(intent, motor_summary, reference=binding.reference)
    canonical = decoded.canonical_v2
    # Original numpy project_pedals explicitly converts the request to float64.
    # Match its cutoff/floor decisions exactly, including float32(.02)<.02.
    codec_values = canonical.to(torch.float64)
    pedals = codec_values[..., 1:]
    gap = (pedals > 0) & (pedals < minimum_positive)
    nearest = torch.where(pedals > minimum_positive/2,
                          torch.full_like(pedals, minimum_positive), torch.zeros_like(pedals))
    encoded_pedals = torch.where(gap, nearest, pedals)
    encoded = torch.cat((codec_values[..., :1], encoded_pedals), -1)
    distance = (pedals-encoded_pedals.detach()).square().sum(-1)
    return dict(canonical=canonical, encoded=encoded, canonical_v1=decoded.canonical_v1,
        common_power=decoded.common_power, span=decoded.span, pedal_in_gap=gap,
        squared_projection_distance=distance, feasibility_loss=distance.mean(),
        original_intent=intent, original_motor_summary=motor_summary)


def native_pedal_supervision(native_pedals, original_valid, *,
                             projection_policy='reject_unreachable',
                             floor_tolerance=1e-4, floor_preimage_margin=1e-4):
    """2axis targets only; original native values/masks remain unmodified."""
    if (native_pedals.ndim != 2 or native_pedals.shape[1] != 2 or len(native_pedals) < 1
            or not torch.is_floating_point(native_pedals) or native_pedals.requires_grad
            or not torch.isfinite(native_pedals).all()
            or torch.any((native_pedals < 0) | (native_pedals > 1))
            or original_valid.shape != native_pedals.shape or original_valid.dtype != torch.bool
            or original_valid.device != native_pedals.device
            or projection_policy not in ('reject_unreachable', 'nearest_reachable')
            or floor_tolerance != 1e-4 or floor_preimage_margin != 1e-4):
        raise ValueError('Original native pedal axes/masks and explicit codec policy required')
    unreachable = (native_pedals > 0) & (native_pedals < .02)
    nearest = torch.where(native_pedals > .01,
                          torch.full_like(native_pedals, .02), torch.zeros_like(native_pedals))
    reachable = torch.where(unreachable, nearest, native_pedals)
    target = reachable if projection_policy == 'nearest_reachable' else native_pedals
    valid = original_valid.clone()
    if projection_policy == 'reject_unreachable':
        valid &= ~unreachable
    coactivation = (native_pedals > 0).all(-1)
    valid &= ~coactivation[:, None]
    zero, floor = target == 0, (target-.02).abs() <= floor_tolerance
    lower = torch.where(zero, 0., torch.where(floor, .01+floor_preimage_margin, target))
    upper = torch.where(zero, .01, torch.where(floor, .02, target))
    return dict(native_target=native_pedals, original_valid=original_valid,
        reachable_projected_native_target=reachable, target_for_supervision=target,
        native_unreachable_pedals=unreachable, projection_applied=target != native_pedals,
        native_pedal_coactivation=coactivation, supervision_valid=valid,
        canonical_lower=lower, canonical_upper=upper, projection_policy=projection_policy)


def v2_pedal_imitation_loss(prediction, supervision):
    """Set-valued v2pedal imitation only; no steering loss/feasibility objective."""
    pedals = prediction['canonical'][..., 1:]
    lower, upper, valid = (supervision[k] for k in (
        'canonical_lower', 'canonical_upper', 'supervision_valid'))
    if (pedals.ndim != 2 or pedals.shape[1] != 2
            or not torch.isfinite(pedals).all() or torch.any((pedals < 0) | (pedals > 1))
            or any(x.shape != pedals.shape for x in (lower, upper, valid))
            or valid.dtype != torch.bool
            or any(x.device != pedals.device for x in (lower, upper, valid))
            or lower.dtype != pedals.dtype or upper.dtype != pedals.dtype
            or lower.requires_grad or upper.requires_grad
            or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
            or torch.any(lower < 0) or torch.any(upper > 1)
            or torch.any(lower > upper)):
        raise ValueError('Matched finite detached pedal intervals and explicit cell masks required')
    error = (lower-pedals).relu()+(pedals-upper).relu()
    counts = valid.sum(0)
    per_axis = (error.square()*valid).sum(0)/counts.clamp_min(1)
    available = counts > 0
    return dict(loss=(per_axis*available).sum()/available.sum().clamp_min(1),
        per_axis_loss=per_axis, supervised_cells=counts, available_axes=available,
        steering_loss_applied=False, state_frames_dropped=0, fit_allowed=False)
