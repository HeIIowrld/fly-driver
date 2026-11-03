"""Trainable engineering command boundary into the unchanged flight circuit.

Only four command-current coefficients can learn. No actuator correction or
readout is added. Applied teacher axes are imitation evidence, never optimal
pedal labels. Built-in AI can apply values unreachable through the vJoy codec;
the target conversion records that distinction and masks incompatible labels.
"""

import hashlib
import math

import numpy as np
import torch
from torch import nn

from .flight_maneuver import MANEUVER_CONTRACT, ManeuverCurrents, PhysicalManeuverCircuit

LEARNING_BOUNDARY_CONTRACT = {
    'schema': 'flybrain.flight.learnable_physical_boundary.v1',
    'source_input_contract': MANEUVER_CONTRACT,
    'trainable_scalars': ['left_DNa_gain', 'right_DNa_gain', 'DNg_error_gain', 'DNg_speed_ff'],
    'initialization': 'Exactly original .2 turn/2 power/.05 tonic command currents',
    'turn_gain_bounds': [.05, .8],
    'power_gain_bounds': [.5, 8.],
    'speed_feedforward_logit_bound': .5,
    'unrouted_body_channels': 'load8:12/temperature12:16/wheelrotation17:21',
    'protected_paths': 'Original chemical graph/body haltere/motor readout remain frozen',
    'biological_claim': 'Engineering DN-current calibration, not measured fly plasticity',
}


class TrainableManeuverCurrents(ManeuverCurrents):
    """Bounded positive DN gains and speed feedforward, exactly old at zero init."""

    def __init__(self, orientations, power_ports):
        super().__init__(orientations, power_ports)
        self.turn_delta = nn.Parameter(torch.zeros(2))
        self.power_delta = nn.Parameter(torch.zeros(()))
        self.speed_feedforward_delta = nn.Parameter(torch.zeros(()))

    def coefficients(self):
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError('Finite learnable command coefficients required')
        return dict(turn=.2*torch.exp(math.log(4)*self.turn_delta.tanh()),
            power=2.*torch.exp(math.log(4)*self.power_delta.tanh()),
            speed_feedforward=.5*self.speed_feedforward_delta.tanh())

    def forward(self, command, body, body_valid, racing_goals, *, command_contract,
                goal_contract, lesion=None):
        validation_lesion = None if lesion == 'speed_feedforward' else lesion
        # Retain every original unit/device/mask/ceiling check without changing
        # the source implementation or its protected body feedback path.
        super().forward(command, body, body_valid, racing_goals,
            command_contract=command_contract, goal_contract=goal_contract,
            lesion=validation_lesion)
        coefficients = self.coefficients()
        yaw = torch.tanh(command[:, :1]/MANEUVER_CONTRACT['yaw_scale_rad_s'])
        if lesion == 'desired_yaw':
            yaw = torch.zeros_like(yaw)
        gains = torch.where(self.orientations > 0,
                            coefficients['turn'][0], coefficients['turn'][1])
        steering = gains*(yaw*self.orientations).relu()
        error = (command[:, 1:2]-body[:, 16:17])/MANEUVER_CONTRACT['speed_error_scale_m_s']
        if lesion == 'speed_error':
            error = torch.zeros_like(error)
        feedforward = coefficients['speed_feedforward']*torch.tanh(command[:, 1:2]/10.)
        if lesion == 'speed_feedforward':
            feedforward = torch.zeros_like(feedforward)
        power = (self.tonic_logit+coefficients['power']*torch.tanh(error)+feedforward).sigmoid()
        return steering, power.expand(-1, self.power_ports)


class TrainablePhysicalManeuverCircuit(PhysicalManeuverCircuit):
    """Reuse original drive/advance equations; replace only the DN current boundary."""

    def __init__(self, source_spec, payload, *, sparse_backend='cached_csr_cpu'):
        super().__init__(source_spec, payload, sparse_backend=sparse_backend)
        self.command = TrainableManeuverCurrents(self.backbone.spec['steering_orientations'],
                                                len(self.backbone.goal_ports))
        self.learning_boundary_contract = LEARNING_BOUNDARY_CONTRACT


def canonical_actuator_tensors(intent, *, minimum_positive=.02):
    """Hard codec and explicit latent feasibility distance, without straight-through gradients.

    The encoded value is a capability prediction, not native application. The
    feasibility loss alone can suppress control; it is not a driving objective.
    """
    if (intent.ndim != 2 or intent.shape[1] != 2 or len(intent) < 1
            or not torch.is_floating_point(intent) or not torch.isfinite(intent).all()
            or torch.any(intent.abs() > 1) or not math.isfinite(minimum_positive)
            or not 0 < minimum_positive <= 1):
        raise ValueError('Finite bounded batched neural intent and pedal floor required')
    canonical = torch.stack([-intent[:, 0], intent[:, 1].relu(), (-intent[:, 1]).relu()], 1)
    pedals = canonical[:, 1:]
    gap = (pedals > 0) & (pedals < minimum_positive)
    encoded_pedals = torch.where(gap,
        torch.where(pedals > minimum_positive/2, minimum_positive, 0.), pedals)
    encoded = torch.cat([canonical[:, :1], encoded_pedals], 1)
    # Detached target supplies the exact selected projection's local derivative.
    # Encoded forward remains a hard projection; no fictitious identity Jacobian.
    distance = (pedals-encoded_pedals.detach()).square().sum(1)
    return dict(canonical=canonical, encoded=encoded, pedal_in_gap=gap,
        squared_projection_distance=distance, feasibility_loss=distance.mean())


def applied_axis_supervision(native, original_valid, *, steering_gain,
                             maximum_calibrated_steering=.1, minimum_positive=.02,
                             projection_policy='reject_unreachable', floor_tolerance=1e-4,
                             floor_preimage_margin=1e-4):
    """Set-valued supervision for observed applied axes, with explicit capability limits.

    Built-in AI native subfloor values stay in `native_target`. They are masked
    unless nearest-reachable target projection is explicitly declared. A native
    zero admits canonical[0,.01]; the minimum positive target admits the open
    interval(.01,.02], represented with a declared positive training margin.
    Larger native targets use identity. Simultaneous positive gas/brake is
    outside one signed forward intent and masks only longitudinal supervision.
    """
    if (native.ndim != 2 or native.shape[1] != 3 or len(native) < 1
            or not torch.is_floating_point(native) or native.requires_grad
            or not torch.isfinite(native).all()
            or original_valid.shape != native.shape or original_valid.dtype != torch.bool
            or original_valid.device != native.device or torch.any(native[:, 1:] < 0)
            or torch.any(native[:, 1:] > 1)
            or not all(math.isfinite(v) for v in (steering_gain, maximum_calibrated_steering,
                minimum_positive, floor_tolerance, floor_preimage_margin))
            or steering_gain <= 0 or not 0 < maximum_calibrated_steering <= .1
            or not 0 < floor_tolerance < minimum_positive/2
            or not 0 < floor_preimage_margin < minimum_positive/2
            or not 0 < minimum_positive <= 1
            or projection_policy not in ('reject_unreachable', 'nearest_reachable')):
        raise ValueError('Native axes/masks and reviewed small-domain calibration required')
    pedals = native[:, 1:]
    unreachable = (pedals > 0) & (pedals < minimum_positive)
    reachable_pedals = torch.where(unreachable,
        torch.where(pedals > minimum_positive/2, minimum_positive, 0.), pedals)
    reachable = torch.cat([native[:, :1], reachable_pedals], 1)
    target = reachable if projection_policy == 'nearest_reachable' else native
    projected = target != native
    valid = original_valid.clone()
    steering = native[:, 0]/steering_gain
    outside_steering = steering.abs() > maximum_calibrated_steering
    valid[:, 0] &= ~outside_steering
    if projection_policy == 'reject_unreachable':
        valid[:, 1:] &= ~unreachable
    # Preserve all observations and actor body validity. No signed-drive label
    # is invented by subtracting simultaneous pedals or keeping only one.
    coactivation = (pedals[:, 0] > 0) & (pedals[:, 1] > 0)
    valid[:, 1:] &= ~coactivation[:, None]
    lower = torch.zeros_like(native)
    upper = torch.zeros_like(native)
    lower[:, 0] = upper[:, 0] = torch.where(valid[:, 0], steering, 0.)
    selected = target[:, 1:]
    zero = selected == 0
    floor = (selected-minimum_positive).abs() <= floor_tolerance
    lower[:, 1:] = torch.where(zero, 0., torch.where(floor,
        minimum_positive/2+floor_preimage_margin, selected))
    upper[:, 1:] = torch.where(zero, minimum_positive/2,
                              torch.where(floor, minimum_positive, selected))
    lower = torch.where(valid, lower, 0.)
    upper = torch.where(valid, upper, 0.)
    return dict(native_target=native, reachable_projected_native_target=reachable,
        target_for_supervision=target, projection_applied=projected,
        native_unreachable_pedals=unreachable, native_pedal_coactivation=coactivation,
        steering_outside_calibration=outside_steering, supervision_valid=valid,
        canonical_lower=lower, canonical_upper=upper,
        projection_policy=projection_policy)


def applied_axis_imitation_loss(intent, supervision):
    """Distance to admissible latent requests; labels are observed teacher axes."""
    canonical = canonical_actuator_tensors(intent)['canonical']
    lower, upper, valid = (supervision[k] for k in (
        'canonical_lower', 'canonical_upper', 'supervision_valid'))
    if (lower.shape != canonical.shape or upper.shape != canonical.shape
            or valid.shape != canonical.shape or valid.dtype != torch.bool
            or lower.device != canonical.device or upper.device != canonical.device
            or valid.device != canonical.device or lower.dtype != canonical.dtype
            or upper.dtype != canonical.dtype or lower.requires_grad or upper.requires_grad
            or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
            or torch.any(lower > upper)):
        raise ValueError('Matched finite set-valued supervision and explicit validity required')
    error = (lower-canonical).relu()+(canonical-upper).relu()
    counts = valid.sum(0)
    per_axis = (error.square()*valid).sum(0)/counts.clamp_min(1)
    available = counts > 0
    return (per_axis*available).sum()/available.sum().clamp_min(1)


def _fingerprint(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()
                          ).hexdigest()


def validate_motor_training_episode(metadata, source_record, arrays):
    """Validate a full episode against its verified immutable source-manifest record.

    The loader must first verify the source manifest SHA; this function never
    upgrades a source role, invents desired yaw, or treats student actions as an
    optimal teacher. Same native packet controls/body are imitation targets;
    request-to-application delays are not silently realigned.
    """
    if (metadata.get('schema') != 'flybrain.flight.motor_training_episode.v1'
            or metadata.get('id') != source_record.get('id')
            or metadata.get('role') != 'train' or source_record.get('role') != 'train'
            or metadata.get('supervision_kind') != 'observed_teacher_applied_axis_imitation'
            or metadata.get('command_source') not in (
                'declared_external_task_and_teacher_yaw_reference',
                'vision_actor_prediction_and_declared_external_task')
            or metadata.get('native_axis_calibration_verified') is not True
            or metadata.get('native_target_is_optimal_control') is not False):
        raise ValueError('Original TRAIN, actual teacher supervision and declared units required')
    needed = ('decision_clock', 'native_receipt_clock', 'native_packet',
              'control_native_receipt_clock', 'control_native_packet', 'body', 'body_valid',
              'physical_command', 'racing_goals', 'native_control', 'native_control_valid')
    if any(k not in arrays for k in needed):
        raise ValueError('Original full clocks/body/commands/native axes and masks required')
    a = {key: np.asarray(arrays[key]) for key in needed}
    clock = a['decision_clock']
    if clock.ndim != 1:
        raise ValueError('One-dimensional original decision clocks required')
    n = len(clock)
    if (n != source_record.get('observations') or n < 2 or clock.shape != (n,)
            or not np.isfinite(clock).all() or not np.all((np.diff(clock) > 0)
                                                       & (np.diff(clock) <= .08))
            or a['body'].shape != (n, 21) or a['body_valid'].shape != (n, 21)
            or a['body_valid'].dtype != bool or a['physical_command'].shape != (n, 2)
            or a['racing_goals'].shape != (n, 2) or a['native_control'].shape != (n, 3)
            or a['native_control_valid'].shape != (n, 3)
            or a['native_control_valid'].dtype != bool
            or not all(np.isfinite(a[k]).all() for k in needed)
            or a['native_receipt_clock'].shape != (n,)
            or a['control_native_receipt_clock'].shape != (n,)
            or a['native_packet'].shape != (n,) or a['control_native_packet'].shape != (n,)
            or a['native_packet'].dtype.kind not in 'iu'
            or a['control_native_packet'].dtype.kind not in 'iu'
            or np.any(a['native_packet'] < 0)
            or not np.array_equal(a['native_packet'], a['control_native_packet'])
            or not np.array_equal(a['native_receipt_clock'], a['control_native_receipt_clock'])
            or not np.all(np.diff(a['native_packet']) > 0)
            or not np.all((clock-a['native_receipt_clock'] >= 0)
                          & (clock-a['native_receipt_clock'] <= .05))
            or _fingerprint(clock) != source_record.get('original_clock_fingerprint')
            or _fingerprint(a['body_valid']) != source_record.get(
                'original_body_mask_fingerprint')):
        raise ValueError('Whole original clocks/masks and same-packet controls required')
    if (not a['body_valid'][:, [4, 16]].all() or np.any(a['body'][:, 16] < 0)
            or np.any(abs(a['physical_command'][:, 0]) > 2)
            or np.any(a['physical_command'][:, 1] < 0)
            or np.any(a['racing_goals'][:, 0] != 0)
            or np.any((a['racing_goals'][:, 1] <= 0) | (a['racing_goals'][:, 1] > 3.5))
            or np.any(a['physical_command'][:, 1] > a['racing_goals'][:, 1]*100/3.6+1e-5)
            or np.any((a['native_control'][:, 1:] < 0) | (a['native_control'][:, 1:] > 1))):
        raise ValueError('Original valid yaw/speed, physical-unit commands and native pedal range')
    return dict(observations=n, intervals=n-1, external_role='train',
        supervised_control_cells=int(a['native_control_valid'].sum()),
        state_frames_dropped=0, actor_body_masks_modified=False,
        native_targets_are_observations_not_optimal_labels=True)
