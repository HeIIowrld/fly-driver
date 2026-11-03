"""Versioned prospective racing demonstration task, not a legacy speed-goal label."""

from copy import deepcopy

from .racing_goal import RACING_GOAL_CONTRACT


def racing_teacher_contract(request, *, declared_actor_ceiling_kmh, role):
    """Separate the external race envelope from the engine AI's nominal speed hint.

    This validates task metadata only. Raw replay, native-control codec, time
    matching, source role and actual speed/track qualification remain separate.
    """
    if role not in ('train', 'development', 'evaluation', 'calibration'):
        raise ValueError('Preassigned source role required')
    if (request.get('schema') != 'flybrain.csp.session.request.v1'
            or request.get('command') != 'expert-collect30-speed60'
            or request.get('speed_limit_kmh') != 60
            or request.get('hold_seconds') != 30
            or request.get('ai_steer_multiplier') != .9
            or isinstance(declared_actor_ceiling_kmh, bool)
            or declared_actor_ceiling_kmh != 80):
        raise ValueError('Separate60kmh demonstration and predeclared80kmh race envelope required')
    if any(key in request for key in ('recovery_target_kmh', 'brake_pulse_seconds',
                                      'student_owner_pid', 'lateral_recovery')):
        raise ValueError('Interventions/handoffs require their own task contract')
    if not request.get('generation') or not request.get('id'):
        raise ValueError('Original producer/request identity required')
    return dict(
        schema='flybrain.engine_ai.racing_receipt_contract.v1',
        source_request_id=request['id'], source_generation=request['generation'], source_role=role,
        fitting_role_allowed=role == 'train',
        actor_goal_contract=deepcopy(RACING_GOAL_CONTRACT),
        actor_external_speed_ceiling_kmh=80., actor_goal=[0., .8],
        teacher_nominal_speed_hint_kmh=60., teacher_hint_is_actor_input=False,
        teacher_is_fastest_or_optimal=False,
        actor_steering_device_scale=.6, ai_steer_multiplier=.9,
        target_receipt_horizon_seconds=.04, maximum_target_lateness_seconds=.025,
        clock='host_monotonic_QPC_time.perf_counter',
        target_kind='future applied native controls, not internal AI requests',
        raw_control_codec_verified=False, native_receipt_matching_performed=False,
        physical_command_time_verified=False, speed_limit_enforcement_verified=False,
        training_admissible=False, actor_qualified=False,
        legacy_speed_goal_checkpoint_compatible=False,
        privileged_route_or_curve_speed_in_actor_goal=False)
