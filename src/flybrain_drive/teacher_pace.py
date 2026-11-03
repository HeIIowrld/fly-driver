"""Explicit engineering speed/stop task, separate from natural AI driving labels."""

import math

PACE_COMMAND = 'expert-pace60-100-stop30'
PACE_CONTRACT = 'flybrain.external_speed_task.v1'
GOAL_SERIALIZATION_ABSOLUTE_TOLERANCE_M_S = 5e-10
PACE_REQUEST = dict(pace_contract=PACE_CONTRACT, pace_boost_after_seconds=5,
                    pace_stop_after_seconds=15, pace_restart_after_seconds=23,
                    pace_stop_seconds=8)


def goal_matches(observed, expected):
    """CSP JSON rounds division results; validate wire precision without replacing it."""
    return (type(observed) in (int, float) and math.isfinite(observed)
            and abs(observed-expected) <= GOAL_SERIALIZATION_ABSOLUTE_TOLERANCE_M_S)


def validate_pace_receipt(receipt):
    if (receipt.get('explicit_pace_task') is not True
            or receipt.get('pace_task_passed') is not True
            or receipt.get('pace_contract') != PACE_CONTRACT
            or receipt.get('motion_hard_speed_kmh') != 120
            or receipt.get('maximum_displacement_guard_m') != 1100
            or not 80 <= receipt.get('maximum_speed_kmh', -1) <= 120
            or receipt.get('pace_brake_peak', 0) <= .05
            or receipt.get('pace_moving_brake_seen') is not True
            or not 40 <= receipt.get('pace_before_stop_speed_kmh', -1) <= 110
            or not 0 <= receipt.get('pace_min_speed_kmh', -1) < 1):
        raise ValueError('Observed explicit60/100/stop/60 task response required')
    transitions = receipt.get('pace_transitions', [])
    if (len(transitions) != 4
            or [r['phase'] for r in transitions] != [
                'accelerate60', 'accelerate100', 'stop0', 'accelerate60']
            or not all(goal_matches(r['desired_speed_m_s'], target) for r, target in zip(
                transitions, (60/3.6, 100/3.6, 0, 60/3.6), strict=True))):
        raise ValueError('Complete ordered external task transitions required')
    first = transitions[0]['script_clock']
    for expected, row in zip((0, 5, 15, 23), transitions, strict=True):
        elapsed = row['script_clock']-first
        if not math.isfinite(elapsed) or not expected <= elapsed <= expected+.15:
            raise ValueError('Original observed transition clock outside declared phase window')
    moving = [s for s in receipt.get('telemetry', []) if s.get('phase') == 'driving'
              and s.get('external_task_phase') == 'stop0'
              and s.get('speed_kmh', 0) > 20 and s.get('brake', 0) > .05]
    if not moving:
        raise ValueError('Moving brake must be observed before final external cleanup')
    return dict(external_speed_task=True, natural_ai_labels=False,
                moving_brake_samples=len(moving), original_transition_clocks_retained=True)
