"""Explicit racing intent, incompatible with the old <=30km/h speed-target task."""

import numpy as np

RACING_GOAL_CONTRACT = {
    'schema': 'flybrain.flight.racing_goal.v1',
    'fields': ['visual_road_following_turn_bias', 'external_speed_ceiling_fraction'],
    'speed_scale_kmh': 100.,
    'maximum_supported_ceiling_kmh': 350.,
    'objective': 'Minimize valid lap time; accelerate/brake from vision and body feedback',
    'speed_semantics': 'Externally declared ceiling, not a constant desired cruising speed',
    'teacher_curve_profile_actor_input': False,
    'checkpoint_compatibility': 'New specification/training required; reject speed-goal v2',
    'biological_claim': 'Engineering task boundary, not measured fly tuning',
}


def racing_goal(speed_ceiling_kmh, count):
    """External ceiling only; curve-dependent teacher speeds stay targets/critic truth."""
    if (isinstance(speed_ceiling_kmh, bool) or not np.isscalar(speed_ceiling_kmh)
            or not np.isfinite(speed_ceiling_kmh) or not 0 < speed_ceiling_kmh <= 350
            or type(count) is not int or count < 1):
        raise ValueError('Explicit positive external ceiling <=350km/h and count required')
    value = np.zeros((count, 2), np.float32)
    value[:, 1] = speed_ceiling_kmh / 100.
    return value


def validate_racing_goal_contract(contract):
    if contract != RACING_GOAL_CONTRACT:
        raise ValueError('Explicit racing ceiling semantics required; no legacy reinterpretation')
