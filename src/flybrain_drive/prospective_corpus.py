"""CPU-only prospective corpus role/profile contracts; no simulator interface.

These validators do not enable a profile or certify its implementation. Original
roles and whole histories are never changed after capture. Geography/teacher
conditions are privileged supervision and evaluation metadata, not actor inputs.
"""

import math

ROLES = ('train', 'independent_validation_geography', 'independent_validation_condition')
PACE30_SCHEMA = 'flybrain.pace60_100_stop30_profile.v1'
PACE30_SCHEDULE = [dict(after_seconds=0, speed_hint_kmh=60),
                   dict(after_seconds=5, speed_hint_kmh=100),
                   dict(after_seconds=15, speed_hint_kmh=0),
                   dict(after_seconds=23, speed_hint_kmh=60)]


def overlaps(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def interval(value):
    if (len(value) != 2 or not all(math.isfinite(x) for x in value)
            or not 0 <= value[0] < value[1] <= 1):
        raise ValueError('Finite ordered nonwrapping spline interval required')
    return value


def validate_split(plan):
    if (plan['schema'] != 'flybrain.prospective_diverse_corpus.v1'
            or plan['historical_roles_changed'] is not False
            or plan['actor_world_or_profile_oracle_input'] is not False):
        raise ValueError('Explicit new prospective corpus and unchanged information roles required')
    protected = [interval(x) for x in plan['known_development_regions']+plan['evaluation_regions']]
    historical = [interval(x) for x in plan['historical_training_observed_envelopes']]
    regions = plan['regions']
    if {r['role'] for r in regions} != set(ROLES):
        raise ValueError('TRAIN plus separate geographic/condition validation are required')
    train = [r for r in regions if r['role'] == 'train']
    for region in regions:
        bounds = interval(region['spline_interval'])
        if region['role'] == 'train' and any(overlaps(bounds, x) for x in protected):
            raise ValueError('Known DEVELOPMENT/EVALUATION geography cannot become TRAIN')
        if region['role'] == 'independent_validation_geography':
            if any(overlaps(bounds, x) for x in historical+protected):
                raise ValueError('New geography must exclude recordedTRAIN and protected regions')
            if any(overlaps(bounds, interval(r['spline_interval'])) for r in train):
                raise ValueError('Geographic validation cannot overlap new TRAIN')
            if region['claim'] != 'unseen_geography_straight_or_gentle_bend_only':
                raise ValueError('Available new geography does not certify both-direction corners')
        if region['role'] == 'independent_validation_condition':
            if region['geometry_seen'] is not True or region['profile_seen_in_fit'] is not False:
                raise ValueError('Condition holdout must disclose seen geometry/unseen profile')
    profiles = plan['profiles']
    fit_profiles = {name for r in train for name in r['profiles']}
    condition_profiles = {name for r in regions
                          if r['role'] == 'independent_validation_condition'
                          for name in r['profiles']}
    if fit_profiles & condition_profiles:
        raise ValueError('Held-out condition profile cannot also fit TRAIN')
    for region in regions:
        if (not 1 <= region['maximum_histories'] <= 6 or not region['profiles']
                or any(name not in profiles for name in region['profiles'])):
            raise ValueError('Finite declared history count and known profiles required')
    for profile in profiles.values():
        if profile.get('schema') == PACE30_SCHEMA:
            validate_pace30_profile(profile)
            continue
        duration = profile['duration_seconds']
        stages = profile['speed_schedule']
        if (duration not in (8, 12) or profile['hard_speed_kmh'] != 120
                or profile['hard_displacement_m'] != 500
                or not stages or stages[0]['after_seconds'] != 0
                or any(not 0 <= s['after_seconds'] < duration
                       or s['speed_hint_kmh'] not in (40, 60, 80, 100) for s in stages)
                or any(a['after_seconds'] >= b['after_seconds']
                       for a, b in zip(stages[:-1], stages[1:], strict=True))):
            raise ValueError('Separate finite speed100-capable profile contract required')
        pulse = profile['brake_intervention']
        if (pulse['api'] != 'physics.setAIStopCounter' or pulse['requested_seconds'] != .4
                or pulse['attempts_maximum'] != 1
                or not 0 <= pulse['after_seconds'] < duration
                or not 20 < pulse['speed_window_kmh'][0] < pulse['speed_window_kmh'][1] <= 110
                or profile['natural_ai_braking_label'] is not False):
            raise ValueError('Explicit once-only labelled brake intervention contract required')
    return dict(status='passed_cpu_prospective_split_contract', control_permission=False,
                collector_implementation_qualified=False, historical_roles_changed=False)


def validate_pace30_profile(profile):
    """Separate30-second contract; never reinterpret existing8/12/old60 profiles."""
    expected = dict(schema=PACE30_SCHEMA, command='expert-pace60-100-stop30',
        duration_seconds=30, hard_speed_kmh=120, hard_displacement_m=1100,
        speed_schedule=PACE30_SCHEDULE, natural_ai_braking_label=False,
        brake_intervention=dict(api='physics.setAIStopCounter', requested_seconds=8,
            after_seconds=15, speed_window_kmh=[40, 110], attempts_maximum=1,
            clear_after_seconds=23, clear_requires_actual_moving_brake_and_stop=True))
    if profile != expected:
        raise ValueError('Exact separate60/100/stop/60 pace30 contract required')
    return dict(status='passed_cpu_pace30_profile_contract', execution_qualified=False)


def next_pace30_train_lease(spline):
    """A planning estimate, never a spline/physical advance guarantee.

    Every original before/driving/stopping/after position still needs whole
    history admission. An ineligible start becomes transit, never a sliced fit.
    """
    if not math.isfinite(spline) or not 0 <= spline < 1:
        raise ValueError('Finite diagnostic current spline required')
    if .275 <= spline <= .44:
        return dict(role='train', profile='expert-pace60-100-stop30', duration_seconds=30,
            spline_interval=[.275, .54], expected_start_spline=spline,
            planning_advance_estimate=.10, estimate_is_not_physical_arc_length_bound=True,
            preferred_buffered_start=.285 <= spline <= .425, fit_allowed=False)
    approach = (.275-spline) % 1
    return dict(role='transit', duration_seconds=3 if approach <= .13 else 30,
                expected_start_spline=spline, fit_allowed=False)


def admit_whole_history(assignment, route_samples):
    if assignment['role'] not in (*ROLES, 'development') or not route_samples:
        raise ValueError('Previously assigned full history and original diagnostic route required')
    lo, hi = interval(assignment['spline_interval'])
    if (abs(route_samples[0]['spline_position']-assignment['expected_start_spline']) > 1e-4
            or any(r['actor_input'] is not False for r in route_samples)
            or any(not lo <= r['spline_position'] <= hi for r in route_samples)
            or any(r['wheels_outside'] != 0 for r in route_samples)
            or len({r['lap_count'] for r in route_samples}) != 1
            or any(a['spline_position']-b['spline_position'] > .001
                   for a, b in zip(route_samples[:-1], route_samples[1:], strict=True))):
        raise ValueError('Whole cleanup route violates prospective role; never slice/relabel')
    return dict(role=assignment['role'], fit_allowed=False, entire_history_retained=True,
                original_samples=len(route_samples), raw_and_label_audits_still_required=True)


def native_condition_counts(samples):
    """One whole history; count original adjacent driving gear samples only.

    Raw continuity/clock validation is a separate prerequisite. Filtering out a
    stopping phase must not create a fictitious driving gear transition.
    """
    driving = [s for s in samples if s['phase'] == 'driving']
    if any(not all(math.isfinite(s[k]) for k in ('speed_kmh', 'brake', 'gear'))
           for s in driving):
        raise ValueError('Finite decoded native control/gear observations required')
    return dict(moving_brake_rows=sum(s['speed_kmh'] > 20 and s['brake'] >= .02 for s in driving),
        speed_at_least70_rows=sum(s['speed_kmh'] >= 70 for s in driving),
        speed_at_least90_rows=sum(s['speed_kmh'] >= 90 for s in driving),
        moving_gear_transitions=sum(a['phase'] == b['phase'] == 'driving'
            and a['speed_kmh'] > 20 and b['speed_kmh'] > 20
            and min(a['gear'], b['gear']) >= 2 and a['gear'] != b['gear']
            for a, b in zip(samples[:-1], samples[1:], strict=True)))
