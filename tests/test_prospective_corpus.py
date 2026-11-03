import copy
import json
from pathlib import Path

import pytest

from flybrain_drive.prospective_corpus import (
    PACE30_SCHEDULE,
    PACE30_SCHEMA,
    admit_whole_history,
    native_condition_counts,
    next_pace30_train_lease,
    validate_pace30_profile,
    validate_split,
)


def test_known_development_region_preserves_whole_history_role():
    assignment = dict(role='development', spline_interval=[.67, .84],
                      expected_start_spline=.68)
    rows = [dict(spline_position=x, actor_input=False, wheels_outside=0, lap_count=0)
            for x in (.68, .74, .81)]
    assert admit_whole_history(assignment, rows)['role'] == 'development'
    rows[-1]['spline_position'] = .85
    with pytest.raises(ValueError):
        admit_whole_history(assignment, rows)


def plan():
    return json.loads(Path('configs/flight-new-diverse-corpus-20261030.json').read_text())


def test_split_contract_preserves_old_roles_and_distinguishes_validation_claims():
    assert validate_split(plan())['control_permission'] is False
    for bounds in ([.2, .25], [.08, .125], [.67, .84], [.94, .99]):
        changed = plan()
        changed['regions'][0]['spline_interval'] = bounds
        with pytest.raises(ValueError, match='cannot become TRAIN'):
            validate_split(changed)
    changed = plan()
    region = next(r for r in changed['regions']
                  if r['role'] == 'independent_validation_geography')
    region['spline_interval'] = [.4, .5]
    with pytest.raises(ValueError, match='recordedTRAIN'):
        validate_split(changed)
    changed = plan()
    region = next(r for r in changed['regions']
                  if r['role'] == 'independent_validation_condition')
    region['profiles'] = [changed['regions'][0]['profiles'][0]]
    with pytest.raises(ValueError, match='cannot also fit'):
        validate_split(changed)


def test_full_crossing_cleanup_cannot_be_sliced_or_relabelled():
    assignment = dict(role='train', spline_interval=[.275, .54], expected_start_spline=.3)
    rows = [dict(spline_position=p, actor_input=False, wheels_outside=0, lap_count=0)
            for p in (.3, .4, .53)]
    assert admit_whole_history(assignment, rows)['fit_allowed'] is False
    for key, value in [('spline_position', .55), ('wheels_outside', 1), ('lap_count', 1),
                       ('actor_input', True)]:
        changed = copy.deepcopy(rows)
        changed[-1][key] = value
        with pytest.raises(ValueError, match='never slice'):
            admit_whole_history(assignment, changed)


def test_actual_native_conditions_do_not_count_speed_hints_or_stationary_cleanup():
    samples = [dict(phase='driving', speed_kmh=75, brake=.025, gear=2),
               dict(phase='driving', speed_kmh=95, brake=0, gear=3),
               dict(phase='stopping', speed_kmh=15, brake=1, gear=2)]
    assert native_condition_counts(samples) == dict(moving_brake_rows=1,
        speed_at_least70_rows=2, speed_at_least90_rows=1, moving_gear_transitions=1)
    samples.append(dict(phase='driving', speed_kmh=95, brake=0, gear=4))
    assert native_condition_counts(samples)['moving_gear_transitions'] == 1


def test_new_pace30_contract_cannot_relax_earlier_profiles_or_assign_a_crossing_start():
    p = dict(schema=PACE30_SCHEMA, command='expert-pace60-100-stop30', duration_seconds=30,
        hard_speed_kmh=120, hard_displacement_m=1100, speed_schedule=PACE30_SCHEDULE,
        natural_ai_braking_label=False,
        brake_intervention=dict(api='physics.setAIStopCounter', requested_seconds=8,
            after_seconds=15, speed_window_kmh=[40, 110], attempts_maximum=1,
            clear_after_seconds=23, clear_requires_actual_moving_brake_and_stop=True))
    assert validate_pace30_profile(p)['execution_qualified'] is False
    changed = copy.deepcopy(p)
    changed['brake_intervention']['clear_requires_actual_moving_brake_and_stop'] = False
    with pytest.raises(ValueError):
        validate_pace30_profile(changed)
    assert next_pace30_train_lease(.3)['role'] == 'train'
    assert next_pace30_train_lease(.44)['role'] == 'train'
    assert next_pace30_train_lease(.441)['role'] == 'transit'
    assert next_pace30_train_lease(.55)['role'] == 'transit'
    with pytest.raises(ValueError):
        admit_whole_history(next_pace30_train_lease(.3), [
            dict(spline_position=.3, actor_input=False, wheels_outside=0, lap_count=0),
            dict(spline_position=.55, actor_input=False, wheels_outside=0, lap_count=0)])
