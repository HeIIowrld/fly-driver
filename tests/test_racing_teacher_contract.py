import pytest

from flybrain_drive.racing_teacher_contract import racing_teacher_contract


def request():
    return dict(schema='flybrain.csp.session.request.v1', command='expert-collect30-speed60',
                speed_limit_kmh=60, hold_seconds=30, ai_steer_multiplier=.9,
                generation='fixture', id='fixture-request')


def test_race_envelope_is_separate_from_teacher_hint_and_never_implies_qualification():
    contract = racing_teacher_contract(request(), declared_actor_ceiling_kmh=80, role='train')
    assert contract['actor_goal'] == [0., .8]
    assert contract['teacher_nominal_speed_hint_kmh'] == 60
    assert contract['fitting_role_allowed'] is True
    assert contract['teacher_hint_is_actor_input'] is False
    assert contract['training_admissible'] is False
    assert contract['legacy_speed_goal_checkpoint_compatible'] is False


@pytest.mark.parametrize('role', ['development', 'evaluation', 'calibration'])
def test_other_preassigned_roles_cannot_become_fit_data(role):
    contract = racing_teacher_contract(request(), declared_actor_ceiling_kmh=80, role=role)
    assert contract['source_role'] == role and not contract['fitting_role_allowed']


@pytest.mark.parametrize('change', [dict(command='expert-collect30-speed30'),
    dict(speed_limit_kmh=30), dict(hold_seconds=3), dict(brake_pulse_seconds=.4),
    dict(recovery_target_kmh=15)])
def test_old_and_intervened_tasks_cannot_be_reinterpreted(change):
    with pytest.raises(ValueError):
        racing_teacher_contract(dict(request(), **change), declared_actor_ceiling_kmh=80,
                                role='train')


def test_ceiling_cannot_be_derived_from_the_teacher_hint_after_collection():
    with pytest.raises(ValueError):
        racing_teacher_contract(request(), declared_actor_ceiling_kmh=60, role='train')
