import pytest

from scripts.probe_ac_maneuver_plant import single_gear_transition, validate_response_envelope


def test_low_speed_displacement_and_damage_are_hard_abort_boundaries():
    validate_response_envelope(5., 2., 0., 0, 2500)
    for row in ((12, 2, 0, 0, 2500), (5, 8, 0, 0, 2500), (5, 2, .02, 0, 2500),
                (5, 2, 0, 1, 2500), (5, 2, 0, 0, 6600), (float('nan'), 2, 0, 0, 2500)):
        with pytest.raises(ValueError):
            validate_response_envelope(*row)


def test_single_transition_direction_depends_on_native_required_gear():
    assert single_gear_transition(2, 1) == ('down', 1)
    assert single_gear_transition(1, 2) == ('up', 0)
    assert single_gear_transition(1, 1) is None
    assert single_gear_transition(2, 2) is None
    for current, required in ((0, 1), (3, 1), (1, 3)):
        with pytest.raises(ValueError):
            single_gear_transition(current, required)
