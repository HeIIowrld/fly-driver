import pytest

from scripts.transit_to_pace_region import child_command, transit_decision


def test_ready_anywhere_eligible_never_activates_and_preserves_transit_role():
    for spline in (.275, .285, .3, .425, .44):
        decision = transit_decision(spline)
        assert decision['action'] == 'stop' and decision['duration_seconds'] is None
        assert decision['role'] == 'transit' and decision['fit_allowed'] is False
    assert transit_decision(.3)['preferred_start']
    assert not transit_decision(.275)['preferred_start']


def test_duration_is_only_existing30_or3_profile_with_explicit_planning_distance():
    assert transit_decision(.94618)['duration_seconds'] == 30
    assert transit_decision(.20)['duration_seconds'] == 30
    assert transit_decision(.22)['duration_seconds'] == 3
    assert transit_decision(.274)['duration_seconds'] == 3
    decision = transit_decision(.20)
    assert decision['estimate_is_not_physical_arc_bound'] is True
    assert decision['actor_input'] is False


def test_initial_wrap_is_allowed_but_approach_overshoot_never_starts_another_lap():
    wrapped = transit_decision(.03, previous=.94618)
    assert wrapped['wraps'] == 1 and wrapped['approaching'] is True
    assert transit_decision(.28, approaching=True, wraps=1, previous=.20)['action'] == 'stop'
    with pytest.raises(ValueError, match='passed above'):
        transit_decision(.45, approaching=True, previous=.20)
    with pytest.raises(ValueError, match='another lap'):
        transit_decision(.02, approaching=True, wraps=1, previous=.99)
    for invalid in (float('nan'), -1., 1.):
        with pytest.raises(ValueError):
            transit_decision(invalid)


def test_child_uses_only_old60_profile_and_never_pace_or_shadow():
    for duration in (3, 30):
        command = child_command(dict(camera_configuration='declared.json'), 'output', duration)
        assert command[command.index('--speed-hint-kmh')+1] == '60'
        assert command[command.index('--duration-seconds')+1] == str(duration)
        assert '--pace-task' not in command and '--flight-shadow-config' not in command
    with pytest.raises(ValueError):
        child_command(dict(camera_configuration='declared.json'), 'output', 12)


def test_development_target_stops_only_in_registered_interval():
    target = dict(eligible=(.67, .72), preferred=(.68, .71))
    assert transit_decision(.35, **target)['duration_seconds'] == 30
    assert transit_decision(.66, **target)['duration_seconds'] == 3
    assert transit_decision(.69, **target)['action'] == 'stop'
    with pytest.raises(ValueError, match='passed'):
        transit_decision(.73, approaching=True, **target)
