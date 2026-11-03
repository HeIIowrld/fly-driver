import pytest

from scripts.control_ac_session import target_reached


def test_lifecycle_metadata_gap_waits_for_new_bytes_and_same_handle_timestamp(monkeypatch):
    from scripts import control_ac_session as client

    clock = [0.]
    calls = []

    def read_new(path):
        calls.append(path)
        if clock[0] < .25:
            raise FileNotFoundError('publisher replacement failed until next heartbeat')
        return {'generation': 'new'}, 123.

    monkeypatch.setattr(client, 'read_published_json', read_new)
    monkeypatch.setattr(client.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(client.time, 'sleep', lambda delta: clock.__setitem__(0, clock[0]+delta))
    value, modified, retries = client.read_session_heartbeat('status.json')
    assert value == {'generation': 'new'} and modified == 123.
    assert retries == len(calls)-1 and retries > 1
    assert .25 <= clock[0] <= .5


def test_lifecycle_metadata_missing_or_malformed_never_becomes_success(monkeypatch):
    from scripts import control_ac_session as client

    clock = [0.]
    monkeypatch.setattr(client.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(client.time, 'sleep', lambda delta: clock.__setitem__(0, clock[0]+delta))

    def missing(path):
        raise FileNotFoundError('absent')

    monkeypatch.setattr(client, 'read_published_json', missing)
    with pytest.raises(FileNotFoundError):
        client.read_session_heartbeat('status.json')
    assert .5 <= clock[0] < .51

    def malformed(path):
        raise ValueError('bad bytes')

    clock[0] = 0.
    monkeypatch.setattr(client, 'read_published_json', malformed)
    with pytest.raises(ValueError, match='bad bytes'):
        client.read_session_heartbeat('status.json')
    assert clock[0] == 0.


def test_disable_line_requires_observed_off_and_idle_session():
    before = dict(generation='one', paused=True)
    after = dict(generation='one', ideal_line_shown=False, paused=True,
                 ai_controlled=False, camera_active=False)
    assert target_reached('disable-ideal-line', before, after)
    for key, value in [('ideal_line_shown', True), ('ideal_line_shown', None),
                       ('paused', False), ('ai_controlled', True), ('camera_active', True)]:
        assert not target_reached('disable-ideal-line', before, {**after, key: value})
    assert target_reached('disable-ideal-line', {**before, 'paused': False},
                          {**after, 'paused': False})


def test_restart_requires_observed_episode_boundary_and_menu():
    before = dict(generation='engine-one', epoch=3)
    assert not target_reached('restart', before,
                              dict(generation='engine-one', epoch=3, in_menu=True))
    assert not target_reached('restart', before,
                              dict(generation='engine-one', epoch=4, in_menu=False))
    assert target_reached('restart', before,
                          dict(generation='engine-one', epoch=4, in_menu=True))
    with pytest.raises(ValueError, match='generation changed'):
        target_reached('restart', before, dict(generation='engine-two', epoch=4, in_menu=True))


def test_start_needs_live_cockpit_and_pause_is_explicit():
    before = dict(generation='engine-one', epoch=3)
    assert not target_reached('start', before,
                              dict(generation='engine-one', in_menu=True, live=True))
    assert not target_reached('start', before,
                              dict(generation='engine-one', in_menu=False, live=False))
    assert target_reached('start', before,
                          dict(generation='engine-one', in_menu=False, live=True))
    assert target_reached('pause', before, dict(generation='engine-one', paused=True))
    assert target_reached('resume', before, dict(generation='engine-one', paused=False))


def test_read_only_error_status_is_inspectable_but_mutations_remain_blocked():
    from scripts.control_ac_session import validate_heartbeat
    error = {"error": "Cannot commit camera metadata", "paused": True}
    validate_heartbeat(error, .1, mutating=False)
    with pytest.raises(ValueError, match="Error-free"):
        validate_heartbeat(error, .1, mutating=True)
    for age in (3., -1., float("nan")):
        with pytest.raises(ValueError, match="Fresh"):
            validate_heartbeat(error, age, mutating=False)
    validate_heartbeat({}, .1, mutating=True)


def test_immutable_terminal_and_legacy_receipts_pin_owned_request(tmp_path):
    import json

    from scripts.control_ac_session import terminal_receipt

    request = dict(id='a'*32, command='pause')
    accepted = tmp_path / (request['id']+'.json')
    terminal = tmp_path / (request['id']+'.terminal.json')
    accepted.write_text(json.dumps(dict(request, status='accepted')), encoding='utf8')
    assert terminal_receipt(tmp_path, request) is None
    final = dict(request, status='api_returned_true')
    terminal.write_text(json.dumps(final), encoding='utf8')
    assert terminal_receipt(tmp_path, request) == final
    assert json.loads(accepted.read_text())['status'] == 'accepted'
    terminal.unlink()
    accepted.write_text(json.dumps(final), encoding='utf8')
    assert terminal_receipt(tmp_path, request) == final
    accepted.write_text(json.dumps(dict(final, id='b'*32)), encoding='utf8')
    with pytest.raises(ValueError, match='ID differs'):
        terminal_receipt(tmp_path, request)


def test_probe_requires_explicit_paused_ai_off_capabilities_and_verified_teardown():
    from scripts.control_ac_session import validate_expert_probe, validate_heartbeat

    ready = dict(paused=True, ai_controlled=False, physics_api_allowed=True,
                 capabilities=dict(expert_probe=True), expert_probe_active=False)
    validate_expert_probe(ready)
    for key in ('paused', 'ai_controlled', 'physics_api_allowed', 'capabilities'):
        changed = {k: v for k, v in ready.items() if k != key}
        with pytest.raises(ValueError):
            validate_expert_probe(changed)
    for key in ('expert_probe_active', 'ai_controlled'):
        with pytest.raises(ValueError, match='inactive'):
            validate_heartbeat(dict(ready, **{key: True}), .1, mutating=True)
    before = dict(generation='one')
    after = dict(ready, generation='one')
    assert target_reached('expert-probe', before, after)
    assert not target_reached('expert-probe', before, dict(after, ai_controlled=None))
    assert not target_reached('expert-probe', before, dict(after, expert_probe_active=True))


@pytest.mark.parametrize('delta', [None, float('nan'), -.01, .03])
def test_probe_receipt_cannot_claim_success_without_stationarity(delta):
    from scripts.control_ac_session import validate_probe_receipt

    with pytest.raises(ValueError):
        validate_probe_receipt(dict(stationary_probe_passed=True, probe_seen_enabled=True,
                                    maximum_position_delta_m=delta))


def test_unknown_probe_receipt_must_pin_command(tmp_path):
    import json

    from scripts.control_ac_session import terminal_receipt

    request = dict(id='a'*32, command='expert-probe')
    (tmp_path/(request['id']+'.terminal.json')).write_text(json.dumps(
        dict(request, status='api_outcome_unknown', command='pause')), encoding='utf8')
    with pytest.raises(ValueError, match='command differs'):
        terminal_receipt(tmp_path, request)


def test_motion_receipt_requires_movement_and_executed_ai_throttle():
    from scripts.control_ac_session import validate_probe_receipt

    receipt = dict(motion_probe_passed=True, probe_seen_enabled=True,
                   maximum_position_delta_m=2, maximum_speed_kmh=10,
                   telemetry=[dict(ai_controlled=True, paused=False, gas=.2),
                              dict(ai_controlled=False, paused=True, gas=0)])
    validate_probe_receipt(receipt, motion=True)
    for key, value in [('maximum_speed_kmh', 21), ('maximum_position_delta_m', 0),
                       ('telemetry', []), ('motion_probe_passed', False)]:
        with pytest.raises(ValueError):
            validate_probe_receipt(dict(receipt, **{key: value}), motion=True)


def test_fast_receipt_cannot_relax_existing_profile_by_default():
    from scripts.control_ac_session import validate_probe_receipt

    receipt = dict(motion_probe_passed=True, probe_seen_enabled=True,
                   maximum_position_delta_m=25, maximum_speed_kmh=33,
                   telemetry=[dict(ai_controlled=True, paused=False, gas=.2)]*2)
    validate_probe_receipt(receipt, motion=True, fast=True)
    with pytest.raises(ValueError):
        validate_probe_receipt(receipt, motion=True)


def test_speed60_receipt_does_not_relax_thirty_kmh_or_accept_over_eighty():
    from scripts.control_ac_session import validate_probe_receipt

    receipt = dict(motion_probe_passed=True, probe_seen_enabled=True,
                   maximum_position_delta_m=600, maximum_speed_kmh=70,
                   telemetry=[dict(ai_controlled=True, paused=False, gas=.2)]*2)
    validate_probe_receipt(receipt, motion=True, collection=True, fast=True, speed60=True)
    with pytest.raises(ValueError):
        validate_probe_receipt(receipt, motion=True, collection=True, fast=True)
    for changed in [dict(maximum_speed_kmh=80.01), dict(maximum_position_delta_m=720.01)]:
        with pytest.raises(ValueError):
            validate_probe_receipt(dict(receipt, **changed), motion=True,
                                   collection=True, fast=True, speed60=True)


def test_lateral_recovery_requires_explicit_capability_and_observed_inside_motion():
    from scripts.control_ac_session import (
        validate_expert_probe,
        validate_lateral_recovery_receipt,
    )

    status = dict(paused=True, ai_controlled=False, physics_api_allowed=True,
                  capabilities=dict(expert_collect30=True))
    with pytest.raises(ValueError):
        validate_expert_probe(status, motion=True, collection=True, lateral=True)
    status['capabilities']['expert_lateral_recovery15'] = True
    validate_expert_probe(status, motion=True, collection=True, lateral=True)
    receipt = dict(lateral_recovery15_passed=True, before=dict(wheels_outside=1),
        after=dict(wheels_outside=0), last_driving_wheels_outside=0,
        maximum_position_delta_m=10., telemetry=[dict(ai_controlled=True,
            paused=False, phase='driving', wheels_outside=0, speed_kmh=10)])
    validate_lateral_recovery_receipt(receipt)
    for field, value in [('lateral_recovery15_passed', False), ('telemetry', []),
                         ('last_driving_wheels_outside', 1), ('after', dict(wheels_outside=1))]:
        with pytest.raises(ValueError):
            validate_lateral_recovery_receipt(dict(receipt, **{field: value}))


def test_speed20_recovery_cannot_reuse30_profile_capability_or_speed_guard():
    from scripts.control_ac_session import validate_expert_probe, validate_probe_receipt

    status = dict(paused=True, ai_controlled=False, physics_api_allowed=True,
                  capabilities=dict(expert_speed_recovery15=True))
    with pytest.raises(ValueError):
        validate_expert_probe(status, motion=True, collection=True, fast=True,
                              recovery=True, warm20=True)
    status['capabilities']['expert_speed20_recovery15'] = True
    validate_expert_probe(status, motion=True, collection=True, fast=True,
                          recovery=True, warm20=True)
    receipt = dict(motion_probe_passed=True, probe_seen_enabled=True,
                   maximum_position_delta_m=200, maximum_speed_kmh=24,
                   telemetry=[dict(ai_controlled=True, paused=False, gas=.2)]*2)
    validate_probe_receipt(receipt, motion=True, collection=True, fast=True, warm20=True)
    for field, value in [('maximum_speed_kmh', 25.1), ('maximum_position_delta_m', 221)]:
        with pytest.raises(ValueError):
            validate_probe_receipt(dict(receipt, **{field: value}), motion=True,
                                   collection=True, fast=True, warm20=True)
