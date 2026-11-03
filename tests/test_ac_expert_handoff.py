import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from flybrain_drive.ac_expert_handoff import OwnedExpertHandoff, validate_owned_handoff


def ready():
    return dict(paused=False, live=True, ai_controlled=False, physics_api_allowed=True,
        is_online=False, car_id='bmw_z4_gt3', track_id='monza', speed_kmh=17.,
        generation='one', epoch=1, script_clock=1., camera_request_id='owned-camera',
        capabilities=dict(expert_moving_handoff3=True))


def test_handoff_heartbeat_metadata_comes_from_same_open_file_when_path_is_replaced(
        tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    folder = tmp_path/'flybrain/csp-session'
    folder.mkdir(parents=True)
    heartbeat = folder/'status.json'
    heartbeat.write_text(json.dumps(ready()), encoding='utf8')
    stamp = time.time()-.1
    os.utime(heartbeat, (stamp, stamp))
    original = Path.stat

    def transient_path_stat(path, *args, **kwargs):
        if path == heartbeat:
            raise FileNotFoundError('Atomic heartbeat replacement after successful read')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', transient_path_stat)
    handoff = OwnedExpertHandoff(SimpleNamespace(), tmp_path/'receipt')
    assert handoff.status()['generation'] == 'one'


@pytest.mark.parametrize('status,owned', [(1, True), (2, True), (0, False)])
def test_handoff_requires_same_process_owned_device(status, owned):
    session = SimpleNamespace(owned=owned, device_id=1,
                              sdk=SimpleNamespace(GetVJDStatus=lambda _: status))
    with pytest.raises(ValueError):
        validate_owned_handoff(session, ready(), .1)


def test_handoff_requests_once_and_pins_observed_ai_owner(tmp_path, monkeypatch):
    import flybrain_drive.ac_expert_handoff as module

    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    root = tmp_path/'flybrain/csp-session'
    root.mkdir(parents=True)
    (root/'status.json').write_text(json.dumps(ready()), encoding='utf8')
    stamp = time.time()-.1
    os.utime(root/'status.json', (stamp, stamp))
    camera = SimpleNamespace(request_id='owned-camera', clock=SimpleNamespace(generation='one'))
    session = SimpleNamespace(owned=True, device_id=1,
                              sdk=SimpleNamespace(GetVJDStatus=lambda _: 0))
    original = module.write_json
    writes = []

    def write(path, value):
        original(path, value)
        if path == root/'request.json':
            writes.append(value)
            original(root/'status.json', dict(ready(), ai_controlled=True,
                expert_probe_active=True, expert_probe_id=value['id']))
            stamp = time.time()-.1
            os.utime(root/'status.json', (stamp, stamp))

    monkeypatch.setattr(module, 'write_json', write)
    handoff = OwnedExpertHandoff(camera, tmp_path/'receipt')
    receipt = handoff.activate(session, SimpleNamespace(
        diagnostic=dict(speed_kmh=17., tyres_out=0)))
    assert receipt['ai_active_observed'] and writes[0]['student_owner_pid'] == os.getpid()
    with pytest.raises(ValueError, match='Never repeat'):
        handoff.activate(session, SimpleNamespace())
    assert len(writes) == 1 and session.owned  # caller must close only after this confirmation


def test_handoff_status_does_not_accept_changed_epoch_or_unknown_api(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    root = tmp_path/'flybrain/csp-session'
    root.mkdir(parents=True)
    handoff = OwnedExpertHandoff(SimpleNamespace(), tmp_path/'receipt')
    handoff.request = dict(generation='one', epoch=1)
    for change in [dict(epoch=2), dict(error='unknown outcome')]:
        (root/'status.json').write_text(json.dumps(dict(ready(), **change)), encoding='utf8')
        with pytest.raises(ValueError):
            handoff.status()


def test_after_release_motion_excludes_stale_packets_stop_and_full_brake():
    from flybrain_drive.ac_expert_handoff import effective_motion_after_release

    sample = SimpleNamespace(received_monotonic_seconds=1.4,
                             fields=dict(speedKmh=17., brake=0.))
    assert effective_motion_after_release(sample, 1.)
    sample.received_monotonic_seconds = 1.1
    assert not effective_motion_after_release(sample, 1.)
    sample.received_monotonic_seconds = 1.4
    sample.fields['brake'] = 1.
    assert not effective_motion_after_release(sample, 1.)
    sample.fields.update(speedKmh=0., brake=0.)
    assert not effective_motion_after_release(sample, 1.)


def mapping_events():
    return [dict(kind='raw_pair', native=dict(packet_id=1)),
        dict(kind='student_request', prediction=[0, 0, 0],
             actor_input=dict(observation_index=0, native_packet_id=1,
             image={}, decision_monotonic_seconds=1.)),
        dict(kind='expert_handoff'), dict(kind='handoff_vjoy_released', vjoy_status=1),
        dict(kind='raw_pair', native=dict(packet_id=2)),
        dict(kind='expert_observation', observation_index=1, native_packet_id=2,
             camera={}, decision_monotonic_seconds=1.3)]


def test_prefix_suffix_mapping_never_relabels_student_or_reorders_handoff():
    from flybrain_drive.ac_expert_handoff import handoff_observation_records

    records, pairs = handoff_observation_records(mapping_events(), [False, True])
    assert not records[0]['expert'] and records[1]['expert'] and set(pairs) == {1, 2}
    with pytest.raises(ValueError):
        handoff_observation_records(mapping_events(), [True, True])
    with pytest.raises(ValueError):
        handoff_observation_records(mapping_events()+[mapping_events()[1]], [False, True])
    with pytest.raises(ValueError):
        handoff_observation_records(mapping_events()+[mapping_events()[0]], [False, True])
    early = mapping_events()
    early[3], early[5] = early[5], early[3]
    with pytest.raises(ValueError):
        handoff_observation_records(early, [False, True])


def test_expert_native_actor_receipts_do_not_require_teacher_only_sdk_truth():
    from flybrain_drive.ac_expert_handoff import handoff_observation_records

    events = mapping_events()
    events[-2]['kind'] = 'expert_raw_native'
    rows, pairs = handoff_observation_records(events, [False, True])
    assert rows[-1]['expert'] and pairs[2]['packet_id'] == 2
    with pytest.raises(ValueError, match='Duplicate'):
        handoff_observation_records(events+[events[-2]], [False, True])


def test_waiting_rows_carry_context_without_more_student_requests_or_expert_labels():
    from flybrain_drive.ac_expert_handoff import handoff_observation_records

    events = mapping_events()
    events.insert(2, dict(kind='expert_handoff_requested'))
    events.insert(3, dict(kind='raw_pair', native=dict(packet_id=3)))
    events.insert(4, dict(kind='handoff_wait_observation', prediction=[0, 0, 0],
        prediction_applied=False, actor_input=dict(observation_index=1, native_packet_id=3,
            image={}, decision_monotonic_seconds=1.04)))
    events[-1]['observation_index'] = 2
    records, _ = handoff_observation_records(events, [False, False, True])
    assert records[1]['bridge'] and not records[1]['expert']
    events[4]['prediction_applied'] = True
    with pytest.raises(ValueError):
        handoff_observation_records(events, [False, False, True])


def test_extended_handoff_requires_separate_capability_and_finite_preset():
    session = SimpleNamespace(owned=True, device_id=1,
                              sdk=SimpleNamespace(GetVJDStatus=lambda _: 0))
    with pytest.raises(ValueError):
        validate_owned_handoff(session, ready(), .1, seconds=30)
    status = ready()
    status['capabilities']['expert_moving_handoff30'] = True
    validate_owned_handoff(session, status, .1, seconds=30)
    for seconds in [True, 10, 120]:
        with pytest.raises(ValueError):
            validate_owned_handoff(session, status, .1, seconds=seconds)


@pytest.mark.parametrize('context,threshold', [('curve200', 200), ('curve320', 320)])
def test_curve_evidence_rejects_early_or_discontinuous_prefix(context, threshold):
    from flybrain_drive.ac_expert_handoff import validate_curve_handoff_evidence

    args = dict(handoff_context=context, trial_profile='receipt15_route480', decisions=2000,
                handoff_native_receipts=True)
    request = dict(hold_seconds=30)
    for distance, error in [(threshold-.01, None), (float('nan'), None),
                            (float('inf'), None), (threshold, 'gap'), (True, None)]:
        with pytest.raises(ValueError):
            validate_curve_handoff_evidence(args, request,
                dict(route_progress_evidence=dict(distance_m=distance, continuity_error=error)))
    validate_curve_handoff_evidence(args, request,
        dict(route_progress_evidence=dict(distance_m=threshold, continuity_error=None)))
    with pytest.raises(ValueError):
        validate_curve_handoff_evidence(dict(args, handoff_native_receipts=False), request,
            dict(route_progress_evidence=dict(distance_m=threshold, continuity_error=None)))


@pytest.mark.parametrize('decisions,seconds,native,valid', [
    (800, 3, False, True), (800, 30, True, True), (1200, 30, True, True),
    (1200, 3, True, False), (1200, 30, False, False), (1201, 30, True, False),
    (True, 30, True, False), (2600, 30, True, False)])
def test_extended_launch_is_explicit_finite30second_native_data_only(
        decisions, seconds, native, valid):
    from flybrain_drive.ac_expert_handoff import validate_handoff_preset

    args = ('launch', decisions, seconds, native, 'receipt15_route240', 'csp', True)
    if valid:
        assert validate_handoff_preset(*args) == 0
    else:
        with pytest.raises(ValueError):
            validate_handoff_preset(*args)


def test_extended_launch_camera_budget_is_bounded_with_original_motion_profile():
    from flybrain_drive.ac_csp_vision import validate_camera_budget
    from flybrain_drive.ac_student_pilot import STUDENT_TRIAL_PROFILES

    profile = STUDENT_TRIAL_PROFILES['receipt15_route240']
    assert profile['speed'] == 20 and profile['clearance'] == 1 and profile['tyres'] == 0
    validate_camera_budget(1200+50+825, .04, profile['camera_bytes'])
