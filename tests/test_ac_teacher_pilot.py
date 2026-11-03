"""No live game input: test acquisition gates, measured feedback and interruption cleanup."""

from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_teacher_pilot import (
    collection_speed_request,
    run_teacher_segment,
    validate_calibration,
)

CODEC = dict(legacy_control_max_delta=[.1333333333, .48, .48],
             legacy_control_bounds=[-1, 1], legacy_product_dtype='float32')


class Device:
    owned = False
    submit_attempts = 0

    def open(self):
        self.owned = True
        return self.submit(0., 0., 1.)

    def submit(self, *values):
        self.submit_attempts += 1
        return dict(requested_control=values, driver_accepted=True, game_applied=False)

    def close(self):
        self.owned = False
        return self.submit(0., 0., 1.)


def observation(*, gear=2, speed=0):
    return SimpleNamespace(state=dict(speed=speed, gap=.1, actualGear=gear, out_of_track=0,
        steerAngle=0., accStatus=0., brakeStatus=1.), observation=np.zeros(125),
        sdk_sequence=1, native_packet_id=1)


def teacher():
    # First decision decreases held brake and increases throttle independently.
    return SimpleNamespace(predict=lambda _: dict(
        deterministic_policy_action=np.array([[0., 1., -1.]])))


@pytest.mark.parametrize('gear', [0, 1])
def test_not_in_gear_never_acquires_even_neutral_device(gear):
    device = Device()
    with pytest.raises(ValueError, match='Forward gear'):
        run_teacher_segment(device, lambda: observation(gear=gear), teacher(), CODEC,
            lambda: None, lambda event: None, decisions=1, sleep=lambda _: None)
    assert device.submit_attempts == 0 and not device.owned


def test_segment_starts_from_measured_neutral_logs_feedback_and_releases():
    device, events = Device(), []
    result = run_teacher_segment(device, observation, teacher(), CODEC,
        lambda: None, events.append, decisions=2, sleep=lambda _: None)
    requests = [event for event in events if event['kind'] == 'teacher_request']
    assert result['decisions'] == 2 and not result['fly_actor_driving_verified']
    assert [event['kind'] for event in events] == [
        'open', 'teacher_request', 'feedback', 'teacher_request', 'feedback', 'close']
    assert requests[0]['legacy_absolute_request'] == pytest.approx([0., -.52, .52])
    assert requests[0]['request']['requested_control'] == pytest.approx([0., .24, .76])
    assert events[-1]['request']['requested_control'] == (0., 0., 1.)
    assert device.submit_attempts == 4 and not device.owned


@pytest.mark.parametrize('failure', [TimeoutError('No new sensor packet'), KeyboardInterrupt()])
def test_sensor_failure_after_request_still_requests_neutral_and_releases(failure):
    device, events, count = Device(), [], 0

    def receive():
        nonlocal count
        count += 1
        if count == 6:  # Start check + four neutral history rows, then requested action.
            raise failure
        return observation()

    with pytest.raises(type(failure)):
        run_teacher_segment(device, receive, teacher(), CODEC, lambda: None, events.append,
                            decisions=2, sleep=lambda _: None)
    assert [event['kind'] for event in events] == ['open', 'teacher_request', 'close']
    assert events[-1]['request']['requested_control'] == (0., 0., 1.)
    assert device.submit_attempts == 3 and not device.owned


def test_calibration_rejects_450_gain_wrong_engine_and_profile():
    config = dict(profile_sha256='profile', dll_sha256='sdk', device_id=1, bindings=['axes'])
    record = dict(config, status='completed', producer_session_id='engine', samples=32,
        request_amplitude=1., sdk_raw_per_normalized_request=270.,
        sdk_steering_stage_medians=[0., 270., -270., 0.])
    validate_calibration(record, config, 'engine')
    for changed in [dict(sdk_raw_per_normalized_request=450.),
                    dict(producer_session_id='old'), dict(profile_sha256='old')]:
        with pytest.raises(ValueError):
            validate_calibration(dict(record, **changed), config, 'engine')


def test_explicit_rescaling_keeps_teacher_history_and_requires_endpoint_coverage():
    config = dict(profile_sha256='profile', dll_sha256='sdk', device_id=1, bindings=['axes'])
    record = dict(config, status='completed', producer_session_id='engine', samples=32,
        request_amplitude=.6, sdk_raw_per_normalized_request=450.,
        sdk_steering_stage_medians=[0., 270., -270., 0.])
    validate_calibration(record, config, 'engine', steering_request_scale=.6)
    with pytest.raises(ValueError):
        validate_calibration(record, config, 'engine')
    with pytest.raises(ValueError):
        validate_calibration(dict(record, request_amplitude=.1), config, 'engine',
                             steering_request_scale=.6)
    device, events = Device(), []
    policy = SimpleNamespace(predict=lambda _: dict(
        deterministic_policy_action=np.array([[.75, 0., 0.]])))
    run_teacher_segment(device, observation, policy, CODEC, lambda: None, events.append,
        decisions=2, sleep=lambda _: None, steering_request_scale=.6)
    requests = [event for event in events if event['kind'] == 'teacher_request']
    assert requests[0]['legacy_absolute_request'][0] == pytest.approx(.1)
    assert requests[1]['legacy_absolute_request'][0] == pytest.approx(.2)
    assert requests[0]['request']['requested_control'][0] == pytest.approx(.06)
    assert requests[1]['request']['requested_control'][0] == pytest.approx(.12)


def test_missing_image_after_neutral_hold_prevents_any_teacher_request():
    device, events = Device(), []

    def missing_image(current):
        raise ValueError('No preceding fresh image')

    with pytest.raises(ValueError, match='fresh image'):
        run_teacher_segment(device, observation, teacher(), CODEC, lambda: None, events.append,
            decisions=2, sleep=lambda _: None, image_context=missing_image)
    assert [event['kind'] for event in events] == ['open', 'close']
    assert device.submit_attempts == 2 and not device.owned


def test_device_acquisition_collects_three_real_new_history_rows_before_driving():
    device, events, resets = Device(), [], []
    count = None

    def reset(reason):
        nonlocal count
        assert device.owned
        resets.append(reason)
        count = 0

    def receive():
        nonlocal count
        result = observation()
        if count is not None:
            count += 1
            if count <= 3:
                result.observation = None
        return result

    run_teacher_segment(device, receive, teacher(), CODEC, lambda: None, events.append,
        decisions=1, sleep=lambda _: None, reset_history=reset)
    assert resets == ['neutral_device_acquisition'] and count == 5
    assert sum(event['kind'] == 'teacher_request' for event in events) == 1
    assert not device.owned


def test_speed_abort_preserves_rejected_feedback_and_releases_device():
    device, events = Device(), []
    rows = iter([observation()] * 5 + [observation(speed=17.)])
    with pytest.raises(ValueError, match='speed_kmh=61.2/60'):
        run_teacher_segment(device, lambda: next(rows), teacher(), CODEC,
            lambda: None, events.append, decisions=2, sleep=lambda _: None)
    assert [row['kind'] for row in events] == ['open', 'teacher_request', 'feedback', 'close']
    assert events[-2]['speed_m_s'] == 17.
    assert not device.owned and events[-1]['request']['requested_control'] == (0., 0., 1.)


def test_collection_governor_preserves_steering_and_expert_braking_with_explicit_headroom():
    request = np.array([.3, 1., -.2])
    np.testing.assert_allclose(collection_speed_request(request, 20., 40.), request)
    limited = collection_speed_request(request, 36., 40.)
    np.testing.assert_allclose(limited, [.3, 0., -.2])
    overspeed = collection_speed_request([.3, 1., -1.], 43., 40.)
    np.testing.assert_allclose(overspeed, [.3, -1., -.6])
    device = Device()
    with pytest.raises(ValueError, match='headroom|10km/h'):
        run_teacher_segment(device, observation, teacher(), CODEC, lambda: None, lambda _: None,
                            collection_speed_kmh=55., max_speed_kmh=60.)
    assert not device.owned and device.submit_attempts == 0


def test_relative_teacher_accumulates_executed_governed_request_not_unlimited_windup():
    device, events = Device(), []
    rows = iter([observation()] * 5 + [observation(speed=41/3.6)] * 3)
    run_teacher_segment(device, lambda: next(rows), teacher(), CODEC, lambda: None, events.append,
        decisions=3, sleep=lambda _: None, collection_speed_kmh=40.)
    requests = [r for r in events if r['kind'] == 'teacher_request']
    assert requests[1]['speed_governor_changed_request']
    assert requests[1]['legacy_absolute_request'][1] == -1.
    assert requests[2]['ungoverned_legacy_request'][1] == pytest.approx(-.52)
    assert not device.owned
