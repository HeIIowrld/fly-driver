"""Bounded virtual-device experiment with fake SDK/sensors; no native input."""

from types import SimpleNamespace

import pytest
from test_ac_vjoy import FakeSDK, bindings

from flybrain_drive.ac_axis_probe import STAGES, run_stationary_probe
from flybrain_drive.ac_vjoy import VJoySession


def sample(*, speed=0, gas=0, brake=1):
    return SimpleNamespace(fields=dict(speedKmh=speed, gas=gas, brake=brake))


def test_axis_experiment_is_finite_and_records_raw_samples_without_claiming_applied_response():
    sdk = FakeSDK()
    session = VJoySession(sdk, 1, bindings())
    events, waits = [], []
    result = run_stationary_probe(session, sample, lambda: None, events.append,
                                   samples_per_stage=2, sleep=waits.append)
    assert len(waits) == 2 * len(STAGES)
    assert len(sdk.frames) == session.submit_attempts == 2 * len(STAGES) + 2
    assert result['experiment_requests'] == 2 * len(STAGES) + 1
    assert not result['response_verified'] and not session.owned
    assert [e['kind'] for e in events] == ['open'] + ['axis_sample'] * 16 + ['close']
    assert all(not e['request']['game_applied'] for e in events)
    assert events[-1]['request']['requested_control'] == dict(steering=0., throttle=0., brake=1.)


def test_profile_guard_fails_before_acquisition_and_stale_sensor_cleanup_releases_device():
    sdk = FakeSDK()
    session = VJoySession(sdk, 1, bindings())

    def changed():
        raise RuntimeError('profile changed')

    with pytest.raises(RuntimeError, match='profile changed'):
        run_stationary_probe(session, sample, changed, lambda e: None, sleep=lambda _: None)
    assert sdk.acquires == session.submit_attempts == 0

    def stale():
        raise TimeoutError('no new sensor packet')

    events = []
    with pytest.raises(TimeoutError, match='no new sensor'):
        run_stationary_probe(session, stale, lambda: None, events.append, sleep=lambda _: None)
    assert sdk.acquires == 1 and not session.owned
    assert events[-1]['kind'] == 'close' and sdk.status == 1


def test_missing_neutral_feedback_and_motion_abort_without_continuing_stages():
    for bad in (sample(gas=.5, brake=0), sample(speed=3)):
        sdk = FakeSDK()
        session = VJoySession(sdk, 1, bindings())
        events = []
        with pytest.raises(RuntimeError, match='neutral|speed'):
            run_stationary_probe(session, lambda bad=bad: bad, lambda: None, events.append,
                                   sleep=lambda _: None)
        assert not session.owned and len(sdk.frames) <= 3
        assert events[-1]['request']['requested_control']['throttle'] == 0
