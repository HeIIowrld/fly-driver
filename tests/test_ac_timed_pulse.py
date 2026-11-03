import threading
import time

import pytest

from flybrain_drive.ac_timed_pulse import TimedSteeringPulse


class SDK:
    def __init__(self):
        self.requests = []

    def submit(self, **control):
        stamp = time.perf_counter()
        self.requests.append(control)
        return dict(request_sequence=len(self.requests), requested_control=control,
                    driver_call_finished_perf_counter_seconds=stamp)


def test_timer_removes_pulse_without_an_inference_update_at_the_deadline():
    sdk, events, ended = SDK(), [], threading.Event()

    def emit(event):
        events.append(event)
        if event['kind'] == 'timed_pulse_off':
            ended.set()

    pulse = TimedSteeringPulse(sdk, emit)
    try:
        base = dict(steering=.02, throttle=0., brake=1.)
        pulse.submit(base, source_render_started_lower=time.perf_counter())
        on = pulse.begin(1)
        # Refresh inputs during the pulse, then leave its ending to the worker.
        for _ in range(3):
            time.sleep(.07)
            pulse.submit(base, source_render_started_lower=time.perf_counter())
        assert ended.wait(.3)
        off = next(e for e in events if e['kind'] == 'timed_pulse_off')
        width = (off['request']['driver_call_finished_perf_counter_seconds']
                 - on['request']['driver_call_finished_perf_counter_seconds'])
        assert .28 <= width < .36
        assert off['request']['requested_control']['steering'] == .02
    finally:
        pulse.stop()


def test_expired_command_lease_brakes_and_cannot_silently_resume():
    sdk, events, expired = SDK(), [], threading.Event()

    def emit(event):
        events.append(event)
        if event['kind'] == 'timed_failure_brake':
            expired.set()

    pulse = TimedSteeringPulse(sdk, emit)
    try:
        pulse.submit(dict(steering=.03, throttle=.04, brake=0.),
                     source_render_started_lower=time.perf_counter())
        assert expired.wait(.5)
        assert events[-1]['safe_override']
        assert events[-1]['request']['requested_control']['brake'] == 1.
        with pytest.raises(RuntimeError, match='expired'):
            pulse.submit(dict(steering=0., throttle=0., brake=1.),
                         source_render_started_lower=time.perf_counter())
    finally:
        pulse.stop()


def test_cancel_prevents_late_pulse_off_write_and_rejects_stale_initial_source():
    sdk, events = SDK(), []
    pulse = TimedSteeringPulse(sdk, events.append)
    with pytest.raises(ValueError, match='stale'):
        pulse.submit(dict(steering=0., throttle=0., brake=1.),
                     source_render_started_lower=time.perf_counter()-1.)
    pulse.submit(dict(steering=0., throttle=0., brake=1.),
                 source_render_started_lower=time.perf_counter())
    pulse.begin(-1)
    pulse.stop()
    count = len(sdk.requests)
    assert not pulse.thread.is_alive() and events[-1]['kind'] == 'timed_cancel_brake'
    with pytest.raises(RuntimeError, match='closed'):
        pulse.begin(1)
    assert len(sdk.requests) == count


def test_cancel_log_failure_still_joins_worker_before_caller_can_release_device():
    sdk = SDK()

    def emit(event):
        if event['kind'] == 'timed_cancel_brake':
            raise OSError('synthetic log failure')

    pulse = TimedSteeringPulse(sdk, emit)
    pulse.submit(dict(steering=0., throttle=0., brake=1.),
                 source_render_started_lower=time.perf_counter())
    with pytest.raises(RuntimeError, match='worker stopped'):
        pulse.stop()
    assert not pulse.thread.is_alive() and getattr(pulse.wake, 'closed', True)


def test_fresh_command_from_older_valid_frame_gets_only80ms_lease_not_a_redated_source():
    sdk, events = SDK(), []
    pulse = TimedSteeringPulse(sdk, events.append)
    try:
        original = time.perf_counter()-.12
        event = pulse.submit(dict(steering=0., throttle=0., brake=1.),
                             source_render_started_lower=original)
        assert event['source_render_started_lower'] == original
        assert event['accepted_command_lease_deadline'] == (
            event['request']['driver_call_finished_perf_counter_seconds']+.08)
        time.sleep(.04)
        # Old camera source is now stale; its timestamp cannot be reused for a new request.
        with pytest.raises(ValueError, match='stale'):
            pulse.submit(dict(steering=0., throttle=0., brake=1.),
                         source_render_started_lower=original)
    finally:
        pulse.stop()


def test_late_submit_cannot_renew_a_lease_before_the_worker_observes_its_expiry():
    stamp, events = [1.], []

    class ClockSDK:
        def submit(self, **control):
            return dict(request_sequence=1, requested_control=control,
                        driver_call_finished_perf_counter_seconds=stamp[0])

    pulse = TimedSteeringPulse(ClockSDK(), events.append, clock=lambda: stamp[0])
    try:
        pulse.submit(dict(steering=0., throttle=0., brake=1.), source_render_started_lower=1.)
        with pulse.lock:
            stamp[0] = 1.081
            with pytest.raises(RuntimeError, match='lease expired'):
                pulse.submit(dict(steering=0., throttle=0., brake=1.),
                             source_render_started_lower=stamp[0])
        assert pulse.failed is not None
    finally:
        pulse.stop()
