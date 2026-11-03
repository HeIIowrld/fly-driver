"""Synthetic trial callbacks only: verify the actor boundary and interruption cleanup."""

from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_student_pilot import (
    StudentTrialReading,
    has_fixed_20kmh_task,
    has_fixed_speed_task,
    run_student_trial,
)


class Clock:
    now = 100.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_tracking_task_contract_requires_explicit_known_kind_and_20kmh_speed():
    assert has_fixed_20kmh_task(dict(kind='geometric_recovery_fixed_task', maximum_speed_kmh=20))
    assert has_fixed_20kmh_task(dict(kind='executed_sac_speed_governed', collection_speed_kmh=20))
    assert not has_fixed_20kmh_task(
        dict(kind='executed_sac_speed_governed', collection_speed_kmh=70))
    assert not has_fixed_20kmh_task(dict(maximum_speed_kmh=20))
    assert not has_fixed_20kmh_task(None)


def test_30kmh_task_cannot_reuse_a_20kmh_checkpoint_contract():
    contract = dict(kind='geometric_recovery_fixed_task', maximum_speed_kmh=30)
    assert has_fixed_speed_task(contract, 30.)
    assert not has_fixed_20kmh_task(contract)
    assert not has_fixed_speed_task(dict(kind='unknown', maximum_speed_kmh=30), 30.)


def test_receipt_task_is_separate_and_requires_exact_horizon_clock_and_actuator_scale():
    from flybrain_drive.ac_student_pilot import has_applied_receipt_task

    contract = dict(kind='engine_ai_future_applied_control_receipt_v1', ai_speed_hint_kmh=15,
        ai_steer_multiplier=.9, actor_steering_device_scale=.6, target_receipt_horizon_seconds=.04,
        maximum_target_lateness_seconds=.025, clock='host_monotonic_QPC_time.perf_counter',
        physical_command_time_verified=False)
    assert has_applied_receipt_task(contract, 15., .6)
    assert not has_fixed_speed_task(contract, 15.)
    assert not has_fixed_speed_task(contract, 30.)
    assert not has_applied_receipt_task(contract, 30., .6)
    assert not has_applied_receipt_task(contract, 15., .7)
    for key in contract:
        missing = {k: v for k, v in contract.items() if k != key}
        assert not has_applied_receipt_task(missing, 15., .6)


class Device:
    owned = False

    def open(self):
        self.owned = True
        return self.submit(0., 0., 1.)

    def submit(self, *values):
        return dict(controls=values)

    def close(self):
        self.owned = False
        return self.submit(0., 0., 1.)


def reading(speed=0., x=0., gear=2):
    return StudentTrialReading(SimpleNamespace(fields=dict(gas=0., brake=1.)),
        dict(speed_kmh=speed, gear=gear, tyres_out=0, x=x, y=0., sdk_steer_raw=0.))


def inference(clock):
    def infer(native):
        assert isinstance(native, SimpleNamespace) and not hasattr(native, 'diagnostic')
        return dict(prediction=np.array([.3, .4, .1]),
                    oldest_input_monotonic_seconds=clock(), metadata=dict(body_only=True))
    return infer


def test_requests_are_unblended_student_outputs_with_axis_conversion_and_feedback():
    clock, device, events = Clock(), Device(), []
    result = run_student_trial(device, reading, inference(clock), lambda: None, events.append,
        decisions=2, clock=clock, sleep=clock.sleep)
    requests = [r for r in events if r['kind'] == 'student_request']
    assert len(requests) == 2 and result['decisions'] == 2
    assert requests[0]['request']['controls'] == pytest.approx((.18, .4, .1))
    assert not requests[0]['teacher_action_used']
    assert not requests[0]['diagnostic_pose_used_by_actor']
    assert events[-1]['kind'] == 'close' and not device.owned


@pytest.mark.parametrize('minimum_route', [200, 320])
@pytest.mark.parametrize('early_tyre_failure', [False, True])
def test_curve_handoff_waits_for_route_without_bypassing_the_original_tyre_guard(
        early_tyre_failure, minimum_route):
    clock, device, events = Clock(), Device(), []
    received, activated = 0, []

    def receive():
        nonlocal received
        step = max(0, received-1)
        received += 1
        value = course_reading(x=step*.2, speed=17. if step else 0.)
        value.diagnostic.update(track_length_m=1000., lap_count=0,
            normalized_spline_position=.1+step*.2/1000.,
            native_received_monotonic_seconds=clock())
        if early_tyre_failure and step == minimum_route*5-100:
            value.diagnostic['tyres_out'] = 1
        return value

    def handoff(owned, current):
        assert owned.owned and current.diagnostic['x'] >= minimum_route
        activated.append(current.diagnostic['x'])
        return dict(ai_active_observed=True)

    def execute():
        return run_student_trial(device, receive, inference(clock), lambda: None,
            events.append, decisions=2000, trial_profile='receipt15_route480',
            max_speed_kmh=20., max_displacement_m=640., clock=clock, sleep=clock.sleep,
            expert_handoff=handoff, expert_handoff_minimum_route_m=minimum_route)

    if early_tyre_failure:
        with pytest.raises(ValueError, match='envelope'):
            execute()
        assert not activated
    else:
        assert execute()['status'] == 'expert_handoff' and len(activated) == 1
        assert sum(row['kind'] == 'student_request' for row in events) >= minimum_route*5-1
    assert not device.owned
    assert events[-1]['request']['controls'] == (0., 0., 1.)


@pytest.mark.parametrize('initial', [reading(speed=1.), reading(gear=1)])
def test_moving_or_neutral_start_never_acquires(initial):
    clock, device, events = Clock(), Device(), []
    with pytest.raises(ValueError):
        run_student_trial(device, lambda: initial, inference(clock), lambda: None, events.append,
            clock=clock, sleep=clock.sleep)
    assert not events and not device.owned


@pytest.mark.parametrize('last', [reading(speed=6.), reading(x=.6)])
def test_motion_limit_preserves_feedback_and_releases(last):
    clock, device, events = Clock(), Device(), []
    rows = iter([reading(), reading(), last])
    with pytest.raises(ValueError):
        run_student_trial(device, lambda: next(rows), inference(clock), lambda: None, events.append,
            clock=clock, sleep=clock.sleep)
    assert [r['kind'] for r in events] == ['open', 'student_request', 'feedback', 'close']
    assert not device.owned


@pytest.mark.parametrize('failure', ['expired', 'slow', 'interrupt'])
def test_expired_inference_or_interrupt_never_sends_student_action(failure):
    clock, device, events = Clock(), Device(), []

    def infer(native):
        value = inference(clock)(native)
        if failure == 'interrupt':
            raise KeyboardInterrupt()
        if failure == 'expired':
            value['oldest_input_monotonic_seconds'] -= .16
        if failure == 'slow':
            clock.sleep(.09)
        return value

    with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else TimeoutError):
        run_student_trial(device, reading, infer, lambda: None, events.append,
            clock=clock, sleep=clock.sleep)
    assert [r['kind'] for r in events] == (['open', 'close'] if failure == 'interrupt'
                                         else ['open', 'close', 'expired_inference'])
    if failure != 'interrupt':
        assert events[-1]['action_request_sent'] is False
    assert not device.owned


def tracked_reading(x=0., *, gap=0., clearance=4., speed=0.):
    row = reading(speed=speed, x=x)
    row.diagnostic.update(reference_gap_m=gap, minimum_lateral_clearance_m=clearance)
    return row


def test_tracking_profile_stops_at_declared_travel_before_hard_distance_limit():
    clock, device, events = Clock(), Device(), []
    rows = iter([tracked_reading(), tracked_reading(), tracked_reading(x=15.1, speed=20.)])
    result = run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
        events.append, decisions=150, trial_profile='tracking_validation', max_speed_kmh=25.,
        max_displacement_m=20., clock=clock, sleep=clock.sleep)
    assert result['termination'] == 'planned_travel_reached'
    assert result['planned_travel_reached'] and result['planned_travel_target_m'] == 15.
    assert result['decisions'] == 1 and not device.owned
    assert events[-2]['kind'] == 'planned_stop' and events[-1]['kind'] == 'close'


@pytest.mark.parametrize('bad', [tracked_reading(gap=3.1), tracked_reading(clearance=.5),
                               tracked_reading(speed=26.)])
def test_tracking_envelope_is_measured_before_inference_and_neutrals_on_failure(bad):
    clock, device, events = Clock(), Device(), []
    rows = iter([tracked_reading(), tracked_reading(), bad])
    with pytest.raises(ValueError):
        run_student_trial(device, lambda: next(rows), inference(clock), lambda: None, events.append,
            decisions=150, trial_profile='tracking_validation', max_speed_kmh=25.,
            max_displacement_m=20., clock=clock, sleep=clock.sleep)
    assert events[-2]['kind'] == 'feedback' and events[-1]['kind'] == 'close'
    assert not device.owned


def test_first_profile_cannot_silently_accept_tracking_budget():
    clock, device = Clock(), Device()
    with pytest.raises(ValueError):
        run_student_trial(device, reading, inference(clock), lambda: None, lambda _: None,
            decisions=150, max_speed_kmh=25., max_displacement_m=20.,
            clock=clock, sleep=clock.sleep)
    assert not device.owned


@pytest.mark.parametrize('profile,decisions,travel,hard_distance', [
    ('tracking_30kmh', 300, 30., 40.), ('tracking_60m', 600, 60., 80.),
    ('tracking_120m', 700, 120., 160.), ('tracking_240m', 1200, 240., 320.)])
def test_30kmh_trial_has_explicit_travel_target_and_keeps_40kmh_guard(
        profile, decisions, travel, hard_distance):
    clock, device, events = Clock(), Device(), []
    rows = iter([tracked_reading(), tracked_reading(), tracked_reading(x=travel+.1, speed=31.)])
    result = run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
        events.append, decisions=decisions, trial_profile=profile, max_speed_kmh=40.,
        max_displacement_m=hard_distance, clock=clock, sleep=clock.sleep)
    assert result['planned_travel_reached'] and result['planned_travel_target_m'] == travel
    assert not device.owned
    rows = iter([tracked_reading(), tracked_reading(), tracked_reading(x=2., speed=41.)])
    with pytest.raises(ValueError):
        run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
            events.append, decisions=decisions, trial_profile=profile, max_speed_kmh=40.,
            max_displacement_m=hard_distance, clock=clock, sleep=clock.sleep)
    assert events[-1]['kind'] == 'close' and not device.owned


def test_60m_trial_does_not_allow_an_unbounded_decision_count():
    clock, device = Clock(), Device()
    with pytest.raises(ValueError):
        run_student_trial(device, reading, inference(clock), lambda: None, lambda _: None,
            decisions=601, trial_profile='tracking_60m', max_speed_kmh=40.,
            max_displacement_m=80., clock=clock, sleep=clock.sleep)
    assert not device.owned


def test_feedback_latency_does_not_accumulate_on_the_neural_40ms_cadence():
    clock, device, calls = Clock(), Device(), []

    def receive():
        clock.sleep(.006)
        return reading()

    def infer(native):
        calls.append(clock())
        return inference(clock)(native)

    result = run_student_trial(device, receive, infer, lambda: None, lambda _: None,
        decisions=6, clock=clock, sleep=clock.sleep)
    # Initial feedback alignment costs6ms; later decisions retain40ms, not46ms.
    np.testing.assert_allclose(np.diff(calls)[1:], .04, atol=1e-12)
    assert result['decisions'] == 6 and not device.owned


def test_final_guard_delay_cannot_send_an_already_expired_prediction():
    clock, device, events = Clock(), Device(), []
    guards = 0

    def guard():
        nonlocal guards
        guards += 1
        if guards == 3:
            clock.sleep(.16)

    with pytest.raises(TimeoutError):
        run_student_trial(device, reading, inference(clock), guard, events.append,
                          clock=clock, sleep=clock.sleep)
    assert [e['kind'] for e in events] == ['open', 'close', 'expired_inference']
    assert not device.owned and not events[-1]['action_request_sent']


def test_120m_trial_keeps_bounded_capture_and_decision_budget():
    import math
    assert math.ceil(700*.04/.03)+50 <= 1000
    clock, device = Clock(), Device()
    with pytest.raises(ValueError):
        run_student_trial(device, reading, inference(clock), lambda: None, lambda _: None,
            decisions=701, trial_profile='tracking_120m', max_speed_kmh=40.,
            max_displacement_m=160., clock=clock, sleep=clock.sleep)
    assert not device.owned


def test_240m_trial_remains_finite_and_below_one_minute_capture():
    import math
    frames = math.ceil(1200*.04/.03)+50
    assert frames <= 2000 and frames*.03+5 <= 60
    clock, device = Clock(), Device()
    with pytest.raises(ValueError):
        run_student_trial(device, reading, inference(clock), lambda: None, lambda _: None,
            decisions=1201, trial_profile='tracking_240m', max_speed_kmh=40.,
            max_displacement_m=320., clock=clock, sleep=clock.sleep)
    assert not device.owned


def course_reading(x=0., **kwargs):
    value = tracked_reading(x=x, gap=4., clearance=5., **kwargs)
    value.diagnostic['lap_invalidated'] = 0
    return value


def test_480m_profile_extends_continuous_episode_with_finite_capture_and_same_road_guards():
    from flybrain_drive.ac_csp_vision import validate_camera_budget
    frames = int(np.ceil(2400*.04/.03))+50
    assert frames == 3250 and frames*.03+5 == pytest.approx(102.5)
    validate_camera_budget(frames, .03, 2*1024**3)
    clock, device = Clock(), Device()
    rows = iter([course_reading(), course_reading(), course_reading(x=480.1, speed=31.)])
    result = run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
        lambda _: None, decisions=2400, trial_profile='course_480m', max_speed_kmh=40.,
        max_displacement_m=640., clock=clock, sleep=clock.sleep)
    assert result['planned_travel_reached'] and result['measured_travel_m'] == 480.1
    assert not result['full_lap_verified'] and not device.owned


def test_480m_extension_does_not_allow_off_track_tyres_or_an_extra_decision():
    clock, device = Clock(), Device()
    invalid = course_reading(x=3.)
    invalid.diagnostic['tyres_out'] = 1
    rows = iter([course_reading(), course_reading(), invalid])
    with pytest.raises(ValueError, match='envelope'):
        run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
            lambda _: None, decisions=2400, trial_profile='course_480m', max_speed_kmh=40.,
            max_displacement_m=640., clock=clock, sleep=clock.sleep)
    assert not device.owned
    with pytest.raises(ValueError):
        run_student_trial(device, course_reading, inference(clock), lambda: None,
            lambda _: None, decisions=2401, trial_profile='course_480m', max_speed_kmh=40.,
            max_displacement_m=640., clock=clock, sleep=clock.sleep)


@pytest.mark.parametrize('tyres', [0, 1, 2])
def test_game_valid_profile_uses_game_validity_while_old_road_profile_stays_strict(tyres):
    from flybrain_drive.ac_student_pilot import check_trial_reading
    current = course_reading(x=480.1, speed=25.)
    current.diagnostic.update(tyres_out=tyres, minimum_lateral_clearance_m=.8)
    with pytest.raises(ValueError):
        check_trial_reading(current, max_speed_kmh=40., max_tyres_out=0,
                            min_lateral_clearance_m=1., require_valid_lap=True)
    clock, device = Clock(), Device()
    rows = iter([course_reading(), course_reading(), current])
    result = run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
        lambda _: None, decisions=2400, trial_profile='game_valid_480m', max_speed_kmh=40.,
        max_displacement_m=640., clock=clock, sleep=clock.sleep)
    assert result['planned_travel_reached'] and not result['full_lap_verified']
    assert not device.owned


@pytest.mark.parametrize('field,value', [('tyres_out', 3), ('lap_invalidated', 1),
                                       ('lap_invalidated', None)])
def test_game_valid_profile_rejects_invalid_unknown_lap_or_three_outside_tyres(field, value):
    current = course_reading(x=4.)
    current.diagnostic[field] = value
    clock, device = Clock(), Device()
    rows = iter([course_reading(), course_reading(), current])
    with pytest.raises(ValueError):
        run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
            lambda _: None, decisions=2400, trial_profile='game_valid_480m', max_speed_kmh=40.,
            max_displacement_m=640., clock=clock, sleep=clock.sleep)
    assert not device.owned


def test_route_completion_uses_sdk_progress_over_curve_chord_without_actor_truth():
    import math
    clock, device, events = Clock(), Device(), []
    received = 0

    def receive():
        nonlocal received
        step = max(0, received-1)
        received += 1
        arc, radius = step*.3, 200.
        value = course_reading(x=radius*math.sin(arc/radius), speed=25. if step else 0.)
        value.diagnostic.update(y=radius*(1-math.cos(arc/radius)), track_length_m=1000.,
            lap_count=0, normalized_spline_position=.1+arc/1000.,
            native_received_monotonic_seconds=clock())
        return value

    result = run_student_trial(device, receive, inference(clock), lambda: None, events.append,
        decisions=2400, trial_profile='game_valid_route_480m', max_speed_kmh=40.,
        max_displacement_m=640., clock=clock, sleep=clock.sleep)
    last = [event['diagnostic'] for event in events if event['kind'] == 'feedback'][-1]
    assert result['planned_travel_reached'] and result['planned_travel_metric'] == 'sdk_route'
    assert result['measured_travel_m'] >= 480.
    assert math.hypot(last['x'], last['y']) < 400.
    assert not result['full_lap_verified'] and not device.owned


def test_course_profile_checks_physical_road_validity_and_keeps_old_tracking_failure():
    from flybrain_drive.ac_student_pilot import check_trial_reading
    with pytest.raises(ValueError, match='reference-gap'):
        check_trial_reading(course_reading(), max_reference_gap_m=3.)
    clock, device, events = Clock(), Device(), []
    rows = iter([course_reading(), course_reading(), course_reading(x=240.1, speed=31.)])
    result = run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
        events.append, decisions=1200, trial_profile='course_240m', max_speed_kmh=40.,
        max_displacement_m=320., clock=clock, sleep=clock.sleep)
    assert result['planned_travel_reached'] and not result['full_lap_verified']
    assert not device.owned


@pytest.mark.parametrize('field,value', [('tyres_out', 1), ('lap_invalidated', 1),
    ('lap_invalidated', None), ('minimum_lateral_clearance_m', .99),
    ('minimum_lateral_clearance_m', None), ('speed_kmh', 41.), ('x', 321.)])
def test_course_profile_stops_on_physical_boundaries_unknown_validity_or_old_motion_limits(
        field, value):
    clock, device, events = Clock(), Device(), []
    invalid = course_reading(x=2.)
    invalid.diagnostic[field] = value
    rows = iter([course_reading(), course_reading(), invalid])
    with pytest.raises(ValueError):
        run_student_trial(device, lambda: next(rows), inference(clock), lambda: None,
            events.append, decisions=1200, trial_profile='course_240m', max_speed_kmh=40.,
            max_displacement_m=320., clock=clock, sleep=clock.sleep)
    assert events[-1]['kind'] == 'close' and not device.owned


@pytest.mark.parametrize('profile,decisions,target,displacement', [
    ('receipt15_route60', 800, 60., 90.),
    ('receipt15_route240', 2600, 240., 320.),
    ('receipt15_route480', 2900, 480., 640.)])
def test_receipt_trial_measures_course_progress_and_keeps_actor_truth_boundary(
        profile, decisions, target, displacement):
    import math

    clock, device, events = Clock(), Device(), []
    received = 0

    def receive():
        nonlocal received
        step = max(0, received-1)
        received += 1
        arc, radius = step*(15./3.6)*.04, 150.
        value = course_reading(x=radius*math.sin(arc/radius), speed=15. if step else 0.)
        value.diagnostic.update(y=radius*(1-math.cos(arc/radius)), track_length_m=1000.,
            lap_count=0, normalized_spline_position=.1+arc/1000.,
            native_received_monotonic_seconds=clock())
        return value

    result = run_student_trial(device, receive, inference(clock), lambda: None, events.append,
        decisions=decisions, trial_profile=profile, max_speed_kmh=20.,
        max_displacement_m=displacement, clock=clock, sleep=clock.sleep)
    assert result['planned_travel_reached'] and result['measured_travel_m'] >= target
    assert result['planned_travel_metric'] == 'sdk_route'
    assert not result['full_lap_verified'] and not device.owned


@pytest.mark.parametrize('field,value', [('speed_kmh', 20.1), ('tyres_out', 1),
    ('lap_invalidated', 1), ('minimum_lateral_clearance_m', .99)])
def test_receipt_profile_does_not_relax_speed_tyres_lap_or_clearance(field, value):
    from flybrain_drive.ac_student_pilot import check_trial_reading

    value_reading = course_reading()
    value_reading.diagnostic[field] = value
    with pytest.raises(ValueError):
        check_trial_reading(value_reading, max_speed_kmh=20., max_tyres_out=0,
                            require_valid_lap=True, min_lateral_clearance_m=1.)


@pytest.mark.parametrize('speed,clearance', [(17., 2.), (16., 1.4)])
@pytest.mark.parametrize('fails', [False, True, 'unobserved'])
def test_expert_handoff_stops_student_requests_and_always_returns_vjoy(fails, speed, clearance):
    clock, device, events = Clock(), Device(), []
    triggered = course_reading(speed=speed, x=.1)
    triggered.diagnostic['minimum_lateral_clearance_m'] = clearance
    rows = iter([course_reading(), course_reading(), triggered])
    seen = []

    def receive():
        value = next(rows)
        value.diagnostic.update(track_length_m=1000., lap_count=0,
            normalized_spline_position=.1+value.diagnostic['x']/1000.,
            native_received_monotonic_seconds=clock())
        return value

    def handoff(owned, current):
        assert owned.owned and current.diagnostic['speed_kmh'] == speed
        seen.append(True)
        if fails is True:
            raise RuntimeError('unknown activation')
        return dict(ai_active_observed=fails != 'unobserved')

    def execute():
        return run_student_trial(device, receive, inference(clock), lambda: None,
            events.append, decisions=800, trial_profile='receipt15_route240',
            max_speed_kmh=20., max_displacement_m=320., clock=clock, sleep=clock.sleep,
            expert_handoff=handoff)

    if fails is True:
        with pytest.raises(RuntimeError, match='unknown activation'):
            execute()
    elif fails == 'unobserved':
        with pytest.raises(ValueError, match='observed AI ownership'):
            execute()
    else:
        result = execute()
        assert result['status'] == 'expert_handoff' and not result['student_closed_loop_trial']
    assert len(seen) == 1
    assert len([r for r in events if r['kind'] == 'student_request']) == 1
    assert events[-1]['kind'] == 'close' and not device.owned
