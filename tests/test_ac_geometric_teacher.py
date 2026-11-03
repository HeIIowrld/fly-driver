"""Synthetic teacher geometry and cleanup; actual low-speed recovery needs game evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.ac_geometric_teacher import GeometricRecoveryTeacher, run_geometric_recovery

LINE = np.array([[-100., 0.], [0., 0.], [100., 0.], [100., 100.],
                 [0., 100.], [-100., 100.]])


def state(y=0., speed=0., yaw=0., gap=0.):
    return dict(world_position_x=0., world_position_y=y, speed=speed, yaw=yaw, gap=gap,
                actualGear=2, out_of_track=0, steerAngle=0., accStatus=0., brakeStatus=1.,
                sensors=np.ones(11)*10)


def test_measured_steering_sign_targets_both_sides_and_speed_responds_to_error():
    teacher = GeometricRecoveryTeacher(LINE, -.1)
    assert teacher.request(state(y=-5, gap=5))[0] < 0
    assert teacher.request(state(y=5, gap=-5))[0] > 0
    requested = teacher.request(state(speed=25/3.6))
    assert requested[1] == -1 and requested[2] > -1
    straight = teacher.request(state())
    assert straight[0] == 0 and straight[1] <= -.5
    with pytest.raises(ValueError, match='behind'):
        teacher.request(state(yaw=np.pi))


def test_closed_route_wrap_and_duplicate_segments_rejected():
    teacher = GeometricRecoveryTeacher(LINE, -.1)
    np.testing.assert_allclose(teacher.point_at(teacher.arc[-1]+8), teacher.point_at(8))
    with pytest.raises(ValueError):
        GeometricRecoveryTeacher(np.r_[LINE, LINE[:1]], -.1)


class Device:
    owned = False

    def open(self):
        self.owned = True
        return self.submit(0, 0, 1)

    def submit(self, *controls):
        return dict(controls=controls)

    def close(self):
        self.owned = False
        return self.submit(0, 0, 1)


def reading(gap=5.):
    return SimpleNamespace(state=state(y=-gap, gap=gap), observation=np.zeros(125),
                           sdk_sequence=1, native_packet_id=1)


def test_recovery_envelope_is_explicit_and_shrinks_with_gap_then_closes():
    device, events = Device(), []
    rows = iter([reading()] * 5 + [reading(2.), reading(1.)])
    result = run_geometric_recovery(device, lambda: next(rows),
        GeometricRecoveryTeacher(LINE, -.1), lambda: None, events.append,
        decisions=2, sleep=lambda _: None)
    assert events[0]['initial_gap_m'] == 5 and events[0]['gap_limit_m'] == 6.5
    feedback = [r for r in events if r['kind'] == 'feedback']
    assert feedback[0]['recovery_gap_limit_m'] == 6.5
    assert feedback[1]['recovery_gap_limit_m'] == 3.5
    assert result['decisions'] == 2 and not device.owned
    assert events[-1]['request']['controls'] == (0, 0, 1)


@pytest.mark.parametrize('gap', [6.6, 9.])
def test_excessive_recovery_start_gap_never_acquires(gap):
    device, events = Device(), []
    with pytest.raises(ValueError, match='6.5m'):
        run_geometric_recovery(device, lambda: reading(gap),
            GeometricRecoveryTeacher(LINE, -.1), lambda: None, events.append)
    assert not device.owned and not events


def test_sensor_failure_after_request_still_releases_and_neutrals():
    device, events = Device(), []
    rows = iter([reading()] * 5)
    with pytest.raises(StopIteration):
        run_geometric_recovery(device, lambda: next(rows),
            GeometricRecoveryTeacher(LINE, -.1), lambda: None, events.append,
            decisions=2, sleep=lambda _: None)
    assert events[-2]['kind'] == 'teacher_request'
    assert events[-1]['kind'] == 'close' and not device.owned


def test_30kmh_task_requires_headroom_and_retains_actual_40kmh_stop():
    teacher = GeometricRecoveryTeacher(LINE, -.1, maximum_speed_kmh=30.)
    device, events = Device(), []
    with pytest.raises(ValueError, match='budget'):
        run_geometric_recovery(device, reading, teacher, lambda: None, events.append,
                               decisions=1, sleep=lambda _: None)
    assert not device.owned and not events
    overspeed = reading(0.)
    overspeed.state['speed'] = 41/3.6
    rows = iter([reading(0.)]*5 + [overspeed])
    with pytest.raises(ValueError):
        run_geometric_recovery(device, lambda: next(rows), teacher, lambda: None,
            events.append, decisions=1, hard_speed_kmh=40., sleep=lambda _: None)
    assert events[0]['hard_speed_kmh'] == 40
    assert events[0]['maximum_teacher_speed_kmh'] == 30
    assert events[-1]['kind'] == 'close' and not device.owned


def test_geometric_feedback_read_latency_does_not_accumulate_in_the_40ms_clock():
    clock = SimpleNamespace(now=100.)
    events, stamps, device = [], [], Device()

    def sleep(seconds):
        clock.now += seconds

    def receive():
        sleep(.006)
        return reading(0.)

    def emit(event):
        events.append(event)
        if event['kind'] == 'teacher_request':
            stamps.append(clock.now)

    result = run_geometric_recovery(device, receive, GeometricRecoveryTeacher(LINE, -.1),
        lambda: None, emit, decisions=6, clock=lambda: clock.now, sleep=sleep)
    np.testing.assert_allclose(np.diff(stamps)[1:], .04, atol=1e-12)
    assert result['decisions'] == 6 and not device.owned
