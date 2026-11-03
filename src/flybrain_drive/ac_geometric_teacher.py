"""Privileged recovery/curriculum teacher; no fly actor or anatomical control claim."""

import time

import numpy as np

from .ac_sac_teacher import legacy_request_to_normalized_controls
from .ac_teacher_pilot import check_pilot_state


class GeometricRecoveryTeacher:
    def __init__(self, line, curvature_per_canonical_steering, *, maximum_speed_kmh=20.):
        self.line = np.asarray(line, dtype=float)
        self.vector = np.roll(self.line, -1, axis=0) - self.line
        self.length = np.linalg.norm(self.vector, axis=1)
        if (self.line.ndim != 2 or self.line.shape[1] != 2 or len(self.line) < 6
                or not np.isfinite(self.line).all() or np.any(self.length < 1e-5)
                or not np.isfinite([curvature_per_canonical_steering, maximum_speed_kmh]).all()
                or not -.5 <= curvature_per_canonical_steering <= -.005
                or not 5 <= maximum_speed_kmh <= 30):
            raise ValueError('Closed route, measured signed steering gain and low speed required')
        self.arc = np.r_[0., np.cumsum(self.length)]
        self.gain = curvature_per_canonical_steering
        self.maximum_speed_kmh = maximum_speed_kmh
        self.last_diagnostics = None

    def point_at(self, distance):
        s = distance % self.arc[-1]
        index = min(int(np.searchsorted(self.arc, s, side='right')-1), len(self.line)-1)
        return self.line[index] + self.vector[index]*(s-self.arc[index])/self.length[index]

    def request(self, state):
        point = np.array([state['world_position_x'], state['world_position_y']])
        if not np.isfinite([*point, state['speed'], state['yaw'], state['gap']]).all():
            raise ValueError('Finite privileged state required by geometric teacher')
        projection = np.clip(np.sum((point-self.line)*self.vector, axis=1)/self.length**2, 0, 1)
        distances = np.linalg.norm(self.line + projection[:, None]*self.vector-point, axis=1)
        nearest = int(np.argmin(distances))
        s = self.arc[nearest] + projection[nearest]*self.length[nearest]
        lookahead = 8. + .6*max(state['speed'], 0.)
        target = self.point_at(s+lookahead)
        relative = target-point
        forward = np.array([np.cos(state['yaw']), np.sin(state['yaw'])])
        along = float(forward @ relative)
        cross = float(forward[0]*relative[1] - forward[1]*relative[0])
        if along <= 0 or abs(np.arctan2(cross, along)) > 1.2:
            raise ValueError('Recovery target is behind car or heading error is too large')
        curvature = 2*cross/float(relative @ relative)
        steering = float(np.clip(curvature/self.gain, -.35, .35))
        target_speed = float(np.clip(self.maximum_speed_kmh /
            (1 + .25*abs(state['gap']) + abs(np.arctan2(cross, along))),
            5., self.maximum_speed_kmh))
        speed = max(state['speed']*3.6, 0.)
        throttle = float(np.clip((target_speed-speed)/20., 0., .25))
        brake = float(np.clip((speed-target_speed)/10., 0., .5))
        self.last_diagnostics = dict(route_arc_m=s, target=target.tolist(),
            heading_error_rad=float(np.arctan2(cross, along)),
            desired_curvature=curvature, target_speed_kmh=target_speed,
            measured_curvature_per_canonical_steering=self.gain, actor_input=False)
        return np.array([steering, 2*throttle-1, 2*brake-1])


def run_geometric_recovery(session, receive, teacher, guard, emit, *, decisions=500,
                           steering_request_scale=.6, image_context=None,
                           reset_history=None, clock=time.monotonic, sleep=time.sleep,
                           hard_speed_kmh=30.):
    if (type(decisions) is not int or not 1 <= decisions <= 500
            or not np.isfinite(steering_request_scale) or not 0 < steering_request_scale <= 1
            or not np.isfinite(hard_speed_kmh)
            or not teacher.maximum_speed_kmh+10 <= hard_speed_kmh <= 60):
        raise ValueError('Bounded geometric recovery budget required')
    completed = 0
    try:
        guard()
        current = receive()
        initial_gap = abs(current.state['gap'])
        if initial_gap > 6.5 or min(current.state['sensors'][0], current.state['sensors'][-1]) < 1.:
            raise ValueError('Recovery requires gap<=6.5m and lateral ray clearance>=1m')
        gap_limit = max(3., initial_gap+1.5)
        check_pilot_state(current, max_speed_kmh=hard_speed_kmh,
                          max_gap_m=gap_limit, require_start=True)
        emit(dict(kind='recovery_envelope', initial_gap_m=initial_gap, gap_limit_m=gap_limit,
                  hard_speed_kmh=hard_speed_kmh,
                  maximum_teacher_speed_kmh=teacher.maximum_speed_kmh))
        emit(dict(kind='open', request=session.open()))
        if reset_history is not None:
            reset_history('geometric_teacher_neutral_acquisition')
        for _ in range(4):
            sleep(.04)
            guard()
            current = receive()
            check_pilot_state(current, max_speed_kmh=.5, max_gap_m=gap_limit)
        if (abs(current.state['steerAngle']) > .5 or current.state['accStatus'] > .02
                or current.state['brakeStatus'] < .98):
            raise ValueError('Game did not acknowledge neutral hold')
        cadence_origin = clock()
        deadline = cadence_origin + (decisions+4)*.04*1.5
        for index in range(decisions):
            started = clock()
            guard()
            guarded = clock()
            if started >= deadline:
                raise TimeoutError('Geometric teacher wall-clock budget exceeded')
            check_pilot_state(current, max_speed_kmh=hard_speed_kmh, max_gap_m=gap_limit)
            image = image_context(current) if image_context else None
            imaged = clock()
            request = teacher.request(current.state)
            controls = legacy_request_to_normalized_controls(request)
            controls[0] *= steering_request_scale
            guard()
            prepared = clock()
            receipt = session.submit(*map(float, controls))
            submitted = clock()
            emit(dict(kind='teacher_request', teacher_kind='geometric_recovery', index=index,
                sdk_sequence=current.sdk_sequence, native_packet_id=current.native_packet_id,
                legacy_absolute_request=request.tolist(), request=receipt, image_candidate=image,
                steering_request_scale=steering_request_scale,
                teacher_diagnostics=teacher.last_diagnostics, delta=None,
                observation=current.observation.tolist()
                    if current.observation is not None else None))
            emitted = clock()
            if emitted-started >= .04:
                emit(dict(kind='deadline_overrun', index=index,
                    phase_seconds=dict(first_guard=guarded-started, image=imaged-guarded,
                        teacher_and_guard=prepared-imaged, driver=submitted-prepared,
                        event_emit=emitted-submitted), total_seconds=emitted-started))
                raise TimeoutError('Geometric teacher decision deadline exceeded')
            sleep(max(0., cadence_origin+(index+1)*.04-clock()))
            guard()
            current = receive()
            emit(dict(kind='feedback', index=index, sdk_sequence=current.sdk_sequence,
                native_packet_id=current.native_packet_id, speed_m_s=current.state['speed'],
                gap_m=current.state['gap'], actual_gear=current.state['actualGear'],
                out_of_track=current.state['out_of_track'], recovery_gap_limit_m=gap_limit,
                sdk_steer_raw=current.state['steerAngle'],
                actual_throttle=current.state['accStatus'],
                actual_brake=current.state['brakeStatus']))
            check_pilot_state(current, max_speed_kmh=hard_speed_kmh, max_gap_m=gap_limit)
            if min(current.state['sensors'][0], current.state['sensors'][-1]) < 1.:
                raise ValueError('Recovery lateral ray clearance fell below 1m')
            gap_limit = min(gap_limit, max(3., abs(current.state['gap'])+1.5))
            completed += 1
        return dict(status='segment_completed', decisions=completed,
            teacher_kind='geometric_recovery', final_gap_m=current.state['gap'],
            recovery_returned_within_1m=bool(abs(current.state['gap']) < 1.),
            full_lap_verified=False, fly_actor_driving_verified=False)
    finally:
        if session.owned:
            emit(dict(kind='close', request=session.close(), game_applied_verified=False))
