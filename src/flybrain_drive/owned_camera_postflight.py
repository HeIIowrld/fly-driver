"""Strict host-side classification of a completed owned camera's shutdown."""
import math


def classify_owned_camera_postflight(state, *, age, generation, request_id, window_handle,
                                    device_status, native_speed, native_gas, native_brake):
    """Return wait or idle; reject changed authority, stale sensing or foreign activity."""
    numeric = (age, native_speed, native_gas, native_brake)
    if (not isinstance(state, dict) or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                           or not math.isfinite(x) for x in numeric)
            or not isinstance(generation, str) or not generation
            or not isinstance(request_id, str) or not request_id
            or not isinstance(window_handle, str) or not window_handle
            or not 0 <= age <= 2 or state.get('generation') != generation
            or state.get('paused') is not True or state.get('ai_controlled') is not False
            or state.get('is_online') is not False or state.get('in_menu') is not False
            or state.get('window_handle') != window_handle
            or state.get('car_id') != 'bmw_z4_gt3' or state.get('track_id') != 'monza'
            or type(device_status) is not int or device_status != 1
            or abs(native_speed) >= .1 or native_gas >= .02 or native_brake <= .98):
        raise ValueError('Non-camera postflight authority or stationary condition failed')
    if state.get('camera_active') is False:
        return 'idle'
    if state.get('camera_active') is True and state.get('camera_request_id') == request_id:
        return 'wait'
    raise ValueError('Unexpected or foreign camera state')


class OwnedCameraPostflightTracker:
    """Require an owned active observation followed by a newer idle heartbeat."""

    def __init__(self):
        self.last_active_clock = None

    def observe(self, state, **kwargs):
        phase = classify_owned_camera_postflight(state, **kwargs)
        clock = state.get('script_clock')
        if isinstance(clock, bool) or not isinstance(clock, (int, float)) or not math.isfinite(clock):
            raise ValueError('Finite producer heartbeat clock required')
        if phase == 'wait':
            if self.last_active_clock is not None and clock < self.last_active_clock:
                raise ValueError('Producer heartbeat clock reversed')
            self.last_active_clock = clock
            return phase
        if self.last_active_clock is None or clock <= self.last_active_clock:
            raise ValueError('Newer idle heartbeat after owned active observation required')
        return phase
