"""Inspect CSP cached car values around a render; never certify pixel time."""
import math
from .diagnostic_sim_clock_pairs import summarize_clock_pairs


def summarize_cached_body_around_render(frame):
    clocks = summarize_clock_pairs(frame)
    rows = []
    for view in frame['views']:
        pair = [view['diagnostic_clock_pair_before'], view['diagnostic_clock_pair_after']]
        body = [x.get('body_diagnostic') for x in pair]
        for value in body:
            if not isinstance(value, dict):
                raise ValueError('Both original diagnostic car state reads are required')
            speed = value.get('speed_kmh')
            vectors = [value.get('local_angular_velocity'), value.get('acceleration_g')]
            if (type(speed) not in (float, int) or not math.isfinite(speed) or speed < 0
                    or any(not isinstance(v, list) or len(v) != 3 for v in vectors)
                    or any(type(x) not in (float, int) or not math.isfinite(x)
                           for v in vectors for x in v)):
                raise ValueError('Finite speed and two three-axis CSP vectors required')
        rows.append({'view_id': view['view_id'],
                     'car_timestamp_before_ms': pair[0]['car_timestamp_ms'],
                     'car_timestamp_after_ms': pair[1]['car_timestamp_ms'],
                     'speed_before_kmh': body[0]['speed_kmh'],
                     'speed_after_kmh': body[1]['speed_kmh'],
                     'speed_difference_kmh': body[1]['speed_kmh'] - body[0]['speed_kmh'],
                     'angular_velocity_difference': [b-a for a, b in zip(
                         body[0]['local_angular_velocity'], body[1]['local_angular_velocity'], strict=True)],
                     'acceleration_difference_g': [b-a for a, b in zip(
                         body[0]['acceleration_g'], body[1]['acceleration_g'], strict=True)]})
    return {'views': rows, 'clock_summary': clocks, 'physical_exposure_sync_verified': False,
            'native_packet_identity_verified': False, 'training_admissible': False,
            'interpretation': 'CSP car cache reads bracket a render call, not a pixel or native-packet timestamp.'}
