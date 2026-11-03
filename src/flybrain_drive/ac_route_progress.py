"""Diagnostic SDK course progress, never an actor input or a full-lap certificate."""

import math


class RouteProgress:
    """Unwrap one original session's lap counter and normalized spline position.

    Callers must separately pin engine/session identity and preserve neural state.
    Reject gaps, counter resets and implausible jumps instead of guessing a wrap.
    SDK track length defines the reported metres; it is not a surveyed distance.
    """

    def __init__(self, diagnostic, *, maximum_speed_kmh):
        if (isinstance(maximum_speed_kmh, bool)
                or not isinstance(maximum_speed_kmh, (int, float))
                or not math.isfinite(maximum_speed_kmh) or maximum_speed_kmh <= 0):
            raise ValueError('Finite positive route speed bound required')
        self.maximum_speed_kmh = maximum_speed_kmh
        length, count, position, stamp, x, y = self._values(diagnostic)
        self.length = length
        self.start = self.last = count+position
        self.last_count, self.last_time = count, stamp
        self.start_time, self.last_xy, self.plane_distance_m = stamp, (x, y), 0.
        self.distance_m = 0.

    @staticmethod
    def _values(diagnostic):
        try:
            length, count, position, stamp, x, y = [diagnostic[k] for k in (
                'track_length_m', 'lap_count', 'normalized_spline_position',
                'native_received_monotonic_seconds', 'x', 'y')]
            valid = (all(not isinstance(v, bool) and math.isfinite(v)
                         for v in (length, count, position, stamp, x, y))
                     and length > 0 and count >= 0 and count == int(count)
                     and 0 <= position <= 1 and stamp >= 0)
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError('Finite SDK length, lap counter, spline and host time required')
        return float(length), int(count), float(position), float(stamp), float(x), float(y)

    def observe(self, diagnostic):
        length, count, position, stamp, x, y = self._values(diagnostic)
        elapsed = stamp-self.last_time
        if (not math.isclose(length, self.length, rel_tol=0, abs_tol=.001)
                or not 0 < elapsed <= .25 or not 0 <= count-self.last_count <= 1):
            raise ValueError('Route identity/length, timestamp gap or lap counter changed')
        unwrapped = count+position
        delta = (unwrapped-self.last)*self.length
        # Spline projection can advance faster than the centre of an off-centre
        # car. This explicit tolerance is not an exact physical odometer proof.
        bound = 1.5*self.maximum_speed_kmh/3.6*elapsed+.5
        step = math.hypot(x-self.last_xy[0], y-self.last_xy[1])
        plane_distance = self.plane_distance_m+step
        distance = (unwrapped-self.start)*self.length
        if (abs(delta) > bound or step > bound
                or distance > 1.5*plane_distance+5.
                or plane_distance > 1.5*self.maximum_speed_kmh/3.6*(stamp-self.start_time)+5.):
            raise ValueError('Implausible route jump or unmatched lap wrap')
        self.last, self.last_count, self.last_time = unwrapped, count, stamp
        self.last_xy, self.plane_distance_m = (x, y), plane_distance
        self.distance_m = distance
        return self.distance_m
