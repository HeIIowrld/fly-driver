"""Experimental longer stationary reference for the existing visual-flight port."""
from .flight_geometry_fresh_hold import FreshEvolvingGeometryPort
from .visual_self_motion import StationaryMotionReference


class LongStationaryMotionReference(StationaryMotionReference):
    """Require one uninterrupted second before sealing the last-three mean.

    This is a timing hypothesis, not proof of sensor or neural stabilization.
    The original native stationary, source identity and freshness checks remain.
    """
    def __init__(self):
        super().__init__(mean_frames=3)

    def observe(self, features, native_sample, *, capture_id, decision_time):
        ready = super().observe(features, native_sample, capture_id=capture_id,
                                decision_time=decision_time)
        if ready and self.rows[-1][0] - self.rows[0][0] < 1.0 - 1e-10:
            self.value = None
            return False
        return ready


class LongStationaryGeometryPort(FreshEvolvingGeometryPort):
    """Use the same geometry weights and body gate with a longer reference window."""
    def reset_episode(self):
        super().reset_episode()
        self.reference = LongStationaryMotionReference()
