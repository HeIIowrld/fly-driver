"""Teacher-only speed progression and conservative full-lap evidence tracking."""
import math

import numpy as np

from .ac_sac_teacher import legacy_request_to_normalized_controls


def speed_request(requested, speed_kmh, target):
    if not math.isfinite(target) or not 10 <= target <= 300:
        raise ValueError('Finite teacher evaluation target 10..300 required')
    if not math.isfinite(speed_kmh):
        raise ValueError('Finite measured speed required')
    controls = legacy_request_to_normalized_controls(requested)
    controls[1] = min(controls[1], float(np.clip((target-speed_kmh)/8, 0, 1)))
    floor = float(np.clip((speed_kmh-target)/15, 0, .4))
    controls[2] = max(controls[2], floor)
    if floor > 0:
        controls[1] = 0
    return np.array([controls[0], 2*controls[1]-1, 2*controls[2]-1])


class EvaluationStateGuard:
    """Qualify at most 300ms of measured automatic-shift neutral.

    This is a separate evaluation contract; stationary acquisition still requires
    a forward gear. Reverse, disabled autoshifter, prolonged neutral and track or
    source failures terminate immediately. The evaluator explicitly chooses
    either a coasting intervention or continued original policy actions.
    """
    def __init__(self, *, max_gap_m=3.):
        if max_gap_m not in (3., 5., 12.):
            raise ValueError('Explicit finite 3/5/12m evaluation gap envelope required')
        self.max_gap_m = max_gap_m
        self.had_forward = False
        self.neutral_since = None

    def check(self, state, native_gear, auto_shifter, now, ceiling):
        values = [state[k] for k in ('speed', 'gap', 'actualGear')]
        if (not np.isfinite(values).all() or abs(state['speed'])*3.6 > ceiling
                or abs(state['gap']) > self.max_gap_m or state['out_of_track']):
            raise ValueError(f'Teacher evaluation envelope exceeded: speed={state["speed"]*3.6:.6f}'
                f'/{ceiling}, gap={state["gap"]:.6f}/{self.max_gap_m}, '
                f'out_of_track={state["out_of_track"]}')
        sdk_gear = state['actualGear']
        if sdk_gear < 1 or native_gear < 1:
            raise ValueError('Reverse gear prohibited')
        neutral = sdk_gear == 1 or native_gear == 1
        if neutral:
            if not self.had_forward or auto_shifter != 1 or state['speed']*3.6 < 10:
                raise ValueError('Unqualified neutral gear')
            if self.neutral_since is None:
                self.neutral_since = now
            if now-self.neutral_since > .3:
                raise ValueError('Automatic shift neutral exceeded 300ms')
            return True
        self.had_forward, self.neutral_since = True, None
        return False


class LapEvidence:
    """Discard the partial first lap; require continuous progress and zero tyre exits.

    Crossing/timing evidence is distinct from AC's authoritative valid-lap flag,
    which this prefix does not expose. Never claim official validity from timing.
    """
    def __init__(self):
        self.previous = None
        self.distance = 0.
        self.crossings = 0
        self.clean = True
        self.laps = []
        self.previous_time = None
        self.pending_finish = None
        self.delayed_counter_finishes = []

    def update(self, progress, timing, tyres_out):
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError('Finite spline progress required')
        if timing.status != 'live':
            raise ValueError('Live graphics required')
        self.clean &= tyres_out == 0
        current_time = getattr(timing, 'current_time_raw', None)
        if current_time is not None and (not math.isfinite(current_time) or current_time < 0):
            raise ValueError('Finite nonnegative native lap timer required')
        if self.previous is not None:
            old_progress, old_lap = self.previous
            delta = progress-old_progress
            crossing = delta < -.5
            if crossing:
                delta += 1
            if not -.002 <= delta <= .02:
                raise ValueError('Progress discontinuity/reverse travel')
            if timing.completed_laps not in (old_lap, old_lap+1):
                raise ValueError('Lap counter reset/skipped')
            # Graphics and SDK are separate snapshots: count only native lap edges,
            # with a nearby spline wrap and integrated travel as independent evidence.
            self.distance += max(delta, 0)
            timer_reset = (current_time is not None and self.previous_time is not None
                and current_time < self.previous_time-1000)
            receipt = getattr(timing, 'received_monotonic_seconds', None)
            if self.pending_finish is not None:
                pending = self.pending_finish
                if (receipt is None or not math.isfinite(receipt)
                        or not 0 <= receipt-pending['receipt'] <= .08
                        or timing.last_time_raw != pending['last_time_ms']
                        or not progress < .03):
                    raise ValueError('Unconfirmed finish counter lag exceeded80ms or original evidence changed')
            if timer_reset and timing.completed_laps == old_lap:
                if not (progress < .03 or old_progress > .97):
                    raise ValueError('Lap timer reset away from initial finish line')
                if self.crossings:
                    # The native graphics fields can publish iLastTime/iCurrentTime
                    # one frame before completedLaps. Keep the real timer edge,
                    # but count nothing until the counter confirms within80ms.
                    if (self.pending_finish is not None or receipt is None
                            or not math.isfinite(receipt) or receipt < 0
                            or not .97 <= self.distance <= 1.03 or timing.last_time_raw <= 0):
                        raise ValueError('Unqualified pending full-lap timer edge')
                    self.pending_finish = dict(receipt=receipt,last_time_ms=timing.last_time_raw)
                else:
                    # Initial partial hotlap timing starts without a lap increment.
                    self.crossings += 1
                    self.distance, self.clean = 0., tyres_out == 0
            elif timing.completed_laps == old_lap+1:
                if not (progress < .03 or old_progress > .97):
                    raise ValueError('Lap counter transition away from finish')
                if self.crossings:
                    if not .97 <= self.distance <= 1.03 or timing.last_time_raw <= 0:
                        raise ValueError('Incomplete lap distance/timing evidence')
                    self.laps.append(dict(lap=timing.completed_laps,
                        lap_time_ms=timing.last_time_raw, clean_all_tyres=self.clean,
                        spline_distance_laps=self.distance,
                        official_validity_verified=False))
                if self.pending_finish is not None:
                    self.delayed_counter_finishes.append(dict(**self.pending_finish,
                        confirmation_receipt=receipt,lag_seconds=receipt-self.pending_finish['receipt']))
                    self.pending_finish=None
                self.crossings += 1
                self.distance, self.clean = 0., tyres_out == 0
        self.previous = (progress, timing.completed_laps)
        self.previous_time = current_time
