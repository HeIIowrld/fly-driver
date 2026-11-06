"""Versioned canonical3 contract for the qualified SAC flight lineage; no I/O."""
from dataclasses import dataclass, asdict
import math
import numpy as np

SCHEMA = "flybrain.current_flight.canonical3.v1"
STEERING_DEVICE_SCALE = .6
FEATURE_SELECTION_MAX_VIEW_AGE = .025
LABEL_SOURCE = "scripts/prepare_synchronized_sac_student.py: labels=[control[steering]/.6,control[throttle],control[brake]]"


def canonical3(value):
    values = np.asarray(value, dtype=float)
    if (values.shape != (3,) or not np.isfinite(values).all()
            or abs(values[0]) > 1 or np.any(values[1:] < 0) or np.any(values[1:] > 1)):
        raise ValueError("Finite bounded canonical steering/throttle/brake required")
    return tuple(map(float, values))


def control_request(value):
    steer, throttle, brake = canonical3(value)
    return dict(schema=SCHEMA, raw_prediction=[steer, throttle, brake],
                steering_device_scale=STEERING_DEVICE_SCALE,
                requested_control=dict(steering=steer*STEERING_DEVICE_SCALE,
                                       throttle=throttle, brake=brake),
                clipping_applied=False, pedal_override=False,
                label_source=LABEL_SOURCE)


@dataclass(frozen=True)
class FlightDecisionEnvelope:
    run_id: str
    episode_id: str
    generation: str
    checkpoint_sha256: str
    decision_id: int
    publication_id: int
    capture_id: int
    body_packet: int
    body_first_receipt: float
    view_render_lowers: tuple
    view_render_uppers: tuple
    view_body_first_copy_starts: tuple
    camera_received: float
    published_at: float
    decision_started: float
    inference_finished: float
    prediction: tuple
    schema: str = "flybrain.current_flight.source_decision.v1"

    def validate(self, *, now, run_id, episode_id, generation, checkpoint_sha256,
                 after_decision=0, after_publication=0, after_capture=-1):
        if (self.schema != "flybrain.current_flight.source_decision.v1"
                or (self.run_id, self.episode_id, self.generation, self.checkpoint_sha256)
                != (run_id, episode_id, generation, checkpoint_sha256)
                or not all(isinstance(v, str) and v for v in (run_id, episode_id, generation))
                or len(checkpoint_sha256) != 64
                or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
                or any(type(v) is not int or v < 0 for v in
                       (self.decision_id, self.publication_id, self.capture_id, self.body_packet))
                or self.decision_id <= after_decision or self.publication_id < max(1, after_publication)
                or self.capture_id < after_capture):
            raise ValueError("Decision identity/sequence differs")
        groups = (self.view_render_lowers, self.view_render_uppers, self.view_body_first_copy_starts)
        if any(not isinstance(v, tuple) or len(v) != 4 for v in groups):
            raise ValueError("Four original view clocks required")
        clocks = (now, self.body_first_receipt, self.camera_received, self.published_at,
                  self.decision_started, self.inference_finished, *sum(groups, ()))
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in clocks):
            raise ValueError("Finite original QPC clocks required")
        if not (self.body_first_receipt <= self.decision_started <= self.inference_finished <= now
                and self.camera_received <= self.published_at <= self.decision_started
                and now-self.body_first_receipt <= .05
                and now-min(self.view_render_lowers) <= .15):
            raise ValueError("Decision body50ms/view150ms freshness/order exceeded")
        for lo, hi, body in zip(*groups, strict=True):
            if not body <= lo <= hi <= self.camera_received or hi-body > .025:
                raise ValueError("Original source-to-render25ms bound exceeded")
        canonical3(self.prediction)
        return min(self.body_first_receipt+.05, min(self.view_render_lowers)+.15)

    def record(self):
        return asdict(self)
