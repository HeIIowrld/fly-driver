"""Host-ordered per-view yaw interval, diagnostic only and not actor qualified."""
from dataclasses import dataclass
import math

from .native_angular_history import NativeAngularHistory
from .virtual_body import VirtualBodyEncoder


@dataclass(frozen=True)
class ViewYawInterval:
    start_render_lower: float
    end_render_lower: float
    camera_receipt: float
    decision: float
    yaw_left_mean_rad_s: float
    yaw_left_integral_rad: float
    first_packet: int
    last_packet: int
    first_receipt: float
    last_receipt: float
    body_contract_sha256: str
    physical_time_verified: bool = False
    actor_qualified: bool = False


def prepare_view_yaw_interval(history, body_encoder, *, previous_render_lower,
                              current_render_lower, camera_receipt, decision):
    """Prepare verified native-yaw average over two host render bounds.

    This does not change a NavigationBodyObservation or imply exposure-time
    synchronization. Call separately for each view, preserving its own bounds.
    """
    clocks = (previous_render_lower, current_render_lower, camera_receipt, decision)
    if (not isinstance(history, NativeAngularHistory)
            or not isinstance(body_encoder, VirtualBodyEncoder)
            or body_encoder.angular_valid[2] is not True
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in clocks)
            or not 0 <= previous_render_lower < current_render_lower <= camera_receipt <= decision
            or current_render_lower - previous_render_lower > .2):
        raise ValueError('Validated yaw mapping and ordered per-view host times required')
    window = history.window(start=previous_render_lower, end=current_render_lower,
                            decision=decision)
    # The encoder has already checked the versioned [2,0,1] axis mapping and
    # its yaw evidence. Keep all other unverified axes out of this result.
    yaw = float(window.native_mean[1])
    return ViewYawInterval(float(previous_render_lower), float(current_render_lower),
                           float(camera_receipt), float(decision), yaw,
                           yaw * (current_render_lower - previous_render_lower),
                           window.first_packet, window.last_packet, window.first_receipt,
                           window.last_receipt, body_encoder.contract_sha256)
