"""Offline, versioned MN-to-vehicle correspondence; no live interfaces.

A structurally valid reference is a declaration, not proof that its source files
were verified. A separate qualifier must replay the pinned graph and registration.
This module neither changes the graph/readout nor certifies braking or driving.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import torch

REFERENCE_SCHEMA = "flybrain.flight.mn_embodiment_reference.v2"
CANONICAL_V1_SCHEMA = "flybrain.flight.signed_intent_canonical.v1"
CANONICAL_V2_SCHEMA = "flybrain.flight.reachable_power_brake_canonical.v2"
MINIMUM_ABSOLUTE_SPAN = 1e-6
MINIMUM_RELATIVE_SPAN = 0.05
MINIMUM_SPAN_TO_UNCERTAINTY = 100.0
MAXIMUM_BRAKE_PARITY_ERROR = 1e-4
SOURCE_ROLES = frozenset(
    {
        "graph_spec",
        "graph_checkpoint",
        "readout_source",
        "decoder_source",
        "registration",
        "trajectory",
    }
)


@dataclass(frozen=True)
class EmbodimentContract:
    schema: str = "flybrain.flight.mn_to_vehicle_embodiment.v2"
    canonical_v1: str = CANONICAL_V1_SCHEMA
    canonical_v2: str = CANONICAL_V2_SCHEMA
    motor_summary_order: tuple[str, ...] = ("b1L", "b1R", "b2L", "b2R", "powerL", "powerR")
    common_power: str = "p=0.5*(powerL+powerR), same original readout state"
    brake: str = "clamp((p0-p)/(p0-p_brake),0,1)"
    unchanged_outputs: str = "raw neural intent, legacy canonical, steering and positive throttle"
    clock_and_state: str = "stateless; no integration, reset, retiming or sensor sample creation"
    actuator_stages: str = "canonical_v1 != canonical_v2 != encoded request != native applied"
    limitation: str = (
        "One common-power coordinate does not establish independent pedal coactivation. "
        "On the original same-p0 signed-forward manifold throttle and brake are exclusive; "
        "coactive teacher targets retain irreducible joint imitation loss. "
        "This mapping establishes no reachable policy, vehicle braking, "
        "learned driving or valid laps."
    )


EMBODIMENT_CONTRACT = EmbodimentContract()


def _finite(value: Real) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _sha(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class HashedSource:
    role: str
    path: str
    sha256: str

    def __post_init__(self):
        if (
            self.role not in SOURCE_ROLES
            or not isinstance(self.path, str)
            or not self.path.strip()
            or not _sha(self.sha256)
        ):
            raise ValueError("Explicit named source reference and SHA256 required")


@dataclass(frozen=True)
class AnchorProvenance:
    """Immutable evidence declarations; this class does not read or qualify files.

    Registration must fix negative-step amplitude, anchor endpoint, span/parity
    gates, dtype and original integration protocol before observing the trajectory
    or optimizing. The endpoint is not the minimum over times, steps or runs.
    Numerical uncertainty must come from matched original-trajectory replays;
    CPU-repeat evidence must never be described as CPU/GPU parity.
    """

    sources: tuple[HashedSource, ...]
    tensor_dtype: Literal["float32", "float64"]
    numerical_scope: Literal["cpu_repeat", "matched_cpu_gpu"]
    original_clock_sha256: str
    original_body_mask_sha256: str
    quiet_end_state_sha256: str
    anchor_state_sha256: str
    quiet_duration_seconds: float
    step_speed_error_m_s: float
    anchor_elapsed_seconds: float
    fixed_time_registered_before_trajectory: bool
    quiet_to_step_state_continuous: bool
    original_graph_body_readout_frozen: bool
    all_original_samples_retained: bool
    anchor_power_observed: bool
    optimizer_updates_before_anchor: int
    teacher_labels_used: bool

    def __post_init__(self):
        if (
            type(self.sources) is not tuple
            or not all(isinstance(s, HashedSource) for s in self.sources)
            or len(self.sources) != len(SOURCE_ROLES)
            or {s.role for s in self.sources} != SOURCE_ROLES
            or self.tensor_dtype not in ("float32", "float64")
            or self.numerical_scope not in ("cpu_repeat", "matched_cpu_gpu")
            or not all(
                _sha(v)
                for v in (
                    self.original_clock_sha256,
                    self.original_body_mask_sha256,
                    self.quiet_end_state_sha256,
                    self.anchor_state_sha256,
                )
            )
            or not _finite(self.quiet_duration_seconds)
            or not math.isclose(self.quiet_duration_seconds, 1.0, rel_tol=0.0, abs_tol=1e-12)
            or not _finite(self.step_speed_error_m_s)
            or self.step_speed_error_m_s >= 0
            or not _finite(self.anchor_elapsed_seconds)
            or self.anchor_elapsed_seconds <= 0
            or any(
                v is not True
                for v in (
                    self.fixed_time_registered_before_trajectory,
                    self.quiet_to_step_state_continuous,
                    self.original_graph_body_readout_frozen,
                    self.all_original_samples_retained,
                    self.anchor_power_observed,
                )
            )
            or type(self.optimizer_updates_before_anchor) is not int
            or self.optimizer_updates_before_anchor != 0
            or self.teacher_labels_used is not False
        ):
            raise ValueError(
                "Pinned original quiet1s, registered negative-step endpoint "
                "and continuous frozen trajectory required"
            )


@dataclass(frozen=True)
class EmbodimentReference:
    schema: str
    quiet_power: float
    brake_anchor_power: float
    original_readout_power_reference: float
    quiet_raw_neural_intent: tuple[float, float]
    common_power_numerical_uncertainty: float
    maximum_decoded_brake_parity_error: float
    quiet_decoded_brake_error: float
    provenance: AnchorProvenance

    def __post_init__(self):
        if (
            self.schema != REFERENCE_SCHEMA
            or not isinstance(self.provenance, AnchorProvenance)
            or not all(
                _finite(v)
                for v in (
                    self.quiet_power,
                    self.brake_anchor_power,
                    self.original_readout_power_reference,
                    self.common_power_numerical_uncertainty,
                    self.maximum_decoded_brake_parity_error,
                    self.quiet_decoded_brake_error,
                )
            )
            or not 0 <= self.brake_anchor_power < self.quiet_power <= 1
            or self.original_readout_power_reference != self.quiet_power
            or type(self.quiet_raw_neural_intent) is not tuple
            or not all(_finite(v) for v in self.quiet_raw_neural_intent)
            or self.quiet_raw_neural_intent != (0.0, 0.0)
            or self.common_power_numerical_uncertainty < 0
            or not 0 <= self.maximum_decoded_brake_parity_error <= MAXIMUM_BRAKE_PARITY_ERROR
            or self.quiet_decoded_brake_error != 0
        ):
            raise ValueError(
                "Explicit observed anchor, same-state neutral "
                "and measured parity declarations required"
            )
        span = self.quiet_power - self.brake_anchor_power
        if (
            span < MINIMUM_ABSOLUTE_SPAN
            or span < MINIMUM_RELATIVE_SPAN * self.quiet_power
            or span < MINIMUM_SPAN_TO_UNCERTAINTY * self.common_power_numerical_uncertainty
        ):
            raise ValueError("Brake anchor has weak absolute, relative or numerical power span")

    @property
    def span(self) -> float:
        return self.quiet_power - self.brake_anchor_power


@dataclass(frozen=True)
class EmbodimentOutput:
    raw_neural_intent: torch.Tensor
    common_power: torch.Tensor
    span: torch.Tensor
    canonical_v1: torch.Tensor
    canonical_v2: torch.Tensor
    pedal_coactivation: torch.Tensor
    reference: EmbodimentReference
    contract: EmbodimentContract = EMBODIMENT_CONTRACT
    encoded_request: None = None
    native_applied: None = None


def decode_embodiment(
    raw_neural_intent: torch.Tensor, motor_summary: torch.Tensor, *, reference: EmbodimentReference
) -> EmbodimentOutput:
    """Pure tensor correspondence for matched original intent/MN state samples.

    Uses ordinary clamp derivatives (zero outside the unsaturated interior), never
    an STE. There are no goals, errors, teacher labels, PID terms, clock inputs or
    actuator residuals. Coactivation is reported without suppressing any sample,
    body mask or loss. Encoding and applied control require separate evidence.
    """
    if not isinstance(reference, EmbodimentReference):
        raise ValueError("Versioned immutable embodiment reference required")
    dtype = {"float32": torch.float32, "float64": torch.float64}[reference.provenance.tensor_dtype]
    if (
        not isinstance(raw_neural_intent, torch.Tensor)
        or not isinstance(motor_summary, torch.Tensor)
        or raw_neural_intent.ndim < 2
        or raw_neural_intent.shape[-1] != 2
        or motor_summary.shape != (*raw_neural_intent.shape[:-1], 6)
        or raw_neural_intent.numel() == 0
        or raw_neural_intent.dtype != dtype
        or motor_summary.dtype != dtype
        or raw_neural_intent.device != motor_summary.device
        or raw_neural_intent.device.type not in ("cpu", "cuda")
        or not torch.isfinite(raw_neural_intent).all()
        or not torch.isfinite(motor_summary).all()
        or torch.any(raw_neural_intent.abs() > 1)
        or torch.any((motor_summary < 0) | (motor_summary > 1))
    ):
        raise ValueError(
            "Matched same-dtype finite bounded original intent and MN6 summary required"
        )
    turn, forward = raw_neural_intent.unbind(-1)
    canonical_v1 = torch.stack((-turn, forward.clamp_min(0), (-forward).clamp_min(0)), -1)
    power = 0.5 * (motor_summary[..., 4] + motor_summary[..., 5])
    p0 = power.new_tensor(reference.quiet_power)
    anchor = power.new_tensor(reference.brake_anchor_power)
    brake = ((p0 - power) / (p0 - anchor)).clamp(0, 1)
    canonical_v2 = torch.stack((canonical_v1[..., 0], canonical_v1[..., 1], brake), -1)
    return EmbodimentOutput(
        raw_neural_intent,
        power,
        p0 - anchor,
        canonical_v1,
        canonical_v2,
        (canonical_v2[..., 1] > 0) & (brake > 0),
        reference,
    )
