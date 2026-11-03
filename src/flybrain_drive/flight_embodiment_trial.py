"""Separate proof-gated MN embodiment v2 component; legacy trial stays unchanged.

Construction validates actual qualification files before the parent can touch
GPU/native identity. This class has no actuator interface. The caller separately
applies the measured minimum-positive codec and every original plant guard.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actuator_codec import project_pedals
from .flight_embodiment import (
    CANONICAL_V1_SCHEMA,
    CANONICAL_V2_SCHEMA,
    AnchorProvenance,
    EmbodimentReference,
    HashedSource,
    decode_embodiment,
)
from .flight_maneuver_trial import NeuralPhysicalProbe
from .graph import sha256

PROOF_CONTRACT_FILE = "configs/flight-mn-embodiment-component-proof-contract-20261030.json"
GPU_PROOF_SCHEMA = "flybrain.flight.matched_mn_anchor_qualification.v1"
GPU_PROOF_STATUS = "completed_matched_cpu_gpu_mn_anchor_qualification"
GPU_REGISTRATION_SCHEMA = "flybrain.flight.matched_mn_anchor_registration.v1"
READOUT_STATE_SCHEMA = "flybrain.flight.frozen_quiet_readout_state.v1"
REQUIRED_ARTIFACTS = (
    "cpu_reference",
    "cpu_readout_state",
    "matched_reference",
    "cpu_repeat_0",
    "cpu_repeat_1",
    "gpu_repeat_0",
    "gpu_repeat_1",
    "qualification_manifest",
    "registration",
)
REQUIRED_MODEL_SOURCES = (
    "src/flybrain_drive/flight_embodiment.py",
    "src/flybrain_drive/flight_circuit.py",
    "src/flybrain_drive/flight_commands.py",
    "src/flybrain_drive/flight_maneuver.py",
    "src/flybrain_drive/flight_maneuver_trial.py",
    "src/flybrain_drive/vehicle_circuit.py",
    "src/flybrain_drive/flight_actuator.py",
    "src/flybrain_drive/actuator_codec.py",
)
NEUTRAL_KEYS = frozenset(
    {
        "readout.power_reference",
        "readout.lateral_reference",
        "readout.reference_calibrated",
    }
)


def _json_bound(path: str | Path, expected: str) -> Any:
    """Parse exactly the bytes whose caller-supplied SHA is checked."""
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(c not in "0123456789abcdef" for c in expected)
    ):
        raise ValueError("Explicit lowercase artifact SHA256 required")
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Pinned v2 qualification artifact changed: " + str(path))
    return json.loads(content)


def _reference(value: dict) -> EmbodimentReference:
    provenance = value["provenance"]
    return EmbodimentReference(
        **{
            **value,
            "quiet_raw_neural_intent": tuple(value["quiet_raw_neural_intent"]),
            "provenance": AnchorProvenance(
                **{
                    **provenance,
                    "sources": tuple(HashedSource(**source) for source in provenance["sources"]),
                }
            ),
        }
    )


def _same_path(first: str | Path, second: str | Path) -> bool:
    return Path(first).resolve() == Path(second).resolve()


def _finite_nonnegative(value: Any, maximum: float | None = None) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
        and (maximum is None or value <= maximum)
    )


def _fingerprint(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()
    ).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu()
    if value.layout == torch.sparse_coo:
        return _tensor_digest(value._indices()) + _tensor_digest(value._values())
    if value.layout == torch.sparse_csr:
        return (
            _tensor_digest(value.crow_indices())
            + _tensor_digest(value.col_indices())
            + _tensor_digest(value.values())
        )
    return _fingerprint(value.numpy())


def _backbone_digest(backbone) -> dict[str, str]:
    return {key: _tensor_digest(value) for key, value in backbone.state_dict().items()}


@dataclass(frozen=True)
class QualifiedEmbodimentInputs:
    reference: EmbodimentReference
    cpu_reference: EmbodimentReference
    readout_state: dict
    proof: dict
    source_sha256: dict[str, str]
    pinned_artifacts: tuple[tuple[str, str], ...]
    cpu_reference_sha256: str
    gpu_qualification_sha256: str
    readout_state_sha256: str

    def recheck_files(self):
        """Catch mutation during model initialization, before caller acquisition."""
        for path, expected in (*self.pinned_artifacts, *self.source_sha256.items()):
            if sha256(path) != expected:
                raise ValueError(
                    "Qualified v2 source/artifact changed during initialization: " + path
                )


def load_qualified_embodiment_inputs(
    spec: str,
    checkpoint: str,
    *,
    cpu_reference_file: str,
    cpu_reference_sha256: str,
    gpu_qualification_file: str,
    gpu_qualification_sha256: str,
    readout_state_file: str,
    readout_state_sha256: str,
) -> QualifiedEmbodimentInputs:
    """Verify real files, status, measured gates and source/manifest links, no GPU/MMF.

    A reference dataclass or well-formed SHA alone is never qualification. Four
    distinct repeat trace artifacts, registration and its successful manifest
    are required. Full graph recomputation is owned by the pinned qualifier.
    """
    proof = _json_bound(gpu_qualification_file, gpu_qualification_sha256)
    cpu_data = _json_bound(cpu_reference_file, cpu_reference_sha256)
    readout_data = _json_bound(readout_state_file, readout_state_sha256)
    if not all(isinstance(value, dict) for value in (proof, cpu_data, readout_data)):
        raise ValueError("Qualification/reference/readout JSON objects required")
    contract_hash = sha256(PROOF_CONTRACT_FILE)
    if (
        proof.get("schema") != GPU_PROOF_SCHEMA
        or proof.get("status") != GPU_PROOF_STATUS
        or proof.get("numerical_scope") != "matched_cpu_gpu"
        or proof.get("gpu_qualified") is not True
        or proof.get("gpu_used") is not True
        or proof.get("fit_allowed") is not False
        or proof.get("teacher_labels_used") is not False
        or proof.get("optimizer_updates") != 0
        or proof.get("live_api_calls") != 0
        or proof.get("mmf_opens") != 0
        or proof.get("control_messages") != 0
        or proof.get("intermediate_state_resets") != 0
        or not _same_path(proof.get("contract_file", ""), PROOF_CONTRACT_FILE)
        or proof.get("contract_sha256") != contract_hash
        or proof.get("cases") != 30
        or proof.get("observations_per_case") != 68
        or proof.get("original_intervals_per_case") != 67
        or any(
            proof.get(key) is not True
            for key in (
                "cpu_repeat_bitwise_exact",
                "gpu_repeat_passed",
                "all_fixed_parameters_unchanged",
                "cpu_private_readout_installed_on_gpu",
                "all_original_clocks_and_synthetic_masks_preserved",
                "quiet_encoded_pedals_zero",
            )
        )
    ):
        raise ValueError("Actual matched CPU/GPU original-clock qualification is required")
    for key in (
        "optimizer_updates",
        "live_api_calls",
        "mmf_opens",
        "control_messages",
        "intermediate_state_resets",
    ):
        if type(proof.get(key)) is not int or proof[key] != 0:
            raise ValueError("Actual zero integer qualification counter required: " + key)
    for key, count in (
        ("cases", 30),
        ("observations_per_case", 68),
        ("original_intervals_per_case", 67),
    ):
        if type(proof.get(key)) is not int or proof[key] != count:
            raise ValueError("Actual complete qualification count required: " + key)
    for key, limit in (
        ("maximum_raw_parity_error", 2e-6),
        ("maximum_state_parity_error", 2e-6),
        ("maximum_decoded_brake_parity_error", 1e-4),
        ("maximum_common_power_numerical_uncertainty", None),
        ("quiet_raw_max_abs", 2e-6),
        ("quiet_v2_brake_max", 1e-4),
    ):
        if not _finite_nonnegative(proof.get(key), limit):
            raise ValueError("Measured matched qualification gate failed: " + key)
    artifacts = proof.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_ARTIFACTS):
        raise ValueError("Every CPU/GPU repeat and qualification artifact must be pinned")
    pinned = [
        (str(gpu_qualification_file), gpu_qualification_sha256),
        (PROOF_CONTRACT_FILE, contract_hash),
    ]
    for name, record in artifacts.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "bytes"}
            or not isinstance(record["path"], str)
            or not record["path"]
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
        ):
            raise ValueError("Exact artifact path/size/SHA record required: " + name)
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Matched qualifier artifact bytes changed: " + name)
        pinned.append((record["path"], record["sha256"]))
    for name, path, digest in (
        ("cpu_reference", cpu_reference_file, cpu_reference_sha256),
        ("cpu_readout_state", readout_state_file, readout_state_sha256),
    ):
        if not _same_path(artifacts[name]["path"], path) or artifacts[name]["sha256"] != digest:
            raise ValueError("Caller artifacts differ from actual GPU qualification: " + name)
    traces = [
        Path(artifacts[name]["path"]).resolve()
        for name in ("cpu_repeat_0", "cpu_repeat_1", "gpu_repeat_0", "gpu_repeat_1")
    ]
    if len(set(traces)) != 4:
        raise ValueError("Four distinct CPU/GPU repeat trace files required")
    matched_data = _json_bound(
        artifacts["matched_reference"]["path"], artifacts["matched_reference"]["sha256"]
    )
    cpu, matched = _reference(cpu_data), _reference(matched_data)
    if (
        cpu.provenance.numerical_scope != "cpu_repeat"
        or matched.provenance.numerical_scope != "matched_cpu_gpu"
        or cpu.provenance.tensor_dtype != "float32"
        or matched.provenance.tensor_dtype != "float32"
        or any(
            getattr(cpu, key) != getattr(matched, key)
            for key in (
                "quiet_power",
                "brake_anchor_power",
                "original_readout_power_reference",
                "quiet_raw_neural_intent",
            )
        )
        or matched.provenance.original_clock_sha256 != cpu.provenance.original_clock_sha256
        or matched.provenance.quiet_end_state_sha256 != cpu.provenance.quiet_end_state_sha256
        or matched.provenance.anchor_state_sha256 != cpu.provenance.anchor_state_sha256
        or matched.provenance.anchor_elapsed_seconds != cpu.provenance.anchor_elapsed_seconds
        or matched.provenance.anchor_elapsed_seconds != 3.6801013999502175
        or matched.common_power_numerical_uncertainty
        != proof["maximum_common_power_numerical_uncertainty"]
        or matched.maximum_decoded_brake_parity_error != proof["maximum_decoded_brake_parity_error"]
    ):
        raise ValueError(
            "Qualified reference must preserve the registered CPU endpoint and neutral"
        )
    for reference in (cpu, matched):
        for source in reference.provenance.sources:
            if sha256(source.path) != source.sha256:
                raise ValueError("Reference source changed: " + source.role)
    roles = {source.role: source for source in matched.provenance.sources}
    for role, path in (
        ("graph_spec", spec),
        ("graph_checkpoint", checkpoint),
        ("decoder_source", "src/flybrain_drive/flight_embodiment.py"),
        ("readout_source", "src/flybrain_drive/flight_circuit.py"),
    ):
        if not _same_path(roles[role].path, path) or roles[role].sha256 != sha256(path):
            raise ValueError("Component source differs from qualified original: " + role)
    sources = proof.get("current_source_sha256")
    required_sources = (*REQUIRED_MODEL_SOURCES, spec, checkpoint)
    if not isinstance(sources, dict) or any(path not in sources for path in required_sources):
        raise ValueError("Actual graph/spec/checkpoint/decoder/readout sources must be pinned")
    for path, expected in sources.items():
        if sha256(path) != expected:
            raise ValueError("Qualified model source changed: " + path)
    program = proof.get("qualification_program", {})
    if (
        not isinstance(program, dict)
        or set(program) != {"path", "sha256"}
        or not isinstance(program["path"], str)
        or sha256(program["path"]) != program["sha256"]
    ):
        raise ValueError("Actual qualification program source must be verified")
    pinned.append((program["path"], program["sha256"]))
    registration = _json_bound(
        artifacts["registration"]["path"], artifacts["registration"]["sha256"]
    )
    manifest = _json_bound(
        artifacts["qualification_manifest"]["path"], artifacts["qualification_manifest"]["sha256"]
    )
    if (
        registration.get("schema") != GPU_REGISTRATION_SCHEMA
        or registration.get("fit_allowed") is not False
        or type(registration.get("optimizer_updates")) is not int
        or registration["optimizer_updates"] != 0
        or registration.get("proof_contract_sha256") != contract_hash
        or manifest.get("config", {}).get("plan") != registration
        or manifest["config"].get("program_sha256") != program["sha256"]
    ):
        raise ValueError(
            "Matched GPU result must link its original registration and program manifest"
        )
    for path in required_sources:
        if registration.get("input_sha256", {}).get(path) != sources[path]:
            raise ValueError("Qualification registration did not pin model source: " + path)
    synthetic_mask = np.zeros((68, 21), dtype=bool)
    synthetic_mask[:, [0, 1, 4, 16]] = True
    if proof.get("historical_clock_sha256") != cpu.provenance.original_clock_sha256 or proof.get(
        "synthetic_mask_sha256"
    ) != _fingerprint(synthetic_mask):
        raise ValueError("Exact original clock and explicit synthetic mask qualification required")
    if (
        readout_data.get("schema") != READOUT_STATE_SCHEMA
        or readout_data.get("cpu_reference_sha256") != cpu_reference_sha256
        or readout_data.get("source_checkpoint_sha256") != sha256(checkpoint)
        or not isinstance(readout_data.get("readout_state"), dict)
        or readout_data["readout_state"].get("power_reference") != cpu.quiet_power
        or readout_data["readout_state"].get("reference_calibrated") is not True
    ):
        raise ValueError("Qualified exact CPU private readout state required")
    return QualifiedEmbodimentInputs(
        matched,
        cpu,
        readout_data["readout_state"],
        proof,
        sources,
        tuple(pinned),
        cpu_reference_sha256,
        gpu_qualification_sha256,
        readout_state_sha256,
    )


class NeuralEmbodimentV2Probe(NeuralPhysicalProbe):
    """Original GPU state/body/held clock, followed by separate pure v2 correspondence."""

    def __init__(
        self,
        spec,
        checkpoint,
        *,
        episode_id,
        cpu_reference_file,
        cpu_reference_sha256,
        gpu_qualification_file,
        gpu_qualification_sha256,
        readout_state_file,
        readout_state_sha256,
    ):
        qualified = load_qualified_embodiment_inputs(
            spec,
            checkpoint,
            cpu_reference_file=cpu_reference_file,
            cpu_reference_sha256=cpu_reference_sha256,
            gpu_qualification_file=gpu_qualification_file,
            gpu_qualification_sha256=gpu_qualification_sha256,
            readout_state_file=readout_state_file,
            readout_state_sha256=readout_state_sha256,
        )
        # Parent quiet1s happens exactly once. No state is reinitialized after it.
        super().__init__(spec, checkpoint, episode_id=episode_id)
        qualified.recheck_files()
        readout = self.model.backbone.readout
        actual = readout.state_dict()
        if set(actual) != set(qualified.readout_state):
            raise ValueError("Qualified readout state fields differ from actual original model")
        for key, value in actual.items():
            if "readout." + key not in NEUTRAL_KEYS:
                if value.detach().cpu().tolist() != qualified.readout_state[key]:
                    raise ValueError("Qualified readout gains/mix/ports differ: " + key)
        before, state_before = _backbone_digest(self.model.backbone), _tensor_digest(self.state)
        with torch.no_grad():
            for key in ("power_reference", "lateral_reference", "reference_calibrated"):
                value = actual[key]
                selected = torch.as_tensor(
                    qualified.readout_state[key], dtype=value.dtype, device=value.device
                )
                if selected.shape != value.shape:
                    raise ValueError("Qualified neutral reference shape differs: " + key)
                value.copy_(selected)
        after = _backbone_digest(self.model.backbone)
        changed = {key for key in before if before[key] != after[key]}
        if not changed <= NEUTRAL_KEYS or _tensor_digest(self.state) != state_before:
            raise ValueError(
                "Only declared neutral offsets may change; quiet state must stay exact"
            )
        self.reference = {
            key: value.detach().cpu().tolist() for key, value in readout.state_dict().items()
        }
        if self.reference != qualified.readout_state:
            raise ValueError("Actual installed private readout differs from qualified CPU values")
        self.embodiment_inputs = qualified
        raw, muscles = self.model.backbone._read_state_tensor(self.state)
        raw_values, muscle_values = (
            raw[0].detach().cpu().tolist(),
            muscles[0].detach().cpu().tolist(),
        )
        output = self._decode(raw_values, muscle_values)
        v2 = output.canonical_v2[0].numpy()
        encoded = project_pedals(v2)["encoded_request"]
        if (
            max(abs(v) for v in raw_values) > 2e-6
            or float(v2[2]) > 1e-4
            or encoded[1:] != [0.0, 0.0]
            or abs(encoded[0]) > 2e-6
        ):
            raise ValueError(
                "Actual GPU quiet raw/v2/encoded neutral gate failed before acquisition"
            )
        self.embodiment_initialization = dict(
            schema="flybrain.flight.embodiment_v2_component_initialization.v1",
            numerical_scope="matched_cpu_gpu",
            cpu_reference_sha256=cpu_reference_sha256,
            gpu_qualification_sha256=gpu_qualification_sha256,
            readout_state_sha256=readout_state_sha256,
            actual_quiet_raw_intent=raw_values,
            actual_quiet_muscles=muscle_values,
            canonical_v1=output.canonical_v1[0].tolist(),
            canonical_v2=v2.tolist(),
            encoded_neutral_check=encoded,
            encoded_pedals_exact_zero=True,
            steering_unchanged_within_raw_tolerance=True,
            changed_neutral_keys=sorted(changed),
            all_other_backbone_values_unchanged=True,
            quiet_state_sha256=state_before,
            quiet_state_preserved=True,
            numerical_state_reset=False,
            quiet_steps=100,
            quiet_step_seconds=0.01,
            per_case_gpu_rebase=False,
            actuator_acquisition_performed=False,
        )
        qualified.recheck_files()

    def _decode(self, raw_values, muscle_values):
        # JSON metadata originates from the parent's same GPU state, float32 exact.
        # Pure CPU decode preserves that recorded sample and adds no GPU/state step.
        return decode_embodiment(
            torch.tensor([raw_values], dtype=torch.float32, device="cpu"),
            torch.tensor([muscle_values], dtype=torch.float32, device="cpu"),
            reference=self.embodiment_inputs.reference,
        )

    @torch.no_grad()
    def request(self, sample, physical_command, decision_clock):
        # The original advance/body/previous-held-time implementation is called ONCE.
        legacy_control, metadata = super().request(sample, physical_command, decision_clock)
        output = self._decode(metadata["intent"], metadata["muscles"])
        v1, v2 = output.canonical_v1[0].numpy(), output.canonical_v2[0].numpy()
        if not np.array_equal(v1, np.asarray(legacy_control)) or not np.array_equal(
            v1, np.asarray(metadata["canonical_request"])
        ):
            raise ValueError("Parent raw intent/MN canonical provenance differs")
        value = dict(metadata)
        value.update(
            canonical_v1_request=list(metadata["canonical_request"]),
            canonical_v1_request_schema=CANONICAL_V1_SCHEMA,
            canonical_request=v2.tolist(),
            canonical_request_schema=CANONICAL_V2_SCHEMA,
            embodiment_v2=dict(
                schema="flybrain.flight.embodiment_v2_component_sample.v1",
                common_power=float(output.common_power[0]),
                span=float(output.span),
                cpu_reference_sha256=self.embodiment_inputs.cpu_reference_sha256,
                gpu_qualification_sha256=self.embodiment_inputs.gpu_qualification_sha256,
                readout_state_sha256=self.embodiment_inputs.readout_state_sha256,
                numerical_scope="matched_cpu_gpu",
                decoder_device="cpu",
                pedal_coactivation=bool(output.pedal_coactivation[0]),
                encoded_request_not_applied_here=True,
                same_parent_state_sample=True,
                numerical_state_reset=False,
            ),
        )
        return v2, value
