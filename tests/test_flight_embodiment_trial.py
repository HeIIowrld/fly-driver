"""CPU stubs only: proof gating and explicit new-mode wiring, never live interfaces."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_ac_maneuver_plant as probe


def component_plan():
    inputs = dict(
        schema=probe.EMBODIMENT_V2_INPUT_SCHEMA,
        cpu_reference_file="cpu-reference.json",
        cpu_reference_sha256="a" * 64,
        gpu_qualification_file="gpu-proof.json",
        gpu_qualification_sha256="b" * 64,
        readout_state_file="readout.json",
        readout_state_sha256="c" * 64,
    )
    paths = dict.fromkeys(probe.EMBODIMENT_V2_REQUIRED_SOURCES, "d" * 64)
    paths.update({"spec.json": "e" * 64, "checkpoint.pt": "f" * 64})
    paths.update(
        {
            inputs[k]: inputs[k.removesuffix("_file") + "_sha256"]
            for k in inputs
            if k.endswith("_file")
        }
    )
    return dict(
        mode=probe.EMBODIMENT_V2_MODE,
        neural_spec="spec.json",
        neural_checkpoint="checkpoint.pt",
        input_sha256=paths,
        embodiment_v2=inputs,
    )


def test_new_mode_requires_explicit_all_artifact_and_component_source_pins():
    plan = component_plan()
    expected = {k: v for k, v in plan["embodiment_v2"].items() if k != "schema"}
    assert probe.validate_embodiment_v2_plan(plan) == expected
    for name in list(plan["input_sha256"]):
        changed = {
            **plan,
            "input_sha256": {k: v for k, v in plan["input_sha256"].items() if k != name},
        }
        with pytest.raises(ValueError, match="hash|pinned"):
            probe.validate_embodiment_v2_plan(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        dict(schema="legacy"),
        dict(extra=True),
        dict(cpu_reference_sha256=True),
        dict(gpu_qualification_sha256="not-a-hash"),
    ],
)
def test_new_mode_has_no_implicit_or_ambiguous_artifact_contract(mutation):
    plan = component_plan()
    plan["embodiment_v2"].update(mutation)
    with pytest.raises(ValueError):
        probe.validate_embodiment_v2_plan(plan)


@pytest.mark.parametrize(
    "mode",
    ["neural_physical_component", "neural_actuator_component", "native_neural_actuator_component"],
)
def test_legacy_factory_is_exact_original_parent_constructor(monkeypatch, mode):
    from flybrain_drive import flight_maneuver_trial

    calls = []
    sentinel = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(flight_maneuver_trial, "NeuralPhysicalProbe", original)
    plan = dict(mode=mode, neural_spec="original-spec", neural_checkpoint="original-checkpoint")
    assert probe.make_neural_component(plan, episode_id="old-episode") is sentinel
    assert calls == [(("original-spec", "original-checkpoint"), dict(episode_id="old-episode"))]
    assert probe.validate_embodiment_v2_plan(plan) is None


def test_new_factory_forwards_only_explicit_artifact_proofs(monkeypatch):
    calls = []
    sentinel = object()

    def new(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        "flybrain_drive.flight_embodiment_trial",
        SimpleNamespace(NeuralEmbodimentV2Probe=new),
    )
    plan = component_plan()
    assert probe.make_neural_component(plan, episode_id="new-episode") is sentinel
    assert calls == [
        (
            ("spec.json", "checkpoint.pt"),
            dict(episode_id="new-episode", **probe.validate_embodiment_v2_plan(plan)),
        )
    ]


def test_v2_return_link_is_exact_and_versioned_before_codec():
    value = np.array([-0.03, 0.0, 0.8], dtype=np.float32)
    metadata = dict(
        canonical_request_schema=probe.EMBODIMENT_V2_CANONICAL_SCHEMA,
        canonical_request=value.tolist(),
    )
    probe.validate_embodiment_v2_return(value, metadata)
    for change in [
        dict(canonical_request_schema="v1"),
        dict(canonical_request=[-0.03, 0.0, 0.7]),
        dict(canonical_request=[-0.03, 0.0]),
    ]:
        with pytest.raises(ValueError, match="match exactly"):
            probe.validate_embodiment_v2_return(value, {**metadata, **change})


def test_new_mode_keeps_exact_neural_program_codec_and_native_qualification(monkeypatch, tmp_path):
    plan = component_plan()
    plan.update(
        program=probe.NEURAL_PROGRAM,
        actuator_codec=dict(contract=probe.CODEC_CONTRACT, minimum_positive=0.02),
        diagnostic_world_input_to_actor=False,
        native_position_qualification=str(tmp_path / "position-proof.json"),
    )
    (tmp_path / "position-proof.json").write_text(
        json.dumps(
            dict(
                status="completed_stationary_native_position_qualification",
                samples=24,
                maximum_coordinate_error_m=0.0,
            )
        ),
        encoding="utf-8",
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["probe", "--plan", str(path), "--output", "unused"])
    monkeypatch.setattr(probe, "sha256", lambda p: plan["input_sha256"].get(str(p), "sealed-plan"))

    class BeforeAnyRuntime(Exception):
        pass

    def stop_before_runtime(*args, **kwargs):
        raise BeforeAnyRuntime

    monkeypatch.setattr(probe, "begin_run", stop_before_runtime)
    with pytest.raises(BeforeAnyRuntime):
        probe.main()
    plan["program"] = probe.PLANT_PROGRAM
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="Exact bounded"):
        probe.main()
    plan["program"] = probe.NEURAL_PROGRAM
    plan["actuator_codec"]["minimum_positive"] = 0.01
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="Explicit measured"):
        probe.main()


# The remaining tests use deliberately synthetic qualification files and a CPU
# parent stub. They never establish GPU qualification and never touch mappings.
import dataclasses

import torch
from torch import nn

from flybrain_drive import flight_embodiment_trial as trial
from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.flight_circuit import FlightMotorReadout
from flybrain_drive.flight_embodiment import (
    REFERENCE_SCHEMA,
    SOURCE_ROLES,
    AnchorProvenance,
    EmbodimentReference,
    HashedSource,
)
from flybrain_drive.graph import sha256


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        groups = dict(
            b1=dict(L=["0"], R=["1"]), b2=dict(L=["2"], R=["3"]), power=dict(L=["4"], R=["5"])
        )
        self.readout = FlightMotorReadout(
            [str(i) for i in range(6)], groups, power_reference=float(torch.tensor(0.010000001))
        )
        self.register_buffer("fixed_graph", torch.eye(2))
        with torch.no_grad():
            self.readout.reference_calibrated.fill_(True)

    def _read_state_tensor(self, state):
        return self.readout(state)


def _dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


@pytest.fixture
def qualified_files(tmp_path):
    """Structurally complete synthetic files, not actual GPU evidence."""
    spec = _dump(tmp_path / "synthetic-spec.json", dict(synthetic=True))
    checkpoint = tmp_path / "synthetic-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic unit fixture, never loaded as a model")
    source_cpu_reg = _dump(tmp_path / "cpu-reg.json", dict(synthetic=True))
    source_cpu_trace = _dump(tmp_path / "cpu-trace.json", dict(synthetic=True))
    q = float(torch.tensor(0.01, dtype=torch.float32))
    anchor = float(torch.tensor(0.002, dtype=torch.float32))
    roles = dict(
        graph_spec=spec,
        graph_checkpoint=str(checkpoint),
        readout_source="src/flybrain_drive/flight_circuit.py",
        decoder_source="src/flybrain_drive/flight_embodiment.py",
        registration=source_cpu_reg,
        trajectory=source_cpu_trace,
    )
    provenance = AnchorProvenance(
        sources=tuple(HashedSource(k, roles[k], sha256(roles[k])) for k in sorted(SOURCE_ROLES)),
        tensor_dtype="float32",
        numerical_scope="cpu_repeat",
        original_clock_sha256="b" * 64,
        original_body_mask_sha256="c" * 64,
        quiet_end_state_sha256="d" * 64,
        anchor_state_sha256="e" * 64,
        quiet_duration_seconds=1.0,
        step_speed_error_m_s=-100 / 3.6,
        anchor_elapsed_seconds=3.6801013999502175,
        fixed_time_registered_before_trajectory=True,
        quiet_to_step_state_continuous=True,
        original_graph_body_readout_frozen=True,
        all_original_samples_retained=True,
        anchor_power_observed=True,
        optimizer_updates_before_anchor=0,
        teacher_labels_used=False,
    )
    cpu = EmbodimentReference(
        schema=REFERENCE_SCHEMA,
        quiet_power=q,
        brake_anchor_power=anchor,
        original_readout_power_reference=q,
        quiet_raw_neural_intent=(0.0, 0.0),
        common_power_numerical_uncertainty=0.0,
        maximum_decoded_brake_parity_error=0.0,
        quiet_decoded_brake_error=0.0,
        provenance=provenance,
    )
    cpu_path = _dump(tmp_path / "cpu-reference.json", dataclasses.asdict(cpu))
    matched = dataclasses.replace(
        cpu,
        provenance=dataclasses.replace(provenance, numerical_scope="matched_cpu_gpu"),
        common_power_numerical_uncertainty=1e-8,
        maximum_decoded_brake_parity_error=2e-6,
    )
    matched_path = _dump(tmp_path / "matched-reference.json", dataclasses.asdict(matched))
    backbone = StubBackbone()
    readout = {k: v.tolist() for k, v in backbone.readout.state_dict().items()}
    readout["power_reference"] = q
    state_path = _dump(
        tmp_path / "readout.json",
        dict(
            schema=trial.READOUT_STATE_SCHEMA,
            cpu_reference_sha256=sha256(cpu_path),
            source_checkpoint_sha256=sha256(checkpoint),
            readout_state=readout,
        ),
    )
    source_hashes = {p: sha256(p) for p in (*trial.REQUIRED_MODEL_SOURCES, spec, str(checkpoint))}
    registration = dict(
        schema=trial.GPU_REGISTRATION_SCHEMA,
        fit_allowed=False,
        optimizer_updates=0,
        proof_contract_sha256=sha256(trial.PROOF_CONTRACT_FILE),
        input_sha256=source_hashes,
    )
    reg_path = _dump(tmp_path / "registration.json", registration)
    program = _dump(tmp_path / "qualifier.py", dict(synthetic=True))
    manifest_path = _dump(
        tmp_path / "manifest.json",
        dict(config=dict(plan=registration, program_sha256=sha256(program))),
    )
    paths = dict(
        cpu_reference=cpu_path,
        cpu_readout_state=state_path,
        matched_reference=matched_path,
        registration=reg_path,
        qualification_manifest=manifest_path,
    )
    for name in ("cpu_repeat_0", "cpu_repeat_1", "gpu_repeat_0", "gpu_repeat_1"):
        paths[name] = _dump(tmp_path / (name + ".json"), dict(synthetic=True, label=name))
    mask = np.zeros((68, 21), dtype=bool)
    mask[:, [0, 1, 4, 16]] = True
    proof = dict(
        schema=trial.GPU_PROOF_SCHEMA,
        status=trial.GPU_PROOF_STATUS,
        numerical_scope="matched_cpu_gpu",
        gpu_qualified=True,
        gpu_used=True,
        fit_allowed=False,
        teacher_labels_used=False,
        optimizer_updates=0,
        live_api_calls=0,
        mmf_opens=0,
        control_messages=0,
        intermediate_state_resets=0,
        contract_file=trial.PROOF_CONTRACT_FILE,
        contract_sha256=sha256(trial.PROOF_CONTRACT_FILE),
        cases=30,
        observations_per_case=68,
        original_intervals_per_case=67,
        cpu_repeat_bitwise_exact=True,
        gpu_repeat_passed=True,
        all_fixed_parameters_unchanged=True,
        cpu_private_readout_installed_on_gpu=True,
        all_original_clocks_and_synthetic_masks_preserved=True,
        quiet_encoded_pedals_zero=True,
        maximum_raw_parity_error=1e-7,
        maximum_state_parity_error=1e-7,
        maximum_decoded_brake_parity_error=2e-6,
        maximum_common_power_numerical_uncertainty=1e-8,
        quiet_raw_max_abs=1e-7,
        quiet_v2_brake_max=0.0,
        artifacts={
            name: dict(path=path, sha256=sha256(path), bytes=Path(path).stat().st_size)
            for name, path in paths.items()
        },
        current_source_sha256=source_hashes,
        qualification_program=dict(path=program, sha256=sha256(program)),
        historical_clock_sha256=provenance.original_clock_sha256,
        synthetic_mask_sha256=trial._fingerprint(mask),
    )
    proof_path = _dump(tmp_path / "gpu-proof.json", proof)
    kwargs = dict(
        cpu_reference_file=cpu_path,
        cpu_reference_sha256=sha256(cpu_path),
        gpu_qualification_file=proof_path,
        gpu_qualification_sha256=sha256(proof_path),
        readout_state_file=state_path,
        readout_state_sha256=sha256(state_path),
    )
    return SimpleNamespace(
        spec=spec,
        checkpoint=str(checkpoint),
        proof=proof,
        kwargs=kwargs,
        readout=readout,
        tmp=tmp_path,
    )


def _rewrite_proof(files):
    _dump(Path(files.kwargs["gpu_qualification_file"]), files.proof)
    files.kwargs["gpu_qualification_sha256"] = sha256(files.kwargs["gpu_qualification_file"])


def test_loader_checks_actual_artifact_source_status_manifest_and_scope(qualified_files):
    f = qualified_files
    loaded = trial.load_qualified_embodiment_inputs(f.spec, f.checkpoint, **f.kwargs)
    assert loaded.reference.provenance.numerical_scope == "matched_cpu_gpu"
    assert loaded.cpu_reference.provenance.numerical_scope == "cpu_repeat"
    assert loaded.readout_state == f.readout
    loaded.recheck_files()


@pytest.mark.parametrize(
    "mutation",
    [
        dict(status="completed_cpu_only"),
        dict(numerical_scope="cpu_repeat"),
        dict(gpu_qualified=False),
        dict(gpu_used=False),
        dict(optimizer_updates=1),
        dict(optimizer_updates=False),
        dict(cases=29),
        dict(maximum_raw_parity_error=2.01e-6),
        dict(maximum_state_parity_error=2.01e-6),
        dict(maximum_decoded_brake_parity_error=1.01e-4),
        dict(cpu_repeat_bitwise_exact=False),
        dict(cpu_private_readout_installed_on_gpu=False),
        dict(quiet_encoded_pedals_zero=False),
    ],
)
def test_unqualified_proof_rejects_before_parent_gpu_or_mapping(
    monkeypatch, qualified_files, mutation
):
    f = qualified_files
    f.proof.update(mutation)
    _rewrite_proof(f)
    called = []
    monkeypatch.setattr(trial.NeuralPhysicalProbe, "__init__", lambda *a, **k: called.append(True))
    with pytest.raises(ValueError):
        trial.NeuralEmbodimentV2Probe(f.spec, f.checkpoint, episode_id="unit", **f.kwargs)
    assert called == []


def test_proof_rejects_changed_artifact_even_with_valid_sha_syntax(qualified_files):
    f = qualified_files
    Path(f.proof["artifacts"]["gpu_repeat_1"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact bytes changed"):
        trial.load_qualified_embodiment_inputs(f.spec, f.checkpoint, **f.kwargs)


def test_proof_rejects_missing_repeat_trace_and_wrong_caller_reference(qualified_files):
    f = qualified_files
    f.proof["artifacts"].pop("gpu_repeat_1")
    _rewrite_proof(f)
    with pytest.raises(ValueError, match="Every CPU/GPU"):
        trial.load_qualified_embodiment_inputs(f.spec, f.checkpoint, **f.kwargs)


@pytest.fixture
def stub_parent(monkeypatch):
    calls = SimpleNamespace(init=0, request=0, metadata=[], state_snapshots=[])

    def initialize(self, spec, checkpoint, *, episode_id):
        calls.init += 1
        self.model = SimpleNamespace(backbone=StubBackbone())
        power = float(torch.tensor(0.010000001))
        self.state = torch.tensor([[0.1, 0.1, 0.1, 0.1, power, power]], dtype=torch.float32)
        self.clock = self.held = None
        self.reference = {
            k: v.tolist() for k, v in self.model.backbone.readout.state_dict().items()
        }
        calls.state_snapshots.append(self.state.clone())

    def request(self, sample, physical_command, decision_clock):
        calls.request += 1
        prior = self.clock
        self.state = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.006, 0.006]], dtype=torch.float32)
        raw, mn = self.model.backbone._read_state_tensor(self.state)
        intent = raw[0].detach().numpy()
        control = canonical_controls(intent)
        self.clock = decision_clock
        self.held = (sample, physical_command)
        meta = dict(
            decision_clock=decision_clock,
            original_elapsed_seconds=None if prior is None else decision_clock - prior,
            native_packet=7,
            original_native_receipt=decision_clock - 0.001,
            body=list(range(21)),
            body_valid=[True] * 21,
            physical_command=list(physical_command),
            intent=intent.tolist(),
            muscles=mn[0].detach().tolist(),
            canonical_request=control.tolist(),
            first_state="synthetic_quiet1s" if prior is None else None,
            current_command_held_for_next_interval=True,
            numerical_state_reset=False,
        )
        calls.metadata.append(meta)
        return control, meta

    monkeypatch.setattr(trial.NeuralPhysicalProbe, "__init__", initialize)
    monkeypatch.setattr(trial.NeuralPhysicalProbe, "request", request)
    return calls


def test_initialization_installs_only_cpu_neutral_preserving_quiet_and_other_weights(
    qualified_files, stub_parent
):
    f = qualified_files
    value = trial.NeuralEmbodimentV2Probe(f.spec, f.checkpoint, episode_id="unit", **f.kwargs)
    assert stub_parent.init == 1 and stub_parent.request == 0
    assert torch.equal(value.state, stub_parent.state_snapshots[0])
    assert value.reference == f.readout
    assert value.embodiment_initialization["changed_neutral_keys"] == ["readout.power_reference"]
    assert value.embodiment_initialization["all_other_backbone_values_unchanged"]
    assert value.embodiment_initialization["encoded_neutral_check"][1:] == [0.0, 0.0]
    assert not value.embodiment_initialization["actuator_acquisition_performed"]


def test_parent_request_exactly_once_per_call_and_generic_control_means_actual_v2(
    qualified_files, stub_parent
):
    f = qualified_files
    value = trial.NeuralEmbodimentV2Probe(f.spec, f.checkpoint, episode_id="unit", **f.kwargs)
    sample = object()
    command = [0.1, 1.0]
    control, metadata = value.request(sample, command, 1.0)
    assert stub_parent.request == 1 and value.clock == 1.0 and value.held == (sample, command)
    original = stub_parent.metadata[0]
    for k, v in original.items():
        if k != "canonical_request":
            assert metadata[k] == v
    assert metadata["canonical_v1_request"] == original["canonical_request"]
    assert np.array_equal(control, metadata["canonical_request"])
    assert metadata["canonical_request_schema"] == probe.EMBODIMENT_V2_CANONICAL_SCHEMA
    assert control[2] > original["canonical_request"][2]
    assert metadata["embodiment_v2"]["encoded_request_not_applied_here"]
    probe.validate_embodiment_v2_return(control, metadata)
    _, second = value.request(sample, command, 1.04)
    assert stub_parent.request == 2 and second["original_elapsed_seconds"] == 1.04 - 1.0


def test_actual_quiet_mismatch_rejects_before_caller_can_acquire(
    monkeypatch, qualified_files, stub_parent
):
    f = qualified_files
    old = trial.NeuralPhysicalProbe.__init__

    def incorrect(self, *args, **kwargs):
        old(self, *args, **kwargs)
        self.state[:, 4:] = 0.006

    monkeypatch.setattr(trial.NeuralPhysicalProbe, "__init__", incorrect)
    with pytest.raises(ValueError, match="quiet raw/v2/encoded neutral"):
        trial.NeuralEmbodimentV2Probe(f.spec, f.checkpoint, episode_id="unit", **f.kwargs)
    assert stub_parent.init == 1 and stub_parent.request == 0


def test_qualified_gain_change_is_not_installed_silently(monkeypatch, qualified_files, stub_parent):
    f = qualified_files
    old = trial.NeuralPhysicalProbe.__init__

    def incorrect(self, *args, **kwargs):
        old(self, *args, **kwargs)
        with torch.no_grad():
            self.model.backbone.readout.forward_log_gain.add_(1.0)

    monkeypatch.setattr(trial.NeuralPhysicalProbe, "__init__", incorrect)
    with pytest.raises(ValueError, match="gains/mix/ports"):
        trial.NeuralEmbodimentV2Probe(f.spec, f.checkpoint, episode_id="unit", **f.kwargs)
