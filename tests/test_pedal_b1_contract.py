"""B1 metadata contracts only; synthetic review dictionaries are not attestations."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from flybrain_drive.actor_cadence_trace_v1 import ActorCadenceProfile
from flybrain_drive.frame_persistence_profile_v4 import HostCopyProfile
from flybrain_drive.pedal_load_b1 import NATIVE_SOURCE_FIELDS, load_candidate
from flybrain_drive.pedal_train_b1 import (
    CONFIGURATION_BINDINGS,
    PROTECTED,
    R2_INPUTS,
    R2_SOURCE_REVIEW_SHA256,
    SOURCE_FILES,
    fresh_start_gate,
    pipeline_gate,
    sha_bytes,
    sha_file,
    validate_declaration,
)
from scripts.build_pedal_b1 import BASE, TARGET, build, edits, inverse

spec = importlib.util.spec_from_file_location(
    "B1_synthetic_fixture", Path(__file__).with_name("test_pedal_b1.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
RUNTIME = Path("C:/Users/jeff4/AppData/Local/flybrain/runtime-runs")


def pipeline_fixture(declaration):
    """In-memory schema fixture only; never saved as a role/qualification artifact."""
    cases = []
    references = (
        "capture_manifest_sha256",
        "parent_result_sha256",
        "raw_seven_array_audit_sha256",
        "heartbeat_full_byte_audit_sha256",
        "actor_trace_audit_sha256",
        "native_authority_audit_sha256",
        "original_terminal_receipt_sha256",
    )
    for i, (mode, duration) in enumerate([("camera3", 3), ("pace30", 30)]):
        case = dict(
            mode=mode,
            duration_seconds=duration,
            original_role="diagnostic",
            original_exit_code=0,
            decisions=3,
            camera_wait_misses=0,
            maximum_original_interval_seconds=0.04,
            terminal_host_qpc_ns=100 + i * 100,
            parent_source_sha256=R2_INPUTS["scripts/qualify_session_heartbeat_actor_trace_v1.py"],
            child_source_sha256=R2_INPUTS["scripts/record_ac_engine_ai_probe_actor_trace_v1.py"],
            actor_trace_source_sha256=R2_INPUTS["src/flybrain_drive/actor_cadence_trace_v1.py"],
            raw_seven_arrays_exact=True,
            all_camera_aids_OFF=True,
            actor_trace_complete=True,
            heartbeat_retirement_complete=True,
        )
        case.update({key: str(i + 1) * 64 for key in references})
        cases.append(case)
    return dict(
        schema="flybrain.pedal_b1.pipeline_review.v1",
        status="qualified_exact_R2_actor3_and30",
        source_sha256=declaration["source_sha256"],
        R2_source_review_sha256=R2_SOURCE_REVIEW_SHA256,
        fit_allowed=False,
        host_clock="perf_counter_ns",
        cases=cases,
        last_diagnostic_terminal_host_qpc_ns=200,
    )


def pipeline_bytes(declaration, value):
    raw = json.dumps(value).encode()
    declaration["pipeline_qualification_review_sha256"] = sha_bytes(raw)
    return raw


def test_parent_inverse_and_exact_actual_eleven_fields_and_full_dependency_closure():
    assert len(edits()) == 6 and build(BASE.read_bytes()) == TARGET.read_bytes()
    assert inverse(TARGET.read_bytes()) == BASE.read_bytes()
    tree = ast.parse(Path("scripts/record_ac_engine_ai_probe_actor_trace_v1.py").read_bytes())
    actual = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SAME_COPY_SOURCE_FILES" for t in n.targets)
    )
    assert len(actual) == 11 and NATIVE_SOURCE_FIELDS == frozenset(actual)
    assert NATIVE_SOURCE_FIELDS <= set(SOURCE_FILES)
    assert set(CONFIGURATION_BINDINGS) <= set(SOURCE_FILES)
    assert set(R2_INPUTS) <= set(SOURCE_FILES)
    assert {
        "src/flybrain_drive/source_closure_inventory_v1.py",
        "scripts/record_pedal_b1.py",
        "scripts/prepare_pedal_b1.py",
        "src/flybrain_drive/pedal_load_b1.py",
        "configs/pedal-b1-20261030.json",
    } <= set(SOURCE_FILES)


def test_real_B1_draft_has_no_actual_identity_pipeline_nonce_or_output():
    draft = json.loads(Path("configs/pedal-b1-20261030.json").read_bytes())
    validate_declaration(draft, sealed=False)
    assert draft["bindings"] == dict(
        engine_pid=None, generation=None, epoch=None, installed_lua_sha256=None, output=None
    )
    assert draft["nonce"] is None and draft["source_sha256"] == {}
    assert draft["actual_pipeline_qualification_completed"] is False
    assert all(
        draft[k] is None
        for k in (
            "pipeline_qualification_review_file",
            "route_preflight_file",
            "native_pedal_continuity_review_file",
            "controls_ini_sha256",
            "native_pedal_calibration_sha256",
            "native_pedal_continuity_review_sha256",
            "route_preflight_sha256",
            "expected_start_spline",
        )
    )
    assert draft["start_region"] == [0.49, 0.54] and draft["whole_region"] == [0.49, 0.64]
    assert tuple(map(tuple, draft["protected_regions"])) == PROTECTED
    assert draft["heldout_or_independent_geographic_claim"] is False
    with pytest.raises(ValueError):
        validate_declaration(draft)


@pytest.mark.parametrize(
    "change", ["missing30", "old_child", "failed3", "gap", "reuse3", "wrong_duration"]
)
def test_external_pipeline_cannot_infer_actual_success_from_false_or_foreign_cases(change):
    declaration = fixture.declaration()
    value = pipeline_fixture(declaration)
    if change == "missing30":
        value["cases"].pop()
    elif change == "old_child":
        value["cases"][0]["child_source_sha256"] = "0" * 64
    elif change == "failed3":
        value["cases"][0]["decisions"] = 0
    elif change == "gap":
        value["cases"][1]["maximum_original_interval_seconds"] = 0.080001
    elif change == "reuse3":
        value["cases"][1]["capture_manifest_sha256"] = value["cases"][0]["capture_manifest_sha256"]
    else:
        value["cases"][1]["duration_seconds"] = 3
    with pytest.raises(ValueError):
        pipeline_gate(pipeline_bytes(declaration, value), declaration)


def test_fresh_observer_is_after_both_terminal_receipts_and_nonce_source_bound():
    declaration = fixture.declaration()
    pipeline = pipeline_fixture(declaration)
    pipeline_gate(pipeline_bytes(declaration, pipeline), declaration)
    start = dict(
        schema="flybrain.pedal_b1.fresh_start_review.v1",
        nonce=declaration["nonce"],
        source_sha256=declaration["source_sha256"],
        engine_pid=123,
        generation="fixture-generation",
        epoch=1,
        controls_ini_sha256=declaration["controls_ini_sha256"],
        host_clock="perf_counter_ns",
        observer_host_qpc_ns_lower_bound=201,
        spline_position=0.5,
        lap_count=0,
        actor_input=False,
        after_diagnostic_terminal_receipt_sha256=[
            c["original_terminal_receipt_sha256"] for c in pipeline["cases"]
        ],
        paused_fullbrake=True,
        CSP_AIoff=True,
        FREE_background=True,
        no_owner=True,
        native_graphics_advance_verified=True,
        original_observer_result_sha256="3" * 64,
    )
    for field, wrong in [
        ("observer_host_qpc_ns_lower_bound", 200),
        ("nonce", "b" * 32),
        ("spline_position", 0.489999),
        ("epoch", True),
        ("lap_count", False),
    ]:
        changed = dict(start, **{field: wrong})
        raw = json.dumps(changed).encode()
        declaration["route_preflight_sha256"] = sha_bytes(raw)
        with pytest.raises(ValueError):
            fresh_start_gate(raw, declaration, pipeline)
    raw = json.dumps(start).encode()
    declaration["route_preflight_sha256"] = sha_bytes(raw)
    assert fresh_start_gate(raw, declaration, pipeline)["actor_input"] is False


@pytest.mark.parametrize(
    "source",
    [
        "20261030-t1856-new-pedal-TRAIN-host-v4",
        "20261030-t1845-host-v4-heartbeat-evidence-camera3",
        "20261030-t1849-host-v4-heartbeat-evidence-pace30",
    ],
)
def test_original_oldTRAIN_schema_and_diagnostic_roles_refused_before_any_payload(
    source, monkeypatch
):
    root = RUNTIME / source
    original_open = Path.open

    def no_payload(self, *args, **kwargs):
        if self.name in ("events.jsonl", "trace.bin") or self.suffix.lower() in (
            ".npz",
            ".zip",
            ".bmp",
            ".png",
        ):
            raise AssertionError("No payload may precede original B1 role/schema refusal")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_payload)
    with pytest.raises(ValueError):
        load_candidate(root, expected_capture_manifest_sha256=sha_file(root / "manifest.json"))


@pytest.mark.parametrize(
    "where", ["parent", "child", "result", "trace", "source_fields", "observations_alias"]
)
def test_full_synthetic_loader_cannot_omit_profiles_trace_or_actual_fields(
    tmp_path, monkeypatch, where
):
    fixture.test_complete_synthetic_file_loader_uses_metadata_only(tmp_path, monkeypatch)
    root = tmp_path / "future-train-candidate"
    if where == "parent":
        p = root / "manifest.json"
        value = json.loads(p.read_bytes())
        value["config"]["actor_cadence_profile"]["nominal_actor_seconds"] = 0.04
    elif where == "child":
        p = root / "recording/manifest.json"
        value = json.loads(p.read_bytes())
        del value["config"]["actor_cadence_profile"]
    elif where == "result":
        p = root / "recording/result.json"
        value = json.loads(p.read_bytes())
        value["frame_persistence_profile"]["original_actor_interval_seconds"] = 0.081
    elif where == "trace":
        p = root / "actor-trace/index.json"
        value = json.loads(p.read_bytes())
        value["complete"] = False
    elif where == "source_fields":
        p = root / "recording/native-authority-source-proof.json"
        value = json.loads(p.read_bytes())
        del value["source_files"]["src/flybrain_drive/actor_cadence_trace_v1.py"]
    else:
        p = root / "recording/result.json"
        value = json.loads(p.read_bytes())
        value["observations"] = value.pop("decisions")
    p.write_text(json.dumps(value), encoding="utf8")
    with pytest.raises(ValueError):
        load_candidate(root, expected_capture_manifest_sha256=sha_file(root / "manifest.json"))


def test_schema_only_observation_of_real_failed_R2_metadata_does_not_qualify_pipeline():
    root = RUNTIME / "20261030-t1875-actor-cadence-heartbeat-evidence-camera3"
    result = json.loads((root / "recording/result.json").read_bytes())
    proof = json.loads((root / "recording/native-authority-source-proof.json").read_bytes())
    ActorCadenceProfile.from_record(result["actor_cadence_profile"])
    HostCopyProfile.from_record(result["frame_persistence_profile"])
    assert (
        result["decisions"] == 0
        and proof["schema"] == "flybrain.native.same_copy_recorder_source_proof.v1"
    )
    assert set(proof["source_files"]) == NATIVE_SOURCE_FIELDS
    assert result["status"] == "failed"
    # No event/pixel/NPZ/trace payload is read, and this observed failure supplies
    # no B1 nonce, qualification flag, source role, calibration or fit permission.
