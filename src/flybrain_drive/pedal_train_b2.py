"""Prospective TRAIN identity and geographic observer contract; no live interfaces."""

import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "flybrain.pedal_b2.declaration.v1"
POLICY_SHA256 = "815b2bb2451fee3e62391f319bfe80bf4cba92549a0e7faebd67c19bdfc83a07"
INVENTORY_SHA256 = "a8891d423fe650d02f0c6eac2eaa005907f930f60e55991be899a5a326fa1d96"
ACTOR_PROFILE = "configs/ac-actor-cadence-trace-30ms-20261030.json"
R2_SOURCE_REVIEW_SHA256 = "d9c12bed20a5e35104e23bce0c9a6e9543ae76774e517bde22a041ab7f8c4310"
R2_INPUTS = {
    "configs/ac-actor-cadence-trace-30ms-20261030.json": (
        "d4812b5705caf89d2b22d54b53ef0c552ca9d80e94ba7dd22f4f8a4c75acd0a3"
    ),
    "configs/flight-pedal-region-b-policy-draft-20261030.json": (
        "815b2bb2451fee3e62391f319bfe80bf4cba92549a0e7faebd67c19bdfc83a07"
    ),
    "scripts/audit_ac_engine_ai_probe_actor_trace_v1.py": (
        "f5f3a8c3b8cac746359085871711ce8de3998d6ac2ec8e320c230ae136877c32"
    ),
    "scripts/prepare_actor_cadence_trace_v1.py": (
        "872d83a3043e25778ded50c5262ddf77197e4c4326e5db52214e347b08b8981f"
    ),
    "scripts/qualify_session_heartbeat_actor_trace_v1.py": (
        "09f25ce66f619ee52abc5b3a32c7cbb380bbe9c21d6cecc72d1ca4b113f3ef64"
    ),
    "scripts/record_ac_engine_ai_probe_actor_trace_v1.py": (
        "4c91923ca189caa1b1aa9ef7b38d41ac9fc4415c9596be96b665eeaee0dc9ba7"
    ),
    "src/flybrain_drive/actor_cadence_trace_v1.py": (
        "37e1448e2d4f54e57ba991f110e3095c03e6f961ffb721d9b40e9824c817595a"
    ),
    "tests/test_actor_cadence_trace_v1.py": (
        "9bc10c8d2f3586c7d932a027ef602b4a5b180251c80a5a23227431bef0d823c4"
    ),
    "tests/test_actor_cadence_trace_v1_edges.py": (
        "36dbfee9c251a3b9c05813a63f1f1906ad3d4a0c37616a2cf727522c57b2e311"
    ),
    "tests/test_actor_cadence_trace_v1_faults.py": (
        "5fea467bdfc32fda3add31d58ff3f8c7f76444490979a6e07796262026478779"
    ),
}
CAMERA_STATUS_PROFILE = "configs/ac-mmf-lifecycle-status-v1-20261030.json"
LIFECYCLE_SOURCE_REVIEW_SHA256 = "b2f0bbfe14fa4d3fb8db6e19fa2c3aa7ea7349d913d541917d20f99ac42484e3"
LIFECYCLE_INPUTS = {
    "integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua": (
        "80605ed6aa1cf64851aad648b5107c9ea6a98a684e3d583ce0638beba4e7ab5f"
    ),
    "src/flybrain_drive/camera_status_policy_v1.py": (
        "7bfcb3ab66de1446fa3f892168b46b3c4ac255e4adecd8d51a3cf04e2148bcc7"
    ),
    "src/flybrain_drive/ac_csp_vision_lifecycle_status_v1.py": (
        "53b87cda9aee25f21396e68d4e911ca6f8723551fe91f23d106c2e4ed9356aab"
    ),
    "scripts/record_ac_engine_ai_probe_lifecycle_status_v1.py": (
        "16ad12a8a91cf7df7073cdba4e8fab3eb88bcad78d21d06d348b126d5bf7d980"
    ),
    "scripts/qualify_session_heartbeat_lifecycle_status_v1.py": (
        "5032213b55135eb5549aa1f5ada148809c4f0d2bbdc8098eaf64e63adf831aba"
    ),
    "scripts/audit_ac_engine_ai_probe_lifecycle_status_v1.py": (
        "ab5c345e818c574249f4b27f9a3d7e6dfd8717208228be67b2a65d5eded063db"
    ),
    "scripts/prepare_camera_lifecycle_status_v1.py": (
        "da16b0d9b02cdd70a70806d834f24e230152b6b2cc2659cd650d60567c37c430"
    ),
    "configs/ac-mmf-lifecycle-status-v1-20261030.json": (
        "1916b398bc920a8f40d5417a448764b2e3b443cc2516802c29562212752c3567"
    ),
    "tests/test_camera_lifecycle_status_v1.py": (
        "380bb09f99d6d572a79091a08e21c8d10b6d87490b6856a058eb28d008bc1248"
    ),
}

PROTECTED = ((0.08, 0.125), (0.2, 0.25), (0.67, 0.84), (0.94, 1.0))
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SEEDS = (
    "scripts/qualify_session_heartbeat_actor_trace_v1.py",
    "scripts/record_ac_engine_ai_probe_actor_trace_v1.py",
    "src/flybrain_drive/frame_persistence_profile_v4.py",
    "src/flybrain_drive/ac_csp_vision_host_v4.py",
    "src/flybrain_drive/ac_csp_retina.py",
    "src/flybrain_drive/ac_native_physics.py",
    "src/flybrain_drive/ac_teacher_physics.py",
    "src/flybrain_drive/ac_teacher_status.py",
    "src/flybrain_drive/ac_image_buffer.py",
    "scripts/record_ac_wgc.py",
    "src/flybrain_drive/session_heartbeat_runtime.py",
    "src/flybrain_drive/session_heartbeat_provenance.py",
    "src/flybrain_drive/session_heartbeat_evidence_v1.py",
    "src/flybrain_drive/session_heartbeat_terminal_v2.py",
    "src/flybrain_drive/session_heartbeat_admission.py",
    "src/flybrain_drive/ac_published_state.py",
    "src/flybrain_drive/ac_csp_clock.py",
    "src/flybrain_drive/teacher_pace.py",
    "src/flybrain_drive/sensor_contract.py",
    "scripts/control_ac_session.py",
    "src/flybrain_drive/virtual_navigation_body.py",
    "src/flybrain_drive/virtual_body.py",
    "src/flybrain_drive/teacher_loss_eligibility.py",
    "src/flybrain_drive/pedal_train_b2.py",
    "src/flybrain_drive/pedal_load_b2.py",
    "scripts/record_pedal_b2.py",
    "configs/ac-frame-host-v4-20261030.json",
    "configs/ac-virtual-navigation-body-v2.json",
    "configs/ac-virtual-body-v1.json",
    "configs/ac-csp-exterior-front-v3-retina.json",
    "configs/ac-axis-probe.json",
    "src/flybrain_drive/immutable_frame_writer.py",
    "src/flybrain_drive/ac_capture.py",
    "src/flybrain_drive/ac_vjoy.py",
    "src/flybrain_drive/evidence.py",
    "scripts/build_pedal_b2.py",
    "scripts/prepare_frame_host_v4.py",
    "scripts/build_pedal_b2.py",
    "scripts/prepare_pedal_b2.py",
    "scripts/load_pedal_b2.py",
    "src/flybrain_drive/source_closure_inventory_v1.py",
    "src/flybrain_drive/actor_cadence_trace_v1.py",
    "configs/ac-actor-cadence-trace-30ms-20261030.json",
    "configs/pedal-b2-20261030.json",
    "configs/flight-pedal-region-b-policy-draft-20261030.json",
    "integrations/ac_csp/candidates/FlyBrainCamera-heartbeat-wire-v2.lua",
    "integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua",
)


def local_import_closure(entries, *, project_root=_PROJECT_ROOT):
    """New-only inventoried legacy semantics; first-adoption equivalence is sealed separately."""
    helper = Path(project_root) / "src/flybrain_drive/source_closure_inventory_v1.py"
    if hashlib.sha256(helper.read_bytes()).hexdigest() != INVENTORY_SHA256:
        raise ValueError("Pinned new inventory helper changed; no fallback to old resolver")
    from .source_closure_inventory_v1 import SourceInventory

    inventory = SourceInventory.capture(project_root)
    result = inventory.local_import_closure(entries)
    inventory.verify_unchanged()
    return result


def consumed_configuration_bindings(*, project_root=_PROJECT_ROOT):
    """Explicit runtime-consumed file refs; unrelated historical pixel refs stay unopened."""
    root = Path(project_root)
    navigation = json.loads((root / "configs/ac-virtual-navigation-body-v2.json").read_bytes())
    legacy_name = navigation["legacy_contract"]
    legacy = json.loads((root / legacy_name).read_bytes())
    axis = json.loads((root / "configs/ac-axis-probe.json").read_bytes())
    return {
        legacy_name: navigation["legacy_contract_sha256"],
        legacy["geometry_review"]: legacy["geometry_review_sha256"],
        axis["dll_path"]: axis["dll_sha256"],
        axis["profile_path"]: axis["profile_sha256"],
    }


CONFIGURATION_BINDINGS = consumed_configuration_bindings()
SOURCE_SEEDS += (
    "scripts/qualify_session_heartbeat_lifecycle_status_v1.py",
    "scripts/record_ac_engine_ai_probe_lifecycle_status_v1.py",
    "scripts/audit_ac_engine_ai_probe_lifecycle_status_v1.py",
    "src/flybrain_drive/ac_csp_vision_lifecycle_status_v1.py",
    "src/flybrain_drive/camera_status_policy_v1.py",
    "scripts/prepare_camera_lifecycle_status_v1.py",
    CAMERA_STATUS_PROFILE,
    "integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua",
    # Mandatory provenance paths from the reviewed1895 closure, not runtime role reuse.
    "scripts/load_pedal_train_candidate_host_v4.py",
    "scripts/prepare_pedal_train_collector.py",
    "scripts/prepare_pedal_train_collector_host_v4.py",
    "scripts/prepare_pedal_train_declaration_host_v4.py",
    "scripts/prepare_pedal_train_host_v4.py",
    "scripts/qualify_session_heartbeat_evidence_host_v4.py",
    "scripts/record_ac_engine_ai_probe_host_v4.py",
    "scripts/record_pedal_train_candidate_host_v4.py",
    "src/flybrain_drive/pedal_train_candidate.py",
    "src/flybrain_drive/pedal_train_candidate_host_v4.py",
    "src/flybrain_drive/pedal_train_loader_host_v4.py",
)

SOURCE_FILES = tuple(
    sorted(
        set(local_import_closure(SOURCE_SEEDS))
        | set(CONFIGURATION_BINDINGS)
        | set(R2_INPUTS)
        | set(LIFECYCLE_INPUTS)
    )
)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def validate_declaration(value, *, sealed=True):
    if not isinstance(value, dict):
        raise ValueError("Typed new B2 declaration required")
    if (
        value.get("schema") != SCHEMA
        or value.get("original_role") != "train"
        or value.get("fit_allowed") is not False
        or value.get("optimizer_updates") != 0
        or value.get("duration_seconds") != 30
        or value.get("yaw_policy") != "prospectively_declared_zero_component"
        or value.get("trainable_parameters")
        != ["command.power_delta", "command.speed_feedforward_delta"]
        or value.get("start_region") != [0.49, 0.54]
        or value.get("whole_region") != [0.49, 0.64]
        or value.get("protected_regions") != [list(r) for r in PROTECTED]
        or value.get("pedal_projection_policy") != "reject_unreachable"
        or value.get("minimum_positive") != 0.02
    ):
        raise ValueError("Exact prospective whole-history TRAIN component contract required")
    if (
        value.get("profile") != "configs/ac-frame-host-v4-20261030.json"
        or value.get("host_copy_profile_schema") != "flybrain.frame_host_copy_profile.v4"
    ):
        raise ValueError("Explicit separately qualified hostV4 profile required")
    if value.get("body_contract") != "configs/ac-virtual-navigation-body-v2.json":
        raise ValueError("Explicit unchanged body21 contract required")
    if (
        value.get("actor_profile") != ACTOR_PROFILE
        or value.get("actor_profile_schema") != "flybrain.actor_cadence_trace.v1"
        or value.get("policy_input_sha256") != POLICY_SHA256
        or value.get("heldout_or_independent_geographic_claim") is not False
    ):
        raise ValueError("Exact R2 actor30ms and non-heldout B2 policy required")
    if (
        value.get("camera_status_profile") != CAMERA_STATUS_PROFILE
        or value.get("camera_status_profile_schema")
        != "flybrain.csp.camera.status_publication_profile.v1"
        or value.get("camera_status_policy") != "mmf_lifecycle_only_v1"
        or value.get("lifecycle_source_review_sha256") != LIFECYCLE_SOURCE_REVIEW_SHA256
        or value.get("data_or_fit_permission") is not False
    ):
        raise ValueError("Exact lifecycle profile/source review and unadmitted B2 policy required")
    if not sealed:
        return
    if value.get("actual_pipeline_qualification_completed") is not True:
        raise ValueError(
            "Actual reviewed hostV4 pipeline qualification required before role seal/capture"
        )
    for field in (
        "pipeline_qualification_review_file",
        "route_preflight_file",
        "native_pedal_continuity_review_file",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError("Fresh external qualification/start/pedal review files required")
    if not is_sha(value.get("controls_ini_sha256")):
        raise ValueError("Exact controls.ini input configuration binding required")
    if not isinstance(value.get("source_sha256"), dict):
        raise ValueError("Explicit source mapping required")
    bindings = value.get("bindings", {})
    if (
        value.get("status") != "sealed_before_capture"
        or not re.fullmatch("[0-9a-f]{32}", str(value.get("nonce")))
        or type(bindings.get("engine_pid")) is not int
        or bindings["engine_pid"] <= 0
        or type(bindings.get("epoch")) is not int
        or bindings["epoch"] < 0
        or not isinstance(bindings.get("generation"), str)
        or not bindings["generation"]
        or not is_sha(bindings.get("installed_lua_sha256"))
        or not isinstance(bindings.get("output"), str)
        or not bindings["output"]
        or not is_sha(value.get("native_pedal_calibration_sha256"))
        or not is_sha(value.get("native_pedal_continuity_review_sha256"))
        or not is_sha(value.get("route_preflight_sha256"))
        or type(value.get("expected_start_spline")) not in (int, float)
        or not 0.49 <= value["expected_start_spline"] <= 0.54
        or not is_sha(value.get("pipeline_qualification_review_sha256"))
        or set(SOURCE_FILES) - set(value.get("source_sha256", {}))
    ):
        raise ValueError("Fresh identity/calibration/source qualification bindings required")
    if any(value["source_sha256"].get(p) != h for p, h in R2_INPUTS.items()):
        raise ValueError("Exact reviewed R2 source/qualification inputs required")
    if any(value["source_sha256"].get(p) != h for p, h in LIFECYCLE_INPUTS.items()):
        raise ValueError("Exact reviewed lifecycle sources required; no automatic upgrade")
    controls = json.loads((_PROJECT_ROOT / "configs/ac-axis-probe.json").read_bytes())
    if value["controls_ini_sha256"] != controls["profile_sha256"]:
        raise ValueError("Declared controls.ini differs from qualified input configuration")
    if any(not is_sha(v) for v in value["source_sha256"].values()):
        raise ValueError("Explicit sealed source hashes required")
    if any(value["source_sha256"][p] != expected for p, expected in CONFIGURATION_BINDINGS.items()):
        raise ValueError("Consumed body geometry/DLL reference binding differs")
    if any(
        value["source_sha256"][p] != bindings["installed_lua_sha256"]
        for p in (
            "integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua",
            "integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua",
        )
    ):
        raise ValueError("Actual child main/candidate Lua source binding differs")


def seal_declaration(value, destination):
    """Write a new immutable role file only; never discover a PID or launch capture."""
    validate_declaration(value, sealed=False)
    sealed = dict(value, status="sealed_before_capture")
    validate_declaration(sealed)
    external_binding_gate(sealed)
    for name, expected in sealed["source_sha256"].items():
        if sha_file(name) != expected:
            raise ValueError("No automatic source repinning: " + name)
    raw = (json.dumps(sealed, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(destination).open("xb") as stream:
        stream.write(raw)
    return dict(
        sha256=sha_bytes(raw),
        nonce=sealed["nonce"],
        original_role="train",
        collection_started=False,
        fit_allowed=False,
    )


def collector_declaration(args):
    raw = Path(args.train_declaration).read_bytes()
    value = json.loads(raw)
    validate_declaration(value)
    external_binding_gate(value)
    b = value["bindings"]
    if (
        args.mode != "pace30"
        or args.expected_engine_pid != b["engine_pid"]
        or args.expected_generation != b["generation"]
        or args.expected_epoch != b["epoch"]
        or args.expected_installed_sha256 != b["installed_lua_sha256"]
        or Path(args.output).resolve() != Path(b["output"]).resolve()
        or sha_file(args.frame_persistence_profile)
        != value["source_sha256"]["configs/ac-frame-host-v4-20261030.json"]
        or sha_file(args.actor_cadence_profile) != value["source_sha256"][ACTOR_PROFILE]
        or sha_file(args.camera_status_profile) != value["source_sha256"][CAMERA_STATUS_PROFILE]
    ):
        raise ValueError("Exact prospectively sealed capture identity/output/profile required")
    for name, expected in value["source_sha256"].items():
        if sha_file(name) != expected:
            raise ValueError("Sealed source changed; no repin/retry: " + name)
    return dict(raw=raw, value=value, sha256=sha_bytes(raw), nonce=value["nonce"])


def pedal_gate(review_bytes, declaration):
    """Bind an EXTERNAL review to original plant calibration and fresh nonce/path.

    Parsing an attestation is not physical recalibration or independent verification.
    The originating reviewer/source evidence must remain separately reviewable.
    """
    if review_bytes is None:
        return dict(attached=False, qualified=False, reason="external_review_missing")
    value = json.loads(review_bytes)
    if (
        sha_bytes(review_bytes) != declaration["native_pedal_continuity_review_sha256"]
        or value.get("schema") != "flybrain.pedal_calibration_continuity_review.v1"
        or value.get("original_calibration_sha256")
        != (declaration["native_pedal_calibration_sha256"])
        or value.get("nonce") != declaration["nonce"]
        or value.get("current_engine_pid") != declaration["bindings"]["engine_pid"]
        or value.get("current_generation") != declaration["bindings"]["generation"]
        or value.get("physical_pedal_input_path_qualified") is not True
        or value.get("steering_qualification_claimed") is not False
        or value.get("fit_allowed") is not False
        or value.get("source_sha256") != declaration["source_sha256"]
        or value.get("controls_ini_sha256") != declaration["controls_ini_sha256"]
    ):
        raise ValueError("Original pedal calibration/fresh continuity review required")
    return dict(
        attached=True,
        qualified=True,
        review_sha256=sha_bytes(review_bytes),
        qualification_is_external_not_loader_measured=True,
    )


def route_admission(receipt, declaration):
    """Check every retained observer sample, never manufacture continuous coverage."""
    reasons = []
    samples = [receipt.get("before"), *receipt.get("telemetry", []), receipt.get("after")]
    positions, laps = [], []
    for sample in samples:
        route = sample.get("diagnostic_route") if isinstance(sample, dict) else None
        if not isinstance(route, dict) or route.get("actor_input") is not False:
            reasons.append("missing_original_observer_route")
            continue
        p, lap = route.get("spline_position"), route.get("lap_count")
        if (
            type(p) not in (int, float)
            or not math.isfinite(p)
            or not 0 <= p < 1
            or type(lap) not in (int, float)
            or not math.isfinite(lap)
            or lap < 0
            or int(lap) != lap
        ):
            reasons.append("invalid_original_observer_route")
            continue
        positions.append(p)
        laps.append(lap)
        if not 0.49 <= p <= 0.64:
            reasons.append("whole_route_outside_train_region")
        if any(lo <= p <= hi for lo, hi in PROTECTED):
            reasons.append("protected_DEV_EVAL_overlap")
        if sample.get("wheels_outside") != 0:
            reasons.append("offtrack_or_unknown_wheels")
    telemetry = receipt.get("telemetry", [])
    if not telemetry or not {"driving", "stopping"} <= {s.get("phase") for s in telemetry}:
        reasons.append("before_driving_stopping_after_coverage_incomplete")
    if positions:
        if not 0.49 <= positions[0] <= 0.54:
            reasons.append("start_outside_declared_train_start_region")
        if abs(positions[0] - declaration["expected_start_spline"]) > 1e-4:
            reasons.append("original_start_differs_from_preassigned_observer_start")
        if any(a - b > 0.001 for a, b in zip(positions, positions[1:], strict=False)):
            reasons.append("route_reverse_wrap_or_discontinuity")
        if len(set(laps)) != 1:
            reasons.append("lap_changed")
    return dict(
        passed=not reasons,
        reasons=sorted(set(reasons)),
        observer_samples=len(samples),
        observed_positions=len(positions),
        minimum_spline=min(positions) if positions else None,
        maximum_spline=max(positions) if positions else None,
        continuous_geometry_verified=False,
        actor_input=False,
        entire_history_retained=True,
        external_observer_coverage_qualification_required=True,
        fit_allowed=False,
    )


def _review_file(path, expected):
    """Only an explicit bounded hash-bound metadata review, never an inferred attestation."""
    if not isinstance(path, str) or Path(path).suffix.lower() != ".json" or not is_sha(expected):
        raise ValueError("Explicit original JSON review path and SHA required")
    raw = Path(path).read_bytes()
    if not 0 < len(raw) <= 256 * 1024 or sha_bytes(raw) != expected:
        raise ValueError("Bounded original review bytes differ")
    return raw


def pipeline_gate(raw, declaration):
    """Typed EXTERNAL review; parsing this is not actual raw or physical qualification."""
    value = json.loads(raw)
    if (
        sha_bytes(raw) != declaration["pipeline_qualification_review_sha256"]
        or value.get("schema") != "flybrain.pedal_b2.pipeline_review.v1"
        or value.get("status") != "qualified_exact_lifecycle_actor3_and30"
        or value.get("source_sha256") != declaration["source_sha256"]
        or value.get("R2_source_review_sha256") != R2_SOURCE_REVIEW_SHA256
        or value.get("lifecycle_source_review_sha256") != LIFECYCLE_SOURCE_REVIEW_SHA256
        or value.get("fit_allowed") is not False
        or value.get("host_clock") != "perf_counter_ns"
    ):
        raise ValueError("Actual reviewed exact R2 camera3+pace30 pipeline evidence required")
    cases = value.get("cases")
    if not isinstance(cases, list) or [c.get("mode") for c in cases] != ["camera3", "pace30"]:
        raise ValueError("Both original R2 diagnostic histories required")
    terminals = []
    required_refs = (
        "capture_manifest_sha256",
        "parent_result_sha256",
        "raw_seven_array_audit_sha256",
        "heartbeat_full_byte_audit_sha256",
        "actor_trace_audit_sha256",
        "native_authority_audit_sha256",
        "original_terminal_receipt_sha256",
        "camera_status_policy_audit_sha256",
    )
    for case, duration in zip(cases, (3, 30), strict=True):
        gap = case.get("maximum_original_interval_seconds")
        terminal = case.get("terminal_host_qpc_ns")
        if (
            case.get("original_role") != "diagnostic"
            or type(case.get("duration_seconds")) is not int
            or case["duration_seconds"] != duration
            or case.get("parent_source_sha256")
            != LIFECYCLE_INPUTS["scripts/qualify_session_heartbeat_lifecycle_status_v1.py"]
            or case.get("child_source_sha256")
            != LIFECYCLE_INPUTS["scripts/record_ac_engine_ai_probe_lifecycle_status_v1.py"]
            or case.get("actor_trace_source_sha256")
            != R2_INPUTS["src/flybrain_drive/actor_cadence_trace_v1.py"]
            or case.get("source_candidate_sha256")
            != LIFECYCLE_INPUTS[
                "integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua"
            ]
            or case.get("frame_persistence_profile_sha256")
            != declaration["source_sha256"]["configs/ac-frame-host-v4-20261030.json"]
            or case.get("actor_cadence_profile_sha256")
            != declaration["source_sha256"][ACTOR_PROFILE]
            or case.get("camera_status_profile_sha256") != LIFECYCLE_INPUTS[CAMERA_STATUS_PROFILE]
            or case.get("camera_status_first_and_terminal_echo_verified") is not True
            or type(case.get("original_exit_code")) is not int
            or case["original_exit_code"] != 0
            or type(case.get("decisions")) is not int
            or case["decisions"] < 2
            or type(case.get("camera_wait_misses")) is not int
            or case["camera_wait_misses"] != 0
            or type(gap) not in (int, float)
            or not math.isfinite(gap)
            or not 0 < gap <= 0.08
            or type(terminal) is not int
            or terminal <= 0
            or any(
                case.get(k) is not True
                for k in (
                    "raw_seven_arrays_exact",
                    "all_camera_aids_OFF",
                    "actor_trace_complete",
                    "heartbeat_retirement_complete",
                )
            )
            or any(not is_sha(case.get(k)) for k in required_refs)
        ):
            raise ValueError(
                "Actual complete R2 diagnostic/raw/trace/clock/terminal gates required"
            )
        terminals.append(terminal)
    if (
        not terminals[0] < terminals[1]
        or type(value.get("last_diagnostic_terminal_host_qpc_ns")) is not int
        or value["last_diagnostic_terminal_host_qpc_ns"] != max(terminals)
        or any(
            len({case[key] for case in cases}) != 2
            for key in (
                "capture_manifest_sha256",
                "parent_result_sha256",
                "original_terminal_receipt_sha256",
            )
        )
    ):
        raise ValueError("Original last diagnostic terminal clock required")
    return value


def fresh_start_gate(raw, declaration, pipeline):
    """Privileged paused observer review only; no route enters the actor."""
    value = json.loads(raw)
    point = value.get("observer_host_qpc_ns_lower_bound")
    p = value.get("spline_position")
    if (
        sha_bytes(raw) != declaration["route_preflight_sha256"]
        or value.get("schema") != "flybrain.pedal_b2.fresh_start_review.v1"
        or value.get("nonce") != declaration["nonce"]
        or value.get("source_sha256") != declaration["source_sha256"]
        or value.get("engine_pid") != declaration["bindings"]["engine_pid"]
        or type(value.get("engine_pid")) is not int
        or value.get("generation") != declaration["bindings"]["generation"]
        or value.get("epoch") != declaration["bindings"]["epoch"]
        or type(value.get("epoch")) is not int
        or value.get("controls_ini_sha256") != declaration["controls_ini_sha256"]
        or value.get("host_clock") != "perf_counter_ns"
        or type(point) is not int
        or point <= pipeline["last_diagnostic_terminal_host_qpc_ns"]
        or value.get("after_diagnostic_terminal_receipt_sha256")
        != [c["original_terminal_receipt_sha256"] for c in pipeline["cases"]]
        or type(p) not in (int, float)
        or not math.isfinite(p)
        or not 0.49 <= p <= 0.54
        or abs(p - declaration["expected_start_spline"]) > 1e-4
        or type(value.get("lap_count")) is not int
        or value["lap_count"] != 0
        or value.get("actor_input") is not False
        or any(
            value.get(k) is not True
            for k in (
                "paused_fullbrake",
                "CSP_AIoff",
                "FREE_background",
                "no_owner",
                "native_graphics_advance_verified",
            )
        )
        or not is_sha(value.get("original_observer_result_sha256"))
    ):
        raise ValueError("Fresh nonce/source/input native start AFTER actual diagnostics required")
    return value


def external_binding_gate(declaration):
    pipeline = pipeline_gate(
        _review_file(
            declaration["pipeline_qualification_review_file"],
            declaration["pipeline_qualification_review_sha256"],
        ),
        declaration,
    )
    start = fresh_start_gate(
        _review_file(declaration["route_preflight_file"], declaration["route_preflight_sha256"]),
        declaration,
        pipeline,
    )
    pedal = pedal_gate(
        _review_file(
            declaration["native_pedal_continuity_review_file"],
            declaration["native_pedal_continuity_review_sha256"],
        ),
        declaration,
    )
    if pedal.get("qualified") is not True:
        raise ValueError("Actual external pedal continuity review required")
    return dict(
        pipeline_qualified_externally=True,
        fresh_start_qualified_externally=True,
        pedal_qualified_externally=True,
        route_actor_input=False,
        external_review_is_not_loader_measurement=True,
        fit_allowed=False,
        observer_start=start["spline_position"],
    )
