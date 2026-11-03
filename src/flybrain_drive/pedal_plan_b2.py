"""Stdlib-only draft/seal gate. Unsealed plans fail before data/model imports."""

import hashlib
import json
import re
from pathlib import Path

DRAFT_PATH = "configs/flight-pedal-first-train-fit-draft-20261030.json"
DRAFT_SHA256 = "02a016f7bee858ede9c7e7d800e611a11e77975c551bfef043206d5dd05761fa"
SCHEMA = "flybrain.first_pedal_b2_execution_plan.v1"
EVIDENCE_NAMES = frozenset(
    {
        "raw_seven_arrays",
        "samecopy_authority",
        "heartbeat_full_bytes",
        "host_v4_profile",
        "whole_route",
        "nonce_pedal_calibration",
        "whole_history_loader",
        "matched_graph_gradient_audit",
    }
)
NEW_SOURCES = (
    "src/flybrain_drive/pedal_tbptt_engine_v1.py",
    "src/flybrain_drive/flight_pedal_train_plan_v1.py",
    "scripts/train_flight_pedal_first_history_v1.py",
    "scripts/prepare_flight_pedal_first_history_v1.py",
    "src/flybrain_drive/pedal_plan_b2.py",
    "scripts/train_pedal_b2.py",
)
ARRAY_FIELDS = frozenset(
    {
        "decision_clock",
        "native_receipt_clock",
        "native_packet",
        "body",
        "body_valid",
        "physical_command",
        "racing_goals",
        "native_pedals",
        "pedal_loss_valid",
        "original_elapsed_seconds",
        "previous_held_row_index",
    }
)
ADAPTER_HELPER = "scripts/qualify_flight_pedal_graph_gpu_v2.py"
ADAPTER_HELPER_SHA = "5a960f6fd61789f1ab93cfaba3ce433e85d5f900325edaa78fa3038e7c1779ee"
RUNTIME = "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/"
KNOWN_GPU_RESULT = (
    RUNTIME + "20261030-t1828-matched-original-graph-pedal-gradient/actual-v2/result.json"
)
KNOWN_GPU_RESULT_SHA = "f97e168720c331216c3f0cb46328a9710a5d02980f0c911b1d377695ba0c85ba"
KNOWN_GPU_QUIET = "36235827daddfde774c8bc6bd9b2183ba6a1391470c8a95d85ca4f221bc1edbb"
ORIGINAL_PLAN = "configs/flight-pedal-original-graph-cpu-20261030.json"
ORIGINAL_PLAN_SHA = "d53399278ce50976fbef938a3a5154eef1efd025d72ac911b93af7319ff7b262"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def preparation_template(draft_bytes):
    if hashlib.sha256(draft_bytes).hexdigest() != DRAFT_SHA256:
        raise ValueError("Exact reviewed two-stage engineering draft required")
    protocol = json.loads(draft_bytes)
    prepared = dict(
        schema=SCHEMA,
        status="unsealed_source_preparation_not_executable",
        sealed=False,
        model_execution_authorized=False,
        fit_allowed=False,
        execution_stage=None,
        protocol_sha256=DRAFT_SHA256,
        protocol=protocol,
        output=None,
        capture_root=None,
        capture_nonce=None,
        capture_manifest_sha256=None,
        original_train_declaration_sha256=None,
        source_closure=dict(path=None, sha256=None),
        loader_binding=dict(
            module=None,
            function=None,
            source_path=None,
            source_sha256=None,
            result_path=None,
            result_sha256=None,
            result_schema=None,
            result_status=None,
        ),
        admission=dict(path=None, sha256=None),
        loader_pedal_review=dict(file=None, sha256=None),
        evidence={
            name: dict(path=None, sha256=None, schema=None, status=None)
            for name in sorted(EVIDENCE_NAMES)
        },
        history=dict(
            fingerprints=None,
            observations=None,
            windows=None,
            windows_with_supervised_cells=None,
            exact_optimizer_update_budget=None,
        ),
        model=dict(
            spec="configs/flight-circuit-separated-v5-20261029.json",
            checkpoint="runs/flight-separated-command-probe-20261029-t1081/checkpoint.pt",
            cpu_reference=RUNTIME + "20261030-t1734-cpu-mn-anchor-qualification/reference.json",
            cpu_readout=RUNTIME
            + "20261030-t1741-matched-gpu-mn-qualification/cpu-readout-state.json",
            matched_graph_result=KNOWN_GPU_RESULT,
            matched_graph_result_sha256=KNOWN_GPU_RESULT_SHA,
            expected_GPU_quiet_sha256=KNOWN_GPU_QUIET,
        ),
        gradient_qualification=dict(
            path=None,
            sha256=None,
            original_terminal_receipt_path=None,
            original_terminal_receipt_sha256=None,
        ),
        unresolved=[
            "FutureV4 whole-history loader module/source/result is unknown.",
            "Exact new prospective TRAIN capture and all source/admission reviews are absent.",
            "Actual TRAIN window-gradient rank2 must be separately qualified before fit.",
            "Preparation does not grant model execution or fitting permission.",
        ],
    )
    prepared["evidence"]["matched_graph_gradient_audit"] = dict(
        path=RUNTIME + "20261030-t1834-matched-graph-gpu-independent/result.json",
        sha256="b8f755cb924040896f2f598ac4ecfa6549a74761054ec38db491efa4e6937c43",
        schema="flybrain.flight.matched_graph_GPU_saved_metadata_audit.v1",
        status="passed_scoped_numerical_evidence_with_nonzero_allocator_cleanup",
    )
    return prepared


def _artifact(binding, label):
    if (
        not isinstance(binding, dict)
        or not isinstance(binding.get("path"), str)
        or not is_sha(binding.get("sha256"))
    ):
        raise ValueError("Exact sealed artifact binding missing: " + label)
    if sha(binding["path"]) != binding["sha256"]:
        raise ValueError("Sealed artifact bytes differ: " + label)
    return json.loads(Path(binding["path"]).read_bytes())


def declared_loader_pedal_review(plan, declaration=None):
    """Return exact original declared bytes, separately from typed review evidence."""
    binding = plan.get("loader_pedal_review")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"file", "sha256"}
        or not isinstance(binding.get("file"), str)
        or not is_sha(binding.get("sha256"))
    ):
        raise ValueError("Exact original loader pedal review file/SHA required")
    raw = Path(binding["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("Original loader pedal review bytes differ")
    if declaration is not None and (
        binding["sha256"] != declaration.get("native_pedal_continuity_review_sha256")
        or Path(binding["file"]).resolve()
        != Path(declaration.get("native_pedal_continuity_review_file") or "").resolve()
    ):
        raise ValueError("Loader pedal review differs from original pre-capture declaration")
    return raw


def validate_before_model(plan, *, stage):
    """No model/data/device import. Hashes payloads only AFTER explicit seal gates."""
    if (
        plan.get("schema") != SCHEMA
        or plan.get("sealed") is not True
        or plan.get("model_execution_authorized") is not True
    ):
        raise ValueError("UNSEALED: no data/model/device work authorized")
    if stage not in {"qualify_gradients", "fit"} or plan.get("execution_stage") != stage:
        raise ValueError("Separate exact gradient-only or fit execution stage required")
    if plan.get("fit_allowed") is not (stage == "fit"):
        raise ValueError("Gradient qualification is fitfalse; fit requires separate authorization")
    encoded = json.dumps(plan.get("protocol"), ensure_ascii=False, sort_keys=True)
    draft = Path(DRAFT_PATH).read_bytes()
    if (
        sha(DRAFT_PATH) != DRAFT_SHA256
        or plan.get("protocol_sha256") != DRAFT_SHA256
        or encoded != json.dumps(json.loads(draft), ensure_ascii=False, sort_keys=True)
    ):
        raise ValueError("Fixed reviewed engineering protocol differs")
    output = Path(plan.get("output") or "")
    if output.drive.upper() != "C:" or output.exists() or output.name in {"", ".", ".."}:
        raise ValueError("A new unique localC execution output is required")
    if (
        not isinstance(plan.get("capture_root"), str)
        or Path(plan["capture_root"]).drive.upper() != "C:"
        or not isinstance(plan.get("capture_nonce"), str)
        or not re.fullmatch("[0-9a-f]{32}", plan["capture_nonce"])
        or not is_sha(plan.get("capture_manifest_sha256"))
        or not is_sha(plan.get("original_train_declaration_sha256"))
    ):
        raise ValueError("Exact new original TRAIN capture/nonce required; no role promotion")
    loader = plan.get("loader_binding", {})
    if (
        not isinstance(loader.get("module"), str)
        or not re.fullmatch("flybrain_drive\\.[a-z][a-z0-9_]*", loader["module"])
        or loader["module"] == "flybrain_drive.pedal_train_loader"
        or not isinstance(loader.get("result_schema"), str)
        or not loader["result_schema"]
        or not isinstance(loader.get("result_status"), str)
        or not loader["result_status"]
        or not isinstance(loader.get("function"), str)
        or not re.fullmatch("[a-z][a-z0-9_]*", loader["function"])
        or not is_sha(loader.get("source_sha256"))
        or not isinstance(loader.get("source_path"), str)
        or Path(loader["source_path"]).resolve()
        != Path("src/" + loader["module"].replace(".", "/") + ".py").resolve()
    ):
        raise ValueError("FutureV4 loader identity/source/callable is unsealed or inconsistent")
    if sha(loader["source_path"]) != loader["source_sha256"]:
        raise ValueError("Exact admittedV4 loader source differs")
    result_binding = dict(path=loader.get("result_path"), sha256=loader.get("result_sha256"))
    loader_result = _artifact(result_binding, "whole-history loader result")
    if loader_result.get("schema") != loader.get("result_schema") or loader_result.get(
        "status"
    ) != loader.get("result_status"):
        raise ValueError("Exact independently reviewedV4 loader result schema/status required")
    closure = _artifact(plan.get("source_closure"), "source closure")
    sources = closure.get("source_sha256", {})
    if (
        closure.get("schema") != "flybrain.first_pedal_training_source_closure.v1"
        or not isinstance(sources, dict)
        or not all(is_sha(v) for v in sources.values())
        or not set(NEW_SOURCES) <= sources.keys()
        or sources.get(loader["source_path"]) != loader["source_sha256"]
        or sources.get(ADAPTER_HELPER) != ADAPTER_HELPER_SHA
    ):
        raise ValueError("Complete exact trainer/originalgraph/futureV4 source closure required")
    if sha(ORIGINAL_PLAN) != ORIGINAL_PLAN_SHA:
        raise ValueError("Original model source registration differs")
    original = json.loads(Path(ORIGINAL_PLAN).read_bytes())["input_sha256"]
    if any(sources.get(k) != value for k, value in original.items()):
        raise ValueError("Original31 source/checkpoint/graph/reference bindings must be retained")
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError("Actual source/payload hash differs: " + path)
    evidence = plan.get("evidence", {})
    if set(evidence) != EVIDENCE_NAMES:
        raise ValueError("All exact new-capture independent evidence required")
    evidence_values = {}
    for name, binding in evidence.items():
        value = _artifact(binding, name)
        evidence_values[name] = value
        if (
            not isinstance(binding.get("schema"), str)
            or not isinstance(binding.get("status"), str)
            or value.get("schema") != binding["schema"]
            or value.get("status") != binding["status"]
        ):
            raise ValueError("Exact reviewed evidence schema/status differs: " + name)
    capture = Path(plan["capture_root"])
    if sha(capture / "manifest.json") != plan["capture_manifest_sha256"]:
        raise ValueError("Exact original TRAIN capture manifest bytes differ")
    captured = json.loads((capture / "manifest.json").read_bytes())
    if (
        captured.get("config", {}).get("role") != "train"
        or captured["config"].get("train_nonce") != plan["capture_nonce"]
        or captured["config"].get("train_declaration_sha256")
        != plan["original_train_declaration_sha256"]
        or sha(capture / "original-train-declaration.json")
        != plan["original_train_declaration_sha256"]
    ):
        raise ValueError("No original TRAIN role/nonce/declaration; never promote other roles")
    original_pedal_review = declared_loader_pedal_review(
        plan, captured["config"]["original_train_declaration"]
    )
    typed_pedal_review = evidence_values["nonce_pedal_calibration"]
    if (
        json.loads(original_pedal_review).get("schema")
        != "flybrain.pedal_calibration_continuity_review.v1"
        or typed_pedal_review.get("loader_pedal_review") != plan["loader_pedal_review"]
        or typed_pedal_review.get("capture_nonce") != plan["capture_nonce"]
    ):
        raise ValueError("Typed pedal review must reference exact original nonce-scoped bytes")
    admission = _artifact(plan.get("admission"), "newTRAIN whole-history admission")
    history = plan.get("history", {})
    if (
        admission.get("schema") != "flybrain.first_pedal_new_train_admission.v1"
        or admission.get("status") != "qualified_exact_original_train_whole_history"
        or admission.get("original_source_role") != "train"
        or admission.get("capture_nonce") != plan["capture_nonce"]
        or admission.get("capture_manifest_sha256") != plan["capture_manifest_sha256"]
        or admission.get("original_train_declaration_sha256")
        != plan["original_train_declaration_sha256"]
        or admission.get("source_closure_sha256") != plan["source_closure"]["sha256"]
        or admission.get("loader_source_sha256") != loader["source_sha256"]
        or admission.get("evidence_sha256") != {k: v["sha256"] for k, v in evidence.items()}
        or admission.get("history") != history
        or admission.get("all_original_states_retained") is not True
        or admission.get("pedal_only") is not True
        or admission.get("steering_qualified") is not False
        or admission.get("yaw_policy") != "prospectively_declared_zero_component"
    ):
        raise ValueError("No exact newTRAIN/source/wholeclock/route/pedal-only admission")
    n, windows, eligible = (
        history.get(k) for k in ("observations", "windows", "windows_with_supervised_cells")
    )
    if (
        type(n) is not int
        or not 2 <= n <= 1200
        or type(windows) is not int
        or windows != (n + 7) // 8
        or type(eligible) is not int
        or not 0 <= eligible <= windows
        or history.get("exact_optimizer_update_budget") != eligible
        or not isinstance(history.get("fingerprints"), dict)
        or set(history["fingerprints"]) != ARRAY_FIELDS
        or not all(is_sha(s) for s in history["fingerprints"].values())
    ):
        raise ValueError("Exact complete history/window/nonempty update budget required")
    model = plan.get("model", {})
    if (
        model.get("spec") != plan["protocol"]["original_graph"]["spec"]
        or model.get("checkpoint") != plan["protocol"]["original_graph"]["checkpoint"]
        or sources.get(model.get("checkpoint"))
        != plan["protocol"]["original_graph"]["checkpoint_sha256"]
        or not is_sha(model.get("expected_GPU_quiet_sha256"))
    ):
        raise ValueError("Unchanged original model and actualGPU quiet binding required")
    matched = _artifact(
        dict(
            path=model.get("matched_graph_result"), sha256=model.get("matched_graph_result_sha256")
        ),
        "actual matched graphGPU proof",
    )
    if (
        matched.get("status") != "completed_matched_original_graph_pedal_gradient_diagnostic"
        or matched.get("scoped_qualification_passed") is not True
        or matched.get("optimizer_updates") != 0
        or set(matched.get("actual_GPU_quiet_hashes", [])) != {model["expected_GPU_quiet_sha256"]}
    ):
        raise ValueError("Actual matchedCPU/GPU originalgraph diagnostic binding differs")
    audit = evidence_values["matched_graph_gradient_audit"]
    if (
        audit.get("schema") != "flybrain.flight.matched_graph_GPU_saved_metadata_audit.v1"
        or audit.get("source_result_sha256") != model["matched_graph_result_sha256"]
        or audit.get("actual_tool_terminal_basis", {}).get("exit_code") != 0
    ):
        raise ValueError("Actual matched graph proof plus original terminal0 audit required")
    if stage == "fit":
        validate_gradient_certificate(plan)
    return dict(admission=admission, source_closure=closure, matched_graph=matched)


def validate_gradient_certificate(plan):
    """Runs before optimizer construction and, for fit CLI, before model loading."""
    binding = plan.get("gradient_qualification", {})
    cert = _artifact(binding, "actual wholeTRAIN window-gradient qualification")
    terminal = _artifact(
        dict(
            path=binding.get("original_terminal_receipt_path"),
            sha256=binding.get("original_terminal_receipt_sha256"),
        ),
        "qualification original terminal",
    )
    context = cert.get("bindings", {})
    if (
        cert.get("status") != "completed_new_train_window_gradient_qualification"
        or cert.get("fit_allowed") is not False
        or cert.get("optimizer_updates") != 0
        or cert.get("qualification", {}).get("rank2_passed") is not True
        or cert.get("qualification", {}).get("original_graph") is not True
        or cert.get("qualification", {}).get("optimizer_updates") != 0
        or cert.get("qualification", {}).get("observations") != plan["history"]["observations"]
        or cert.get("source_drift_count") != 0
        or cert.get("program_sha256") != sha("scripts/train_pedal_b2.py")
        or context.get("capture_manifest_sha256") != plan["capture_manifest_sha256"]
        or context.get("source_closure_sha256") != plan["source_closure"]["sha256"]
        or context.get("history") != plan["history"]
        or context.get("model_binding") != plan["model"]
        or context.get("protocol_sha256") != plan["protocol_sha256"]
        or context.get("capture_nonce") != plan["capture_nonce"]
        or context.get("loader_source_sha256") != plan["loader_binding"]["source_sha256"]
        or context.get("loader_pedal_review") != plan["loader_pedal_review"]
        or terminal.get("schema") != "flybrain.original_process_terminal_receipt.v1"
        or terminal.get("status") != "original_process_terminal"
        or terminal.get("exit_code") != 0
        or terminal.get("result_sha256") != binding["sha256"]
    ):
        raise ValueError("Actual same-model/data/source rank2 certificate and terminal0 required")
    return cert
