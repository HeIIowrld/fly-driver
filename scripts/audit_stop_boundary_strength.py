"""Audit bounded stop signal strength in the original flight graph, no optimization."""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_pedal_learning_v2 import native_pedal_supervision
from flybrain_drive.flight_stop_vector_adapter import ExperimentalStopVectorCircuit
from flybrain_drive.graph import sha256
from flybrain_drive.pedal_plan_b2 import declared_loader_pedal_review
from flybrain_drive.pedal_tbptt_engine_v1 import whole_history
from scripts.train_pedal_b2 import OriginalGraphAdapter


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-plan", required=True)
    parser.add_argument("--fit-result", required=True)
    parser.add_argument("--pedal-checkpoint", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan, original_result, pedal = (
        read(args.fit_plan),
        read(args.fit_result),
        read(args.pedal_checkpoint),
    )
    qualification = read(Path(args.qualification) / "result.json")
    if (
        qualification["status"] != "qualified_neutral_body_adapter_actual_graph"
        or qualification["original_body_observations"] != 986
        or original_result["status"] != "completed_one_pass_pedal_component_fit"
        or original_result["optimizer_updates"] != 123
        or pedal["bindings"] != original_result["bindings"]
    ):
        raise ValueError("Qualified new boundary and original completed DNg fit required")
    if sha256(plan["loader_binding"]["source_path"]) != plan["loader_binding"]["source_sha256"]:
        raise ValueError("Original complete-history loader changed")
    loader = importlib.import_module(plan["loader_binding"]["module"])
    capture = Path(plan["capture_root"])
    declaration = read(capture / "original-train-declaration.json")
    # Historical collection sources changed after capture. Verify exact retained snapshots
    # for these two source-evidence paths; all other loader checks remain unchanged.
    snapshots = {
        "integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua": Path(
            "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/"
            "20261030-airflow-diagnostic-install/installed-before.lua"
        ),
        "src/flybrain_drive/ac_csp_vision.py": Path(
            "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/"
            "20261030-t1808-frame-persistence-profile-preparation/revision2/"
            "source-snapshot/src/flybrain_drive/ac_csp_vision.py"
        ),
    }
    for original, snapshot in snapshots.items():
        if sha256(snapshot) != declaration["source_sha256"][original]:
            raise ValueError("Exact original captured collector source snapshot required")
    current_hashes = {path: sha256(path) for path in snapshots}
    original_sha_file = loader.sha_file

    def historical_source_hash(path):
        name = str(path).replace("\\", "/")
        return original_sha_file(snapshots[name]) if name in snapshots else original_sha_file(path)

    loader.sha_file = historical_source_hash
    try:
        loaded = getattr(loader, plan["loader_binding"]["function"])(
            plan["capture_root"],
            expected_capture_manifest_sha256=plan["capture_manifest_sha256"],
            pedal_review_bytes=declared_loader_pedal_review(plan),
        )
    finally:
        loader.sha_file = original_sha_file
    if any(sha256(path) != digest for path, digest in current_hashes.items()):
        raise ValueError("Current project sources changed during offline historical loading")
    if (
        loaded.get("fit_allowed") is not False
        or loaded.get("all_states_retained") is not True
        or loaded.get("source_clock_schedule_valid") is not True
        or loaded.get("physical_body_interface_valid") is not True
    ):
        raise ValueError("Original complete-history loader evidence required")
    history = whole_history(loaded["arrays"], expected_fingerprints=plan["history"]["fingerprints"])
    if history.observations != 986 or history.nonempty_windows != 123:
        raise ValueError("Original unretimed complete TRAIN history required")
    pins = {
        path: sha256(path)
        for path in (
            args.fit_plan,
            args.fit_result,
            args.pedal_checkpoint,
            str(Path(args.qualification) / "result.json"),
            plan["loader_binding"]["source_path"],
        )
    }
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "adapter_source_sha256": sha256("src/flybrain_drive/flight_stop_vector_adapter.py"),
            "historical_source_snapshots": {
                original: {
                    "path": str(snapshot),
                    "sha256": sha256(snapshot),
                    "current_sha256": current_hashes[original],
                }
                for original, snapshot in snapshots.items()
            },
            "source_scope": "offline original capture evidence; not current runtime qualification",
            "history_fingerprints": history.fingerprints,
            "training_role": "original_1909_TRAIN",
            "optimizer": None,
            "tbptt_observations": 64,
            "scope": "fixed boundary strength comparison, evaluation only",
            "candidate": args.candidate,
            "fixed_scale_factors": [0.0, 1.0, 8.0, 64.0, 512.0],
            "update_schedule": "none",
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    base = OriginalGraphAdapter(plan, lambda: None)
    pedal_params = base.parameters()
    with torch.no_grad():
        for name, parameter in pedal_params.items():
            parameter.copy_(torch.tensor(pedal["values"][name], device=parameter.device))
        quiet = base.initialize()
    frozen_pedal = {name: parameter.detach().clone() for name, parameter in pedal_params.items()}
    expected_fixed = original_result["fit"]["final_fixed_rollout"]["fixed_fingerprint"]
    if base.fixed_fingerprint() != expected_fixed:
        raise ValueError("Fixed original flight graph and readout changed")
    base.model.requires_grad_(False)
    circuit = ExperimentalStopVectorCircuit(base.model).to("cuda").eval()
    candidate = Path(args.candidate)
    candidate_result = read(candidate / "result.json")
    checkpoint = candidate / "stop-task-adapter.pt"
    if (
        candidate_result["status"] != "completed_fixed_episode_sustained_brake_fit"
        or sha256(checkpoint) != candidate_result["checkpoint_sha256"]
    ):
        raise ValueError("Pinned completed original stop boundary candidate required")
    loaded_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if loaded_checkpoint["source_pedal_checkpoint_sha256"] != sha256(args.pedal_checkpoint):
        raise ValueError("Exact original pedal source binding required")
    weights = loaded_checkpoint["model"]["stop_delta"].to("cuda")
    factors = [0.0, 1.0, 8.0, 64.0, 512.0]
    arrays = history.arrays
    inputs = {
        key: torch.tensor(arrays[key], device="cuda")
        for key in ("physical_command", "body", "body_valid", "racing_goals")
    }
    drives = []
    with torch.no_grad():
        for factor in factors:
            circuit.adapter.stop_delta.copy_(weights * factor)
            drives.append(
                circuit.drive(
                    *(
                        inputs[k]
                        for k in ("physical_command", "body", "body_valid", "racing_goals")
                    ),
                    **base.kwargs,
                )
            )
        state = quiet.detach().expand(len(factors), -1).clone()
        predictions = []
        nominal = base.model.backbone.spec["dynamics"]["dt"]
        for i in range(history.observations):
            if i:
                elapsed = float(arrays["original_elapsed_seconds"][i])
                drive = torch.cat([d[i - 1 : i] for d in drives], dim=0)
                count = int(np.floor(elapsed / nominal))
                for _ in range(count):
                    state = base.model.backbone._advance_elapsed_tensor(drive, state, nominal)
                tail = elapsed - count * nominal
                if tail > 1e-12:
                    state = base.model.backbone._advance_elapsed_tensor(drive, state, tail)
            predictions.append(base.predict(state)["canonical"].cpu().numpy())
    predictions = np.stack(predictions)
    reference = np.asarray(
        [
            row["prediction"]["canonical"][0]
            for row in original_result["fit"]["final_fixed_rollout"]["rows"]
        ]
    )
    parity = float(np.max(np.abs(predictions[:, 0, :] - reference)))
    if parity > 1e-6:
        raise ValueError("Original zero-strength replay differs")
    native = torch.tensor(arrays["native_pedals"], device="cuda")
    valid = torch.tensor(arrays["pedal_loss_valid"], device="cuda")
    supervision = native_pedal_supervision(native, valid, projection_policy="reject_unreachable")
    keep = supervision["supervision_valid"]
    if not np.array_equal(keep.cpu().numpy(), history.effective_loss_valid):
        raise ValueError("Original loss masks changed")
    ages = np.zeros(history.observations)
    start_time = None
    for i, brake in enumerate(arrays["native_pedals"][:, 1]):
        if brake == 1:
            if start_time is None:
                start_time = arrays["decision_clock"][i]
            ages[i] = arrays["decision_clock"][i] - start_time
        else:
            start_time = None
    sustained = torch.tensor((ages >= 3) & (arrays["native_pedals"][:, 1] == 1), device="cuda")
    reports = []
    for j, factor in enumerate(factors):
        out = torch.tensor(predictions[:, j], device="cuda")
        pedals = out[:, 1:]
        error = (supervision["canonical_lower"] - pedals).relu() + (
            pedals - supervision["canonical_upper"]
        ).relu()
        count = keep.sum(0)
        mse = (error.square() * keep).sum(0) / count.clamp_min(1)
        full = (native[:, 1] == 1) & keep[:, 1]
        hold = full & sustained
        reports.append(
            {
                "factor": factor,
                "pedal_mse": mse.cpu().tolist(),
                "full_brake_mse": float(error[full, 1].square().mean()),
                "hold_rows": int(hold.sum()),
                "hold_brake_mean": float(out[hold, 2].mean()),
                "hold_brake_min": float(out[hold, 2].min()),
                "hold_brake_max": float(out[hold, 2].max()),
                "steer_mean_abs_change": float(
                    np.abs(predictions[:, j, 0] - predictions[:, 0, 0]).mean()
                ),
            }
        )
    if (
        base.fixed_fingerprint() != expected_fixed
        or sha256(checkpoint) != candidate_result["checkpoint_sha256"]
        or any(sha256(path) != expected for path, expected in pins.items())
        or any(not torch.equal(parameter, frozen_pedal[name])
               for name, parameter in pedal_params.items())
    ):
        raise ValueError("Original graph, checkpoint or source changed")
    np.savez_compressed(run / "predictions.npz", canonical=predictions, full_age=ages)
    result = {
        "status": "completed_stop_boundary_strength_audit",
        "reports": reports,
        "baseline_max_abs_error": parity,
        "candidate_checkpoint_sha256": sha256(checkpoint),
        "predictions_sha256": sha256(run / "predictions.npz"),
        "posthoc_training_history_only": True,
        "not_physiological_tuning": True,
        "actor_deployed": False,
        "optimizer_updates": 0,
        "wall_seconds": time.perf_counter()-started,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
