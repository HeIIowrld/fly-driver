"""Audit gas and brake gradient directions on the original fixed flight graph."""

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
            "one_pass": True,
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
    arrays = history.arrays
    values = {
        k: torch.tensor(arrays[k], device="cuda")
        for k in ("physical_command", "body", "body_valid", "racing_goals")
    }

    state = quiet.detach().clone()
    gradients, outputs, reports = [], [], []
    for window, start in enumerate(range(0, history.observations, 64)):
        end = min(start + 64, history.observations)
        predictions = []
        for index in range(start, end):
            if index:
                held = index - 1
                state = circuit.advance(
                    *(
                        values[k][held : held + 1]
                        for k in ("physical_command", "body", "body_valid", "racing_goals")
                    ),
                    state,
                    float(arrays["original_elapsed_seconds"][index]),
                    **base.kwargs,
                )[2]
            predictions.append(base.predict(state)["canonical"])
        canonical = torch.cat(predictions)
        outputs.append(canonical.detach().cpu().numpy())
        native = torch.tensor(arrays["native_pedals"][start:end], device="cuda")
        valid = torch.tensor(arrays["pedal_loss_valid"][start:end], device="cuda")
        supervision = native_pedal_supervision(
            native, valid, projection_policy="reject_unreachable"
        )
        mask = supervision["supervision_valid"]
        if not np.array_equal(mask.cpu().numpy(), history.effective_loss_valid[start:end]):
            raise ValueError("Original eligible loss mask must match")
        pedals = canonical[:, 1:]
        error = (supervision["canonical_lower"] - pedals).relu() + (
            pedals - supervision["canonical_upper"]
        ).relu()
        counts = mask.sum(0)
        losses = (error.square() * mask).sum(0) / counts.clamp_min(1)
        g = []
        for axis in range(2):
            grad = (
                torch.autograd.grad(
                    losses[axis], circuit.adapter.stop_delta, retain_graph=True, allow_unused=True
                )[0]
                if losses[axis].requires_grad
                else None
            )
            g.append(
                torch.zeros_like(circuit.adapter.stop_delta) if grad is None else grad.detach()
            )
        matrix = torch.stack(g).cpu().numpy()
        if not np.isfinite(matrix).all():
            raise ValueError("Finite actual graph gradients required")
        norms = np.linalg.norm(matrix, axis=1)
        cosine = (
            float(np.dot(matrix[0], matrix[1]) / (norms[0] * norms[1]))
            if np.all(norms > 0)
            else None
        )
        reports.append(
            {
                "window": window,
                "start": start,
                "end": end,
                "valid_counts": counts.detach().cpu().tolist(),
                "gas_gradient_norm": float(norms[0]),
                "brake_gradient_norm": float(norms[1]),
                "gas_brake_gradient_cosine": cosine,
            }
        )
        gradients.append(matrix)
        state = state.detach()
        if (window + 1) % 4 == 0:
            print(json.dumps({"window": window + 1}), flush=True)
    output = np.concatenate(outputs)
    reference = np.array(
        [
            r["prediction"]["canonical"][0]
            for r in original_result["fit"]["final_fixed_rollout"]["rows"]
        ]
    )
    parity = float(np.max(np.abs(output - reference)))
    if parity > 1e-6 or torch.any(circuit.adapter.stop_delta != 0):
        raise ValueError("Neutral baseline and adapter must remain exact")
    if (
        base.fixed_fingerprint() != expected_fixed
        or any(
            not torch.equal(parameter, frozen_pedal[name])
            for name, parameter in pedal_params.items()
        )
        or any(sha256(path) != expected for path, expected in pins.items())
    ):
        raise ValueError("Fixed model or source evidence changed")
    if any(sha256(path) != digest for path, digest in current_hashes.items()) or any(
        sha256(snapshot) != declaration["source_sha256"][path]
        for path, snapshot in snapshots.items()
    ):
        raise ValueError("Current or original collector snapshot changed")
    np.savez_compressed(run / "gradients.npz", gradients=np.stack(gradients), canonical=output)
    nonzero = [r for r in reports if r["gas_brake_gradient_cosine"] is not None]
    result = {
        "status": "completed_stop_vector_gradient_conflict_audit",
        "windows": reports,
        "both_nonzero_windows": len(nonzero),
        "conflicting_windows": sum(r["gas_brake_gradient_cosine"] < 0 for r in nonzero),
        "baseline_max_abs_difference": parity,
        "gradients_sha256": sha256(run / "gradients.npz"),
        "scope": "local TBPTT64 gradients; previous state sensitivity detached",
        "training_data_not_independent": True,
        "optimizer_updates": 0,
        "actor_deployed": False,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
