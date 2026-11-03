"""Replay a fixed stop candidate at counterfactual speeds on original body history."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stop_narrow_adapter import ExperimentalNarrowStopCircuit
from flybrain_drive.graph import sha256
from scripts.train_pedal_b2 import OriginalGraphAdapter


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-plan", required=True)
    parser.add_argument("--fit-result", required=True)
    parser.add_argument("--pedal-checkpoint", required=True)
    parser.add_argument("--stop-fit", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--body-replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan, fitted, pedal = read(args.fit_plan), read(args.fit_result), read(args.pedal_checkpoint)
    diagnostic, replay = Path(args.diagnostic), Path(args.body_replay)
    diag_result, body_result = read(diagnostic / "result.json"), read(replay / "result.json")
    stop_fit = Path(args.stop_fit)
    stop_result = read(stop_fit / "result.json")
    if (
        diag_result["role"] != "diagnostic"
        or diag_result["fit_allowed"] is not False
        or diag_result["actor_trace"]["decisions"] != 986
        or body_result["status"] != "completed"
        or sha256(replay / "virtual-body.npz") != body_result["virtual_body_sha256"]
        or fitted["status"] != "completed_one_pass_pedal_component_fit"
        or pedal["bindings"] != fitted["bindings"]
    ):
        raise ValueError("Original unpromoted diagnostic and original physical fit required")
    if (
        stop_result["status"] != "completed_fixed_stop_strength_selection"
        or sha256(stop_fit / "stop-task-adapter.pt") != stop_result["checkpoint_sha256"]
    ):
        raise ValueError("Original fixed stop-task candidate required")
    recording = diagnostic / "recording"
    event_path = recording / "events.jsonl"
    with event_path.open(encoding="utf8") as stream:
        events = [json.loads(line) for line in stream]
    with np.load(replay / "virtual-body.npz", allow_pickle=False) as a:
        body, valid = a["values"].copy(), a["valid"].copy()
        clocks, packets = a["decision_monotonic_seconds"].copy(), a["native_packet_ids"].copy()
    if (
        len(events) != 986
        or body.shape != (986, 21)
        or valid.dtype != bool
        or not valid[:, [4, 16]].all()
        or np.any((np.diff(clocks) <= 0) | (np.diff(clocks) > 0.08))
    ):
        raise ValueError("Complete original diagnostic history and intervals required")
    commands = []
    for i, event in enumerate(events):
        task = event["external_task_at_receipt"]
        native = event["native"]
        if (
            event["decision_monotonic_seconds"] != clocks[i]
            or native["packet_id"] != packets[i]
            or task["schema"] != "flybrain.external_speed_task.v1"
            or task["held_external_instruction"] is not True
            or task["actor_input"] is not True
            or not 0
            <= task["observed_at_host_seconds"]
            <= native["received_monotonic_seconds"]
            <= clocks[i]
            or not 0 <= task["desired_speed_m_s"] <= 100 / 3.6 + 1e-5
        ):
            raise ValueError("Original sensor clocks and held external task required")
        commands.append([0.0, task["desired_speed_m_s"]])
    paths = [
        args.fit_plan,
        args.fit_result,
        args.pedal_checkpoint,
        str(diagnostic / "result.json"),
        str(event_path),
        str(replay / "result.json"),
        str(replay / "virtual-body.npz"),
    ]
    paths.extend([str(stop_fit / "result.json"), str(stop_fit / "stop-task-adapter.pt")])
    pins = {path: sha256(path) for path in paths}
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "narrow_gate_source_sha256": sha256("src/flybrain_drive/flight_stop_narrow_adapter.py"),
            "source_hashes": pins,
            "role": "original_1903_diagnostic_only",
            "fit_allowed": False,
            "optimizer_updates": 0,
            "student_deployed": False,
            "scope": "fixed artificial goal speeds, not actual driving",
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    base = OriginalGraphAdapter(plan, lambda: None)
    with torch.no_grad():
        for name, parameter in base.parameters().items():
            parameter.copy_(torch.tensor(pedal["values"][name], device=parameter.device))
        quiet = base.initialize()
    if base.fixed_fingerprint() != fitted["fit"]["final_fixed_rollout"]["fixed_fingerprint"]:
        raise ValueError("Original fixed flight graph/readout changed")
    base.model.requires_grad_(False)
    saved = torch.load(stop_fit / "stop-task-adapter.pt", map_location="cpu", weights_only=True)
    if saved["source_pedal_checkpoint_sha256"] != sha256(args.pedal_checkpoint):
        raise ValueError("Original pedal model binding changed")
    wrapped = ExperimentalNarrowStopCircuit(base.model).to("cuda").eval()
    wrapped.adapter.load_state_dict(saved["model"], strict=True)
    wrapped.requires_grad_(False)
    body_t = torch.tensor(body, dtype=torch.float32, device="cuda")
    valid_t = torch.tensor(valid, dtype=torch.bool, device="cuda")
    goals = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device="cuda").repeat(986, 1)
    targets_m_s = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 60 / 3.6]
    with torch.no_grad():
        drives = []
        for target in targets_m_s:
            inputs = torch.zeros((986, 2), dtype=torch.float32, device="cuda")
            inputs[:, 1] = target
            drives.append(wrapped.drive(inputs, body_t, valid_t, goals, **base.kwargs))
        state = quiet.detach().expand(len(targets_m_s), -1).clone()
        output = []
        nominal = base.model.backbone.spec["dynamics"]["dt"]
        for i in range(986):
            if i:
                elapsed = float(clocks[i] - clocks[i - 1])
                step = torch.cat([d[i - 1 : i] for d in drives], 0)
                count = int(np.floor(elapsed / nominal))
                for _ in range(count):
                    state = base.model.backbone._advance_elapsed_tensor(step, state, nominal)
                tail = elapsed - count * nominal
                if tail > 1e-12:
                    state = base.model.backbone._advance_elapsed_tensor(step, state, tail)
            output.append(base.predict(state)["canonical"].cpu().numpy())
    predictions = np.stack(output)
    if not np.isfinite(predictions).all():
        raise ValueError("Finite original-history predictions required")
    # Current actual body state only selects diagnostic examples, never feeds a target label.
    stationary = valid[:, 16] & (body[:, 16] < 0.2)
    if not stationary.any():
        raise ValueError("Original low-speed body examples required")
    reports = []
    for j, target in enumerate(targets_m_s):
        pedal = predictions[stationary, j, 1:]
        reports.append(
            {
                "fixed_target_m_s": target,
                "low_speed_rows": int(stationary.sum()),
                "gas_mean": float(pedal[:, 0].mean()),
                "brake_mean": float(pedal[:, 1].mean()),
                "brake_over_0_1_count": int((pedal[:, 1] > 0.1).sum()),
                "simultaneous_positive_count": int(((pedal[:, 0] > 0) & (pedal[:, 1] > 0)).sum()),
            }
        )
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Original source changed during diagnostic")
    np.savez_compressed(
        run / "predictions.npz",
        canonical=predictions,
        stationary=stationary,
        targets_m_s=np.asarray(targets_m_s),
    )
    write_json(
        run / "result.json",
        {
            "status": "completed_counterfactual_target_speed_audit",
            "reports": reports,
            "predictions_sha256": sha256(run / "predictions.npz"),
            "original_body_recording_with_counterfactual_commands": True,
            "vehicle_closed_loop_or_teacher_ground_truth": False,
            "causal_time_or_physiology_verified": False,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
