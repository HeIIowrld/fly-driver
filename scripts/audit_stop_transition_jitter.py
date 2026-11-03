"""Compare stop-go and goal jitter on an original body history without actuation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stop_narrow_adapter import ExperimentalNarrowStopCircuit
from flybrain_drive.flight_stop_vector_adapter import ExperimentalStopVectorCircuit
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
            "source_hashes": pins,
            "role": "original_1903_diagnostic_only",
            "fit_allowed": False,
            "optimizer_updates": 0,
            "student_deployed": False,
            "scope": "synthetic stop-go and jitter on recorded body history; no driving",
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
    wrapped = ExperimentalStopVectorCircuit(base.model).to("cuda").eval()
    wrapped.adapter.load_state_dict(saved["model"], strict=True)
    wrapped.requires_grad_(False)
    body_t = torch.tensor(body, dtype=torch.float32, device="cuda")
    valid_t = torch.tensor(valid, dtype=torch.bool, device="cuda")
    goals = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device="cuda").repeat(986, 1)
    age_root = Path(
        "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/"
        "20261030-stop-strength-512-1903-diagnostic"
    )
    age_result = read(age_root / "result.json")
    age_file = age_root / "predictions.npz"
    if sha256(age_file) != age_result["predictions_sha256"]:
        raise ValueError("Exact independently preserved diagnostic age required")
    with np.load(age_file, allow_pickle=False) as age_data:
        age = age_data["full_brake_age"].copy()
    indices = np.flatnonzero(age >= 3.0)
    if not len(indices):
        raise ValueError("Original sustained stop required")
    switch = int(indices[0])
    elapsed = clocks - clocks[switch]
    region = (elapsed >= 0) & (elapsed < 4.0) & (body[:, 16] < 0.2)
    if int(region.sum()) < 50:
        raise ValueError("Sufficient original near-stationary evaluation history required")
    definitions = [
        ("wide_step_0p5", "wide", "step05"),
        ("narrow_step_0p5", "narrow", "step05"),
        ("wide_step_1", "wide", "step1"),
        ("narrow_step_1", "narrow", "step1"),
        ("wide_jitter", "wide", "jitter"),
        ("narrow_jitter", "narrow", "jitter"),
    ]
    narrow = ExperimentalNarrowStopCircuit(base.model).to("cuda").eval()
    narrow.adapter.load_state_dict(saved["model"], strict=True)
    narrow.requires_grad_(False)
    schedules = []
    for _, _, scenario in definitions:
        task = np.zeros(986, dtype=np.float32)
        if scenario == "step05":
            task[elapsed >= 0] = 0.5
        elif scenario == "step1":
            task[elapsed >= 0] = 1.0
        else:
            active = elapsed >= 0
            task[active] = (np.floor(elapsed[active] / 0.4).astype(int) % 2) * 0.05
        schedules.append(task)
    with torch.no_grad():
        drives = []
        for (_, family, _), schedule in zip(definitions, schedules, strict=True):
            cmd = torch.zeros((986, 2), dtype=torch.float32, device="cuda")
            cmd[:, 1] = torch.tensor(schedule, device="cuda")
            circuit = wrapped if family == "wide" else narrow
            drives.append(circuit.drive(cmd, body_t, valid_t, goals, **base.kwargs))
        state = quiet.detach().expand(len(definitions), -1).clone()
        output = []
        nominal = base.model.backbone.spec["dynamics"]["dt"]
        for i in range(986):
            if i:
                dt = float(clocks[i] - clocks[i - 1])
                step = torch.cat([d[i - 1 : i] for d in drives], 0)
                for _ in range(int(np.floor(dt / nominal))):
                    state = base.model.backbone._advance_elapsed_tensor(step, state, nominal)
                tail = dt - int(np.floor(dt / nominal)) * nominal
                if tail > 1e-12:
                    state = base.model.backbone._advance_elapsed_tensor(step, state, tail)
            output.append(base.predict(state)["canonical"].cpu().numpy())
    output = np.stack(output)
    reports = []
    for j, (name, _, scenario) in enumerate(definitions):
        brake = output[:, j, 2]
        if scenario.startswith("step"):
            released = np.flatnonzero((elapsed >= 0) & (brake < 0.02))
            report = {
                "condition": name,
                "first_brake": float(brake[switch]),
                "release_under_0_02_seconds": float(elapsed[released[0]])
                if len(released)
                else None,
                "gas_mean_4s": float(output[region, j, 1].mean()),
                "brake_mean_4s": float(brake[region].mean()),
            }
        else:
            on = region & (schedules[j] == 0)
            off = region & (schedules[j] > 0)
            report = {
                "condition": name,
                "zero_target_rows": int(on.sum()),
                "positive_target_rows": int(off.sum()),
                "brake_mean_zero_target": float(brake[on].mean()),
                "brake_mean_positive_target": float(brake[off].mean()),
                "brake_range_4s": [float(brake[region].min()), float(brake[region].max())],
            }
        reports.append(report)
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Original inputs changed")
    np.savez_compressed(
        run / "predictions.npz",
        canonical=output,
        elapsed=elapsed,
        region=region,
        schedules=np.stack(schedules),
    )
    write_json(
        run / "result.json",
        {
            "status": "completed_stop_transition_counterfactual",
            "reports": reports,
            "switch_index": switch,
            "sample_region": int(region.sum()),
            "predictions_sha256": sha256(run / "predictions.npz"),
            "age_source_sha256": sha256(age_file),
            "counterfactual_body_not_recomputed": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
