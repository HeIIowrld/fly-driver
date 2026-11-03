"""Reload a speed-task checkpoint; audit saved outputs, fixed edges and original state history."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.flight_learning import GOAL_CONTRACT
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import drives_for, replay, substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


@torch.no_grad()
def partitioned(model, data, schedule, *, reset_at=None):
    state, held, outputs = model.initial_state(1), None, []
    for start in range(0, len(schedule), 37):
        drive = drives_for(model, data, start, min(start + 37, len(schedule)))
        for local, index in enumerate(range(start, min(start + 37, len(schedule)))):
            if index == reset_at:
                state, held = model.initial_state(1), None
            if index and held is not None:
                for dt in schedule[index]:
                    state = model._advance_elapsed_tensor(held, state, dt)
            outputs.append(model._read_state_tensor(state)[0][0])
            held = drive[local:local + 1]
    return torch.stack(outputs).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training = Path(args.training)
    result = read(training / "result.json")
    configuration = read(training / "manifest.json")["config"]
    plan = configuration["plan"]
    data_root = Path(configuration["arguments"]["data"])
    dataset = read(data_root / "dataset.json")
    checkpoint = Path(result["checkpoint"])
    if (result["status"] != "completed_offline_initialization"
            or sha256(checkpoint) != result["checkpoint_sha256"]
            or sha256(data_root / "dataset.json") != configuration["dataset_manifest_sha256"]
            or sha256(plan["spec"]) != configuration["spec_sha256"]):
        raise ValueError("Successful unchanged fit/spec/data required")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no CPU fallback")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["goal_contract"] != GOAL_CONTRACT or payload["actor_qualified"] is not False
            or payload["spec_sha256"] != sha256(plan["spec"])):
        raise ValueError("Explicit v4 task and unqualified checkpoint required")
    model = FlightCircuit(plan["spec"], glutamate_sign=payload["glutamate_sign"]).to("cuda")
    original = model.circuit.matrix.coalesce().detach().clone()
    model.load_state_dict(payload["model"], strict=True)
    matrix = model.circuit.matrix.coalesce()
    if (not torch.equal(matrix.indices(), original.indices())
            or not torch.equal(matrix.values(), original.values())):
        raise ValueError("Checkpoint changes original graph values or indices")
    run = begin_run(args.output, dict(arguments=vars(args), seed=plan["seed"],
                                    checkpoint_sha256=sha256(checkpoint),
                                    dataset_manifest_sha256=sha256(data_root / "dataset.json"),
                                    actuator_messages_sent=0))
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(dataset["episodes"]), 2, figsize=(13, 14),
                                layout="constrained", squeeze=False)
    reports = []
    for number, entry in enumerate(dataset["episodes"]):
        if sha256(data_root / entry["file"]) != entry["sha256"]:
            raise ValueError("Complete dataset episode changed")
        with load_array(data_root / entry["file"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        data = {key: torch.tensor(value, device="cuda") for key, value in arrays.items()
                if value.dtype.kind in "fbiu" and key != "observation_time"}
        clock = arrays["observation_time"]
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(clock)]
        expected = replay(model, data, schedule)[0].cpu().numpy()
        with load_array(training / entry["file"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(clock, saved["observation_time"])
            np.testing.assert_array_equal(arrays["target_valid"], saved["target_valid"])
            np.testing.assert_array_equal(arrays["target"], saved["target"])
            saved_error = float(np.abs(expected - saved["prediction"]).max())
        split = partitioned(model, data, schedule)
        split_error = float(np.abs(expected - split).max())
        np.testing.assert_allclose(expected, split, atol=3e-6, rtol=2e-5)
        if saved_error > 3e-6:
            raise ValueError("Saved prediction differs from fixed-weight reload")
        teacher = np.flatnonzero(arrays["phases"] == "expert_observation")
        reset_difference = None
        if len(teacher) and teacher[0] > 0:
            reset = partitioned(model, data, schedule, reset_at=int(teacher[0]))
            reset_difference = float(np.abs(expected - reset).max())
            if reset_difference < 1e-5:
                raise ValueError("Handoff reset counterexample is not sensitive")
        reports.append(dict(file=entry["file"], observations=len(clock),
                            saved_prediction_max_error=saved_error,
                            chunk37_max_error=split_error,
                            handoff_reset_max_difference=reset_difference,
                            saved_prediction_sha256=sha256(training / entry["file"])))
        for axis, label in enumerate(["Left-turn intent", "Signed forward drive"]):
            panel = axes[number, axis]
            target = np.where(arrays["target_valid"][:, axis], arrays["target"][:, axis], np.nan)
            panel.plot(clock - clock[0], target, label="Applied-control target", linewidth=.8)
            panel.plot(clock - clock[0], expected[:, axis], label="Fresh v4 prediction",
                       linewidth=.8)
            title = f"{entry['file']} / {entry['role']} / {entry['desired_speed_kmh']} km/h"
            panel.set(xlabel="Original elapsed time (s)", ylabel=label,
                      title=title)
            panel.legend(fontsize=7)
    figure.savefig(run / "complete-episode-fits.png", dpi=140)
    plt.close(figure)
    write_json(run / "result.json", dict(status="completed", episodes=reports,
               termination_reason="strict reload, all original clocks and histories checked",
               original_graph_indices_and_weights_unchanged=True,
               full_history_preserved=True, actor_qualified=False,
               physiological_stabilization_verified=False, independent_course_test=False,
               actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(dict(status="completed", episodes=len(reports))))


if __name__ == "__main__":
    main()
