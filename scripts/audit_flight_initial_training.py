"""Reproduce the saved trained candidate, preserve chunk history, and audit sign uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import drives_for, metrics, replay, substeps


@torch.no_grad()
def partitioned_replay(model, data, schedule, *, reset_at=None):
    state, held = model.initial_state(1), None
    predictions = []
    for start in range(0, len(schedule), 37):
        stop = min(start + 37, len(schedule))
        drive = drives_for(model, data, start, stop)
        for local, index in enumerate(range(start, stop)):
            if index == reset_at:
                state = model.initial_state(1)
            if index:
                for dt in schedule[index]:
                    state = model._advance_elapsed_tensor(held, state, dt)
            intent, _ = model._read_state_tensor(state)
            predictions.append(intent[0])
            held = drive[local : local + 1]
        state, held = state.detach(), held.detach()
    return torch.stack(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    report = json.loads((root / "result.json").read_text(encoding="utf8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))["config"]
    plan = manifest["plan"]
    checkpoint = Path(report["checkpoint"])
    if sha256(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("Trained checkpoint changed")
    cache = Path(args.data) / "episode.npz"
    if sha256(cache) != manifest["dataset_sha256"]:
        raise ValueError("Sealed training cache changed")
    torch.set_num_threads(1)
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=plan["seed"],
            checkpoint_sha256=report["checkpoint_sha256"],
            dataset_sha256=manifest["dataset_sha256"],
            actuator_messages_sent=0,
        ),
    )
    with load_array(cache, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    data = {
        key: torch.tensor(value, device="cuda")
        for key, value in arrays.items()
        if value.dtype.kind in "fbiu" and key != "observation_time"
    }
    schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if loaded["actor_qualified"] or loaded["epoch"] != plan["epochs"]:
        raise ValueError("Expected fixed-last-epoch unqualified initialization candidate")
    model = FlightCircuit(plan["spec"], glutamate_sign=plan["glutamate_sign"]).to("cuda")
    source_values = model.circuit.matrix.coalesce().values().clone()
    source_indices = model.circuit.matrix.coalesce().indices().clone()
    model.load_state_dict(loaded["model"], strict=True)
    if not torch.equal(model.circuit.matrix.coalesce().values(), source_values):
        raise ValueError("Training altered fixed measured signed counts")
    if not torch.equal(model.circuit.matrix.coalesce().indices(), source_indices):
        raise ValueError("Training altered measured topology")
    prediction, _ = replay(model, data, schedule)
    with load_array(root / "predictions.npz", allow_pickle=False) as archive:
        saved = torch.tensor(archive["intent"], device="cuda")
        np.testing.assert_array_equal(archive["observation_time"], arrays["observation_time"])
    torch.testing.assert_close(prediction, saved, rtol=2e-5, atol=2e-6)
    partitioned = partitioned_replay(model, data, schedule)
    torch.testing.assert_close(prediction, partitioned, rtol=2e-5, atol=2e-6)
    reset = partitioned_replay(model, data, schedule, reset_at=807)
    reset_error = float((prediction - reset).abs().max().item())
    if reset_error <= 1e-4:
        raise ValueError("AI-handoff reset counterexample has insufficient sensitivity")
    sign_metrics = {str(plan["glutamate_sign"]): metrics(prediction, data)}
    for sign in (0, 1):
        altered = FlightCircuit(plan["spec"], glutamate_sign=sign).to("cuda")
        source_parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, parameter in altered.named_parameters():
                parameter.copy_(source_parameters[name])
            for name in ("power_reference", "lateral_reference", "reference_calibrated"):
                getattr(altered.readout, name).copy_(getattr(model.readout, name))
        alternative, _ = replay(altered, data, schedule)
        sign_metrics[str(sign)] = metrics(alternative, data)
        del altered
    write_json(
        run / "result.json",
        dict(
            status="completed",
            termination_reason=("checkpoint, whole-history/chunk replay "
                                "and sign sensitivity verified"),
            saved_prediction_max_error=float((prediction - saved).abs().max().item()),
            partitioned_prediction_max_error=float((prediction - partitioned).abs().max().item()),
            handoff_reset_counterexample_max_difference=reset_error,
            original_topology_and_signed_counts_unchanged=True,
            all_1543_decision_times_exact=True,
            sign_sensitivity_metrics=sign_metrics,
            sign_hypothesis_selected_from_validation=False,
            checkpoint_selected_from_validation=False,
            development_tail_reused_after_training_fix=True,
            independent_test=False,
            actor_qualified=False,
            actuator_messages_sent=0,
            goal_achieved=False,
        ),
    )
    print(
        json.dumps(
            dict(
                status="completed",
                sign_metrics=sign_metrics,
                replay_error=float((prediction - saved).abs().max().item()),
                partition_error=float((prediction - partitioned).abs().max().item()),
                reset_error=reset_error,
            )
        )
    )


if __name__ == "__main__":
    main()
