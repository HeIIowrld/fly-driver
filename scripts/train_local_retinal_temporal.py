"""Train local retinal transduction on complete teacher episodes with detached BPTT boundaries."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.local_retinal_circuit import LocalRetinalController
from flybrain_drive.local_retinal_temporal import (
    episode_chunk,
    load_teacher_sequences,
    replay_chunk,
    replay_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "teacher", "poses", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--target-output-rms", type=float, default=.2)
    parser.add_argument("--incoming-boundary",
                        help="Optional full-source incoming totals from exact candidate audit")
    args = parser.parse_args()
    if (min(args.epochs, args.chunk_length) < 1
            or not np.isfinite([args.learning_rate, args.target_output_rms]).all()
            or min(args.learning_rate, args.target_output_rms) <= 0):
        raise ValueError("Positive finite budgets and optimization settings required")
    torch.set_num_threads(1)
    torch.manual_seed(740101)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if sha256(args.poses) != payload["source_sha256"]["poses"]:
        raise ValueError("Original calibration pose source differs")
    calibration = json.loads((Path(args.poses).parent / "sensory-calibration.json")
                              .read_text(encoding="utf8"))
    episodes, teacher_manifest, metadata = load_teacher_sequences(args.teacher, calibration)
    graph = {k: np.asarray(v, dtype={"pre": np.int64, "post": np.int64,
                                      "weights": np.float32}.get(k))
             for k, v in payload["graph"].items()}
    constructor = copy.deepcopy(payload["constructor"])
    if args.incoming_boundary:
        boundary = json.loads(Path(args.incoming_boundary).read_text(encoding="utf8"))["boundary"]
        totals = {r["body_id"]: r["incoming_synapses_total"] for r in boundary}
        if len(totals) != len(boundary) or set(totals) != set(graph["ids"]):
            raise ValueError("Incoming boundary must match exact candidate IDs")
        constructor["incoming_totals"] = [totals[str(body)] for body in graph["ids"]]
    model = LocalRetinalController(graph, **constructor)
    constructed_matrix = model.circuit.matrix.clone()
    model.load_state_dict(payload["model"])
    if args.incoming_boundary:
        model.circuit.matrix = constructed_matrix
        payload["constructor"] = constructor
    if model.output_activation != "linear":
        raise ValueError("This conditioning pilot requires a linear pre-actuator output")
    run = begin_run(args.output, {"arguments": vars(args), "seed": 740101,
        "source_checkpoint_sha256": sha256(args.checkpoint), "script_sha256": sha256(__file__),
        "teacher_manifest_sha256": sha256(Path(args.teacher) / "manifest.json"),
        "teacher_dataset_sha256": sha256(Path(args.teacher) / "dataset.json"),
        "teacher_episode_records": metadata["episodes"],
        "incoming_boundary_sha256": (sha256(args.incoming_boundary)
                                      if args.incoming_boundary else None),
        "incoming_normalization": "full_source" if args.incoming_boundary else "selected_subgraph",
        "training_observation_configuration": teacher_manifest["config"]["policy_configuration"],
        "scope": "training-only temporal BC; no held-out or closed-loop success claim",
        "state_contract": "reset per episode/epoch; carry numerical state across chunks",
        "optimizer_updates": "one per full replay epoch; gradients accumulated across chunks",
        "episode_loss_weighting": "equal episode mean losses",
        "frozen_organ_weights": "zero; retinal-only temporal stage"})
    try:
        before, _ = replay_metrics(model, episodes, args.chunk_length)
        if before["output_rms"] < 1e-9:
            raise ValueError("No output signal for train-input-only gain conditioning")
        original_gain = float(model.readout_gain)
        with torch.no_grad():
            model.readout_gain.mul_(args.target_output_rms / before["output_rms"])
        conditioned, _ = replay_metrics(model, episodes, args.chunk_length)
        write_json(run / "conditioning.json", {"before": before, "after": conditioned,
            "original_gain": original_gain, "conditioned_gain": float(model.readout_gain),
            "gain_uses_teacher_labels": False})
        initialized = copy.deepcopy(payload)
        initialized["model"] = copy.deepcopy(model.state_dict())
        torch.save(initialized, run / "initial-checkpoint.pt")
        optimizer = torch.optim.Adam([model.adapter.visual_adapter.weight], lr=args.learning_rate)
        lengths = [len(e["label"]) for e in episodes]
        weights = torch.tensor([1 / (len(episodes) * length) for length in lengths])
        history = []
        for epoch in range(args.epochs):
            started = time.perf_counter()
            optimizer.zero_grad()
            state = model.initial_state(len(episodes))
            total_loss = 0.
            for start in range(0, max(lengths), args.chunk_length):
                obs, labels, active = episode_chunk(episodes, start, args.chunk_length)
                predicted, state = replay_chunk(model, obs, active, state)
                loss = ((predicted - labels).square() * active * weights).sum()
                loss.backward()
                total_loss += float(loss.detach())
                state = state.detach()  # Numerical state is preserved, never zeroed here.
                write_json(run / "heartbeat.json", {"epoch": epoch + 1, "chunk_start": start})
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            row = {"epoch": epoch + 1, "episode_balanced_rmse_before_update": total_loss**.5,
                   "gradient_norm": float(norm), "seconds": time.perf_counter() - started}
            history.append(row)
            write_json(run / "progress.json", row)
            print(json.dumps(row), flush=True)
        final, traces = replay_metrics(model, episodes, args.chunk_length)
        payload["model"] = model.state_dict()
        payload["temporal_training"] = {"source_checkpoint_sha256": sha256(args.checkpoint),
            "teacher_manifest_sha256": sha256(Path(args.teacher) / "manifest.json"),
            "epochs": args.epochs, "chunk_length": args.chunk_length,
            "conditioned_gain": float(model.readout_gain), "training_only": True}
        torch.save(payload, run / "checkpoint.pt")
        restored = LocalRetinalController(graph, **payload["constructor"])
        restored.load_state_dict(torch.load(run / "checkpoint.pt", weights_only=True)["model"])
        restored_metrics, restored_traces = replay_metrics(restored, episodes, args.chunk_length)
        for a, b in zip(traces, restored_traces, strict=True):
            np.testing.assert_array_equal(a, b)
        assert final == restored_metrics
        np.savez_compressed(run / "training-replay.npz", **{
            f"prediction_{i}": trace for i, trace in enumerate(traces)})
        write_json(run / "history.json", history)
        write_json(run / "summary.json", {"training_replay": final,
            "episode_count": len(episodes), "training_decisions": sum(lengths),
            "checkpoint_restore_max_error": 0, "scope": "training-only replay, not evaluation"})
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False,
                                         "checkpoint_sha256": sha256(run / "checkpoint.pt")})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error)})
        raise


if __name__ == "__main__":
    main()
