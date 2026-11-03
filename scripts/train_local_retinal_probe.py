"""Static pose fit through local retinal->MeTu adapters and frozen measured structure."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import load_graph, sha256
from flybrain_drive.local_retinal_circuit import LocalRetinalController


def settle(model, observation, steps):
    drive = model.prepare_drive(observation)
    state = model.initial_state(len(drive))
    for _ in range(steps):
        prediction, state = model.advance(drive, state)
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "ports", "groups", "poses", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--settling-steps", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=730101)
    parser.add_argument("--output-activation", choices=("linear", "tanh"), default="linear")
    parser.add_argument("--learning-rate", type=float, default=.00005)
    args = parser.parse_args()
    if min(args.epochs, args.settling_steps, args.neighbors) < 1:
        raise ValueError("Positive fixed training budgets required")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Positive finite learning rate required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    graph = load_graph(args.graph)
    if graph["ids"][graph["roles"] == "output"].tolist() != ["10360", "523769"]:
        raise ValueError("This pilot's output sign hypothesis requires exact DNa02 IDs")
    groups = json.loads(Path(args.groups).read_text(encoding="utf8"))["groups"]
    visual_types = groups["MeTu"]["types"]
    visual_ids = graph["ids"][np.isin(graph["types"], visual_types)].tolist()
    ports = json.loads(Path(args.ports).read_text(encoding="utf8"))
    with load_array(args.poses, allow_pickle=False) as data:
        observations = {k[4:]: torch.tensor(data[k]) for k in data.files if k.startswith("obs_")}
        labels = torch.tensor(data["teacher_only_labels"][:, 2], dtype=torch.float32)
        train = torch.tensor(data["train_mask"], dtype=torch.bool)
    if not train.any() or train.all():
        raise ValueError("Separate train and test pose rows required")
    # Retinal geometry/coverage are calibration, not pose truth or teacher actions.
    directions = observations["ray_directions"][0].numpy()
    available = observations["ray_valid"][train].any(0).numpy()
    organ_size = observations["organs"].shape[1]
    organ_groups = {}
    for name, bodies in ports["organ_input_groups"].items():
        if name.startswith("haltere_"):
            features = list(range(organ_size - 6, organ_size))
        elif name.startswith("chordotonal_"):
            features = list(range(organ_size - 6))
        else:
            raise ValueError("Unknown organ feature-mapping hypothesis")
        organ_groups[name] = {"body_ids": bodies, "feature_indices": features}
    constructor = {"visual_ids": visual_ids, "organ_groups": organ_groups,
                   "directions": directions.tolist(), "available": available.tolist(),
                   "organ_size": organ_size, "neighbors": args.neighbors,
                   "seed": args.seed, "dt": .01, "output_signs": [-1, 1],
                   "output_activation": args.output_activation}
    model = LocalRetinalController(graph, **constructor)
    training = {k: v[train] for k, v in observations.items()}
    testing = {k: v[~train] for k, v in observations.items()}
    features = training["ray_values"].flatten(1)
    with torch.no_grad():
        model.feature_mean.copy_(features.mean(0))
        model.feature_scale.copy_(features.std(0, correction=0).clamp_min(.05))
        raw = settle(model, training, args.settling_steps)
        rms = raw.square().mean().sqrt()
        if rms < 1e-9:
            raise ValueError("Local retinal drive has no usable differential output")
        model.readout_gain.fill_(1 / rms)
    source_hashes = {name: sha256(getattr(args, name))
                     for name in ("graph", "ports", "groups", "poses")}
    run = begin_run(args.output, {"arguments": vars(args), "source_sha256": source_hashes,
        "script_sha256": sha256(__file__), "training_rows": int(train.sum()),
        "test_rows": int((~train).sum()), "initial_raw_output_rms": float(rms),
        "scope": "static pose fitting only, no driving or anatomical RF claim",
        "learned_parameters": "local retinal input weights only",
        "frozen_organ_weights": "zero; body coupling not yet fitted",
        "physiological_signs_assigned": False, "action_bottleneck_before_circuit": False,
        "static_settling_seconds": args.settling_steps * .01})
    optimizer = torch.optim.Adam([model.adapter.visual_adapter.weight], lr=args.learning_rate)
    history = []
    try:
        for epoch in range(args.epochs):
            started = time.perf_counter()
            optimizer.zero_grad()
            predicted = settle(model, training, args.settling_steps)
            mse = (predicted - labels[train]).square().mean()
            loss = mse + 1e-4 * model.adapter.visual_adapter.weight.square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            row = {"epoch": epoch + 1, "train_rmse": float(mse.detach().sqrt()),
                   "gradient_norm": float(norm), "seconds": time.perf_counter() - started}
            history.append(row)
            write_json(run / "progress.json", row)
            print(json.dumps(row), flush=True)
        with torch.no_grad():
            predictions = {}
            for condition in ("normal", "vision_missing", "circuit_cut", "EPG_silenced"):
                inputs = {k: v.clone() for k, v in testing.items()}
                saved_matrix = model.circuit.matrix.clone()
                if condition == "vision_missing":
                    inputs["ray_valid"].zero_()
                if condition == "circuit_cut":
                    model.circuit.matrix = model.circuit.matrix * 0
                if condition == "EPG_silenced":
                    model.circuit.state_mask[np.flatnonzero(graph["types"] == "EPG")] = 0
                predictions[condition] = settle(model, inputs, args.settling_steps)
                model.circuit.matrix = saved_matrix
                model.circuit.state_mask.fill_(1)
            assert not predictions["circuit_cut"].any()
            assert not predictions["vision_missing"].any()
        checkpoint = {"constructor": constructor, "model": model.state_dict(),
                      "graph": {k: v.tolist() for k, v in graph.items()},
                      "source_sha256": source_hashes, "settling_steps": args.settling_steps}
        torch.save(checkpoint, run / "checkpoint.pt")
        restored = LocalRetinalController(graph, **constructor)
        restored.load_state_dict(torch.load(run / "checkpoint.pt", weights_only=True)["model"])
        with torch.no_grad():
            torch.testing.assert_close(settle(restored, testing, args.settling_steps),
                                       predictions["normal"], atol=0, rtol=0)
        np.savez_compressed(run / "predictions.npz",
                            **{k: v.numpy() for k, v in predictions.items()},
                            teacher_only_test_labels=labels[~train].numpy())
        write_json(run / "history.json", history)
        write_json(run / "summary.json", {"test_rmse": {k: float((v - labels[~train])
            .square().mean().sqrt()) for k, v in predictions.items()},
            "test_clipped_rmse": {k: float((v.clamp(-1, 1) - labels[~train])
                .square().mean().sqrt()) for k, v in predictions.items()},
            "scope": "static held-out-position steering probe; no closed-loop driving proof",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "readout_gain": float(model.readout_gain), "checkpoint_restore_max_error": 0})
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False,
                                         "checkpoint_sha256": sha256(run / "checkpoint.pt")})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error)})
        raise


if __name__ == "__main__":
    main()
