"""Choose a fixed temporal weight increment gain using only original TRAIN replays."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--replays", nargs=3, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    initial, candidate = Path(a.initial), Path(a.candidate)
    ir, cr = read(initial / "result.json"), read(candidate / "result.json")
    if (
        ir["status"] != "completed_temporal_motion_boundary_onepass"
        or cr["status"] != "completed_exterior_temporal_preview_train"
        or sha256(initial / "temporal-boundary.pt") != ir["checkpoint_sha256"]
        or sha256(candidate / "temporal-boundary.pt") != cr["checkpoint_sha256"]
        or cr["initial_temporal_sha256"] != ir["checkpoint_sha256"]
    ):
        raise ValueError("Exact original initial and updated checkpoints required")
    factors = (0.0, 1.0, 4.0, 16.0, 64.0)
    positive, negative = [[] for _ in factors], [[] for _ in factors]
    seen, sources = set(), []
    for directory in map(Path, a.replays):
        result, config = (
            read(directory / "result.json"),
            read(directory / "manifest.json")["config"],
        )
        path = directory / "predictions.npz"
        if (
            result["status"] != "completed_exterior_temporal_increment_gain_train_replay"
            or result["original_role"] != "train"
            or result["episode"] in seen
            or result["episode"] not in {row["id"] for row in cr["training"]}
            or sha256(path) != result["predictions_sha256"]
            or config["candidate_checkpoint_sha256"] != cr["checkpoint_sha256"]
            or config["initial_temporal_sha256"] != ir["checkpoint_sha256"]
            or config["gain_factors"] != list(factors)
        ):
            raise ValueError("Three distinct exact original TRAIN gain replays required")
        seen.add(result["episode"])
        with load_array(path, allow_pickle=False) as x:
            output, curve, active = x["output"].copy(), x["curvature"].copy(), x["active"].copy()
        if (
            output.ndim != 3
            or output.shape[1:] != (5, 2)
            or curve.shape != (len(output),)
            or active.shape != curve.shape
            or active.dtype != bool
            or not np.isfinite(output).all()
            or not np.isfinite(curve).all()
        ):
            raise ValueError("Original finite five-condition predictions and target mask required")
        target = curve[active] < 0
        if not target.any() or target.all():
            raise ValueError("Both original path signs required")
        for i in range(len(factors)):
            probability = (output[active, i, 0] + 1) / 2
            if np.any((probability <= 0) | (probability >= 1)):
                raise ValueError("Finite interior normalized steering required")
            loss = -(target * np.log(probability) + (~target) * np.log1p(-probability))
            positive[i].append(loss[target])
            negative[i].append(loss[~target])
        sources.append(
            {
                "episode": result["episode"],
                "result_sha256": sha256(directory / "result.json"),
                "predictions_sha256": result["predictions_sha256"],
            }
        )
    if seen != {row["id"] for row in cr["training"]}:
        raise ValueError("All and only original TRAIN episodes required")
    scores = [
        float((np.concatenate(positive[i]).mean() + np.concatenate(negative[i]).mean()) / 2)
        for i in range(len(factors))
    ]
    selected_index = min(range(len(factors)), key=lambda i: (scores[i], factors[i]))
    selected = factors[selected_index]
    old = torch.load(initial / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    new = torch.load(candidate / "temporal-boundary.pt", map_location="cpu", weights_only=True)
    if (
        old["schema"] != "flybrain.temporal_motion_boundary.v1"
        or new["schema"] != old["schema"]
        or old["actor_qualified"] is not False
        or new["actor_qualified"] is not False
        or old["source_checkpoint_sha256"] != new["source_checkpoint_sha256"]
        or old["time_constants_seconds"] != new["time_constants_seconds"]
        or set(old["model"]) != set(new["model"])
    ):
        raise ValueError("Matching original temporal architecture and fixed graph required")
    for key in old["model"]:
        if key != "projection.weight" and not torch.equal(old["model"][key], new["model"][key]):
            raise ValueError("Only temporal projection may differ")
    weight = old["model"]["projection.weight"] + selected * (
        new["model"]["projection.weight"] - old["model"]["projection.weight"]
    )
    if not torch.isfinite(weight).all():
        raise ValueError("Finite selected projection required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "factors": list(factors),
            "initial_sha256": ir["checkpoint_sha256"],
            "updated_sha256": cr["checkpoint_sha256"],
            "sources": sources,
            "criterion": "global class-balanced 20m teacher path sign BCE",
            "selection_uses_only_train": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    old["model"]["projection.weight"] = weight
    old["actor_qualified"] = False
    torch.save(old, run / "temporal-boundary.pt")
    result = {
        "status": "completed_train_selected_temporal_increment_gain",
        "factor": selected,
        "scores": dict(zip([str(int(x)) for x in factors], scores, strict=True)),
        "neutral_bce": float(np.log(2)),
        "checkpoint_sha256": sha256(run / "temporal-boundary.pt"),
        "source_checkpoint_sha256": cr["source_checkpoint_sha256"],
        "boundary_source_sha256": cr["boundary_source_sha256"],
        "labels_dataset_sha256": cr["labels_dataset_sha256"],
        "sensors_result_sha256": cr["sensors_result_sha256"],
        "initial_temporal_sha256": ir["checkpoint_sha256"],
        "training": cr["training"],
        "selection_uses_only_train": True,
        "scaling_is_not_learned_or_biological_tuning": True,
        "actor_deployed": False,
        "optimizer_updates": 0,
        "goal_achieved": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
