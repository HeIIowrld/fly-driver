"""Posthoc straight-segment description of fixed visual-speed graph replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.turn_residual import current_turn_prior
from scripts.train_visual_body_preview import direct_body_reference


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.runtime)
    replay = root / "20261031-preview-speed-physical-graph-replay"
    previous = read(replay / "result.json")
    predictions = replay / "predictions.npz"
    if (
        previous["status"] != "completed_visual_preview_speed_graph_replay"
        or sha256(predictions) != previous["predictions_sha256"]
    ):
        raise ValueError("Exact completed fixed graph replay required")
    fold = root / "20261030-t1692-fold-setup"
    plan = read(fold / "manifest.json")["config"]["plan"]
    dataset = Path(plan["source_data"]) / "dataset.json"
    entry = next(e for e in read(dataset)["episodes"] if e["id"] == "episode-04-train")
    if entry["role"] != "train" or sha256(entry["targets_file"]) != entry["targets_sha256"]:
        raise ValueError("Original internal fold target and TRAIN role required")
    with load_array(predictions, allow_pickle=False) as a:
        clock, commands, actions = a["time"].copy(), a["commands"].copy(), a["canonical"].copy()
    body, valid, _ = direct_body_reference(entry, clock, sha256(plan["body_contract"]))
    with load_array(entry["targets_file"], allow_pickle=False) as a:
        np.testing.assert_array_equal(clock, a["observation_time"])
        target, target_valid = a["preview"][:, 2, 1].copy(), a["preview_valid"][:, 2].copy()
    prior, available = current_turn_prior(body, valid)
    straight = available & target_valid & (np.abs(prior[:, 5]) <= .001) & (np.abs(target) <= .001)
    if straight.sum() != 466 or commands.shape != (len(clock), 3, 2):
        raise ValueError("Original near-straight diagnostic and prediction layout required")
    run = begin_run(args.output, {
        "program_sha256": sha256(__file__), "replay_predictions_sha256": sha256(predictions),
        "teacher_target_sha256": entry["targets_sha256"], "dataset_sha256": sha256(dataset),
        "posthoc_diagnostic_only": True, "teacher_actor_input": False, "actor_deployed": False,
    })
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = {
        "status": "completed_preview_speed_straight_diagnostic", "rows": int(straight.sum()),
        "conditions": previous["conditions"],
        "mean_speed_target_kmh": (commands[straight, :, 1].mean(axis=0) * 3.6).tolist(),
        "mean_canonical_accelerator": actions[straight, :, 1].mean(axis=0).tolist(),
        "mean_canonical_brake": actions[straight, :, 2].mean(axis=0).tolist(),
        "brake_above_0_1": (actions[straight, :, 2] > .1).sum(axis=0).tolist(),
        "threshold_is_posthoc_not_driving_gate": True,
        "teacher_is_not_proven_optimal": True, "body_is_fixed_counterfactual": True,
        "actor_deployed": False, "goal_achieved": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
