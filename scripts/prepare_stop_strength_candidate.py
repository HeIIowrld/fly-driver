"""Prepare a fixed training-selected stop-boundary scale for diagnostic replay."""

import argparse
import json
from pathlib import Path

import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, audit = Path(args.source), Path(args.audit)
    trained, comparison = read(source / "result.json"), read(audit / "result.json")
    checkpoint = source / "stop-task-adapter.pt"
    if (trained["status"] != "completed_fixed_episode_sustained_brake_fit"
            or comparison["status"] != "completed_stop_boundary_strength_audit"
            or sha256(checkpoint) != trained["checkpoint_sha256"]
            or sha256(checkpoint) != comparison["candidate_checkpoint_sha256"]):
        raise ValueError("Exact training candidate and its original comparison required")
    choices = [row for row in comparison["reports"] if row["factor"] == 512.0]
    if len(choices) != 1 or choices[0]["hold_brake_mean"] <= .9:
        raise ValueError("Predeclared 512x condition with original training response required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["model"]["stop_delta"] *= 512.0
    payload["schema"] = "flybrain.experimental_stop_vector_scaled.v1"
    payload["actor_qualified"] = False
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "original_checkpoint_sha256": sha256(checkpoint),
        "training_comparison_result_sha256": sha256(audit / "result.json"),
        "scale": 512.0, "scope": "training-selected offline comparison, no actor"})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    torch.save(payload, run / "stop-task-adapter.pt")
    write_json(run / "result.json", {"status": "completed_fixed_stop_strength_selection",
        "checkpoint_sha256": sha256(run / "stop-task-adapter.pt"), "actor_deployed": False,
        "optimizer_updates": 0, "selection": "posthoc training data, not biological tuning"})


if __name__ == "__main__":
    main()
