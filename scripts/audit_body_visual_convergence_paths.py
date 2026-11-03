"""Audit structural and effective shortest paths from body/vision ports to DNb06."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def distances_to_target(n, pre, post, target):
    """Reverse BFS: shortest forward directed path from each node to target."""
    order = np.argsort(post, kind="stable")
    predecessors = pre[order]
    offsets = np.r_[0, np.cumsum(np.bincount(post, minlength=n))]
    distance = np.full(n, -1, dtype=np.int32)
    distance[target] = 0
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for prev in predecessors[offsets[node] : offsets[node + 1]]:
            if distance[prev] < 0:
                distance[prev] = distance[node] + 1
                queue.append(int(prev))
    return distance


def summarize(distance, sources):
    vals = distance[sources]
    return {
        "source_count": len(sources),
        "reachable_count": int((vals >= 0).sum()),
        "shortest_hops": int(vals[vals >= 0].min()) if (vals >= 0).any() else None,
        "hop_histogram": {str(i): int((vals == i).sum()) for i in range(1, 9)},
        "more_than_8_hops": int((vals > 8).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.training)
    result = json.loads((root / "result.json").read_text(encoding="utf8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if (
        result["status"] != "completed_fixed_joint_command_training"
        or sha256(result["checkpoint"]) != result["checkpoint_sha256"]
    ):
        raise ValueError("Exact frozen joint checkpoint required")
    spec_path = Path(manifest["config"]["plan"]["source_spec"])
    spec = json.loads(spec_path.read_text(encoding="utf8"))
    if sha256(spec["graph_file"]) != spec["graph_sha256"]:
        raise ValueError("Exact original structural graph required")
    with load_array(spec["graph_file"], allow_pickle=False) as graph:
        ids, pre, post, weight = (graph[k].copy() for k in ("ids", "pre", "post", "weights"))
    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    embedded = payload["spec"]
    if (
        embedded["source_visual_spec_sha256"] != sha256(spec_path)
        or embedded["graph_sha256"] != spec["graph_sha256"]
        or embedded["body_input_ids"] != spec["body_input_ids"]
        or embedded["vision_input_ids"] != spec["vision_input_ids"]
    ):
        raise ValueError("Original model specification required")
    state = payload["model"]
    matrix = state["backbone.circuit.matrix"].coalesce()
    indices, values = matrix.indices().numpy(), matrix.values().numpy()
    active = values != 0
    active_pre, active_post = indices[1, active], indices[0, active]
    indexes = {str(body): i for i, body in enumerate(ids)}
    body = np.array([indexes[x] for x in spec["body_input_ids"]])
    vision = np.array([indexes[x] for x in spec["vision_input_ids"]])
    yaw_weights = state["backbone.body.weight"][:, 2].numpy()
    if len(body) != len(yaw_weights) or not np.isfinite(yaw_weights).all():
        raise ValueError("Matching calibrated yaw input matrix required")
    yaw_body = body[yaw_weights != 0]
    reports = []
    for name, target_id in [("DNb06_L", "10888"), ("DNb06_R", "11067")]:
        target = indexes[target_id]
        for label, source, dest in [
            ("structural", pre, post),
            ("nonzero_model_weight", active_pre, active_post),
        ]:
            distance = distances_to_target(len(ids), source, dest, target)
            reports.append(
                {
                    "target": name,
                    "target_id": target_id,
                    "edges": label,
                    "yaw_body_ports": summarize(distance, yaw_body),
                    "vision_ports": summarize(distance, vision),
                }
            )
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "graph_sha256": spec["graph_sha256"],
            "spec_sha256": sha256(spec_path),
            "method": "directed reverse BFS; reachability only",
            "yaw_port_definition": "nonzero column 2 of calibrated body boundary",
            "nonzero_edges": "actual fixed model matrix, including its sign/NT hypotheses",
            "optimizer_updates": 0,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    out = {
        "status": "completed_body_visual_structural_convergence_audit",
        "reports": reports,
        "structural_edges": len(pre),
        "nonzero_model_edges": int(active.sum()),
        "yaw_nonzero_boundary_ports": len(yaw_body),
        "yaw_weight_max_abs": float(abs(yaw_weights).max()),
        "reachability_not_function_or_dynamics": True,
        "output_actor_controlled": False,
        "optimizer_updates": 0,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
