"""Audit actual saved four-view input currents at anatomically labelled VS ports."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "candidate", "sensors", "labels", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    training, candidate, sensors, labels = map(Path, (a.training, a.candidate, a.sensors, a.labels))
    tr, plan, cr = (
        read(training / "result.json"),
        read(training / "manifest.json")["config"]["plan"],
        read(candidate / "result.json"),
    )
    sr, lm = read(sensors / "result.json"), read(labels / "dataset.json")
    if (
        tr["status"] != "completed_fixed_joint_command_training"
        or sha256(tr["checkpoint"]) != tr["checkpoint_sha256"]
        or cr["status"] != "completed_exterior_motion_preview_train"
        or cr["source_checkpoint_sha256"] != tr["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != cr["checkpoint_sha256"]
        or cr["sensors_result_sha256"] != sha256(sensors / "result.json")
        or cr["labels_dataset_sha256"] != sha256(labels / "dataset.json")
    ):
        raise ValueError("Exact trained VS candidate and original sources required")
    spec = read(plan["source_spec"])
    if (
        sha256(spec["graph_file"]) != spec["graph_sha256"]
        or sha256(spec["annotations_file"]) != spec["annotations_sha256"]
        or sha256(spec["scene_registration_file"]) != spec["scene_registration_sha256"]
    ):
        raise ValueError("Exact original visual graph and registration required")
    with load_array(spec["graph_file"], allow_pickle=False) as g:
        ids, types = g["ids"].tolist(), g["types"].copy()
    annotated = {str(x["bodyId"]): x for x in read(spec["annotations_file"])}
    positions = [ids.index(str(i)) for i in spec["vision_input_ids"]]
    sides = np.array([annotated[str(i)]["somaSide"] for i in spec["vision_input_ids"]])
    vs = np.char.startswith(types[positions], "VS")
    left, right = vs & (sides == "L"), vs & (sides == "R")
    if (int(left.sum()), int(right.sum())) != (16, 18):
        raise ValueError("Original left and right VS input identities required")
    checkpoint = torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    w0, b0 = state["backbone.vision.weight"], state["backbone.vision.bias"]
    candidate_state = torch.load(
        candidate / "visual-boundary.pt", map_location="cpu", weights_only=True
    )
    w1, b1 = candidate_state["weight"], candidate_state["bias"]
    if w0.shape != w1.shape or b0.shape != b1.shape or w0.shape[0] != len(positions):
        raise ValueError("Original port order and candidate dimensions required")
    if torch.any((w1 - w0)[~torch.tensor(vs)] != 0):
        raise ValueError("Candidate must keep HS weights unchanged")
    with load_array(spec["scene_registration_file"], allow_pickle=False) as reg:
        pool = FlightScenePool(reg["image_uv"], spec["scene_basis"])
    sensor_map = {e["id"]: e for e in sr["episodes"]}
    if {e["id"] for e in lm["episodes"]} != set(sensor_map):
        raise ValueError("Exact role-bound original histories required")
    torch.set_num_threads(1)
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "original_checkpoint_sha256": tr["checkpoint_sha256"],
            "candidate_checkpoint_sha256": cr["checkpoint_sha256"],
            "sensors_result_sha256": sha256(sensors / "result.json"),
            "labels_dataset_sha256": sha256(labels / "dataset.json"),
            "scope": ("saved visual port current statistics; not neural state "
                      "or physiological response"),
            "side": "source somaSide, not a proven receptive-field or steering sign",
            "target": "teacher 20m path side for posthoc diagnosis only",
            "threshold_1_m": 0.001,
            "known_development_not_independent": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    saved, reports = {}, []
    for le in lm["episodes"]:
        e = sensor_map[le["id"]]
        vision, body, recording = map(Path, (e["output"], e["body_replay"], e["recording"]))
        vr, br, vm = (
            read(vision / "result.json"),
            read(body / "result.json"),
            read(vision / "manifest.json")["config"],
        )
        if (
            e["role"] != le["role"]
            or sha256(vision / "result.json") != e["result_sha256"]
            or sha256(body / "result.json") != e["body_result_sha256"]
            or sha256(body / "virtual-body.npz") != br["virtual_body_sha256"]
            or sha256(vision / "responses.npz") != vr["responses_sha256"]
            or vr["responses_sha256"] != e["responses_sha256"]
            or vm["events_sha256"] != sha256(recording / "events.jsonl")
            or sha256(le["targets_file"]) != le["targets_sha256"]
        ):
            raise ValueError("Role-bound visual source, body and teacher labels required")
        with (
            load_array(spec["scene_registration_file"], allow_pickle=False) as old,
            load_array(vision / "retina-registration.npz", allow_pickle=False) as reg,
        ):
            for key in (
                "u",
                "v",
                "image_uv",
                "body_directions",
                "view_order",
                "motion_types",
                "motion_indices",
            ):
                np.testing.assert_array_equal(old[key], reg[key])
        with load_array(vision / "responses.npz", allow_pickle=False) as x:
            clocks = x["decision_monotonic_seconds"].copy()
            scene = np.concatenate(
                (x["retina_intensity"][:, :, None], x["motion_activity"]), axis=2
            )
        if scene.shape != (len(clocks), 4, 9, 721) or not np.isfinite(scene).all():
            raise ValueError("Complete finite four-view registered histories required")
        with torch.no_grad():
            features = pool(torch.tensor(scene, dtype=torch.float32))
            visual = torch.cat([torch.zeros((len(clocks), 12)), features], dim=1)
            currents = torch.stack(
                [F.linear(visual, w0, b0).sigmoid(), F.linear(visual, w1, b1).sigmoid()]
            ).numpy()
        # Privileged teacher side is opened after the fixed sensor current computation.
        with load_array(le["targets_file"], allow_pickle=False) as y:
            np.testing.assert_array_equal(clocks, y["observation_time"])
            curvature, valid = y["preview"][:, 2, 1].copy(), y["preview_valid"][:, 2].copy()
        if valid.dtype != bool or not np.isfinite(curvature).all():
            raise ValueError("Finite original teacher targets and explicit mask required")
        active = valid & (np.abs(curvature) > 0.001)
        per_model = {}
        for j, name in enumerate(("original", "vs_candidate")):
            current = currents[j]
            balance = current[:, left].mean(1) - current[:, right].mean(1)
            classes = {}
            for side, name_side in ((1, "screen_right"), (-1, "screen_left")):
                mask = active & (np.sign(curvature) == side)
                classes[name_side] = {
                    "count": int(mask.sum()),
                    "mean_left_minus_right": float(balance[mask].mean()) if mask.any() else None,
                }
            per_model[name] = {
                "vs_min": float(current[:, vs].min()),
                "vs_max": float(current[:, vs].max()),
                "balance_min": float(balance.min()),
                "balance_max": float(balance.max()),
                "class_means": classes,
            }
            saved[le["id"] + "_" + name + "_vs_balance"] = balance
        saved[le["id"] + "_currents"] = currents[:, :, vs]
        saved[le["id"] + "_target_curvature"] = curvature
        saved[le["id"] + "_active"] = active
        reports.append(
            {"id": le["id"], "role": le["role"], "observations": len(clocks), "models": per_model}
        )
        print(json.dumps(reports[-1]), flush=True)
    np.savez_compressed(run / "currents.npz", **saved)
    write_json(
        run / "result.json",
        {
            "status": "completed_vs_saved_port_current_audit",
            "reports": reports,
            "currents_sha256": sha256(run / "currents.npz"),
            "actual_neural_response_measured": False,
            "source_time_alignment_verified": False,
            "known_development_not_independent": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )


if __name__ == "__main__":
    main()
