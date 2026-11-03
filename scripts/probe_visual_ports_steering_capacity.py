"""Probe fixed circuit yaw response to bounded visual-port interventions, offline only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.training)
    receipt, plan = read(root / "result.json"), read(root / "manifest.json")["config"]["plan"]
    source = read(Path(plan["source"]) / "result.json")
    spec = read(plan["source_spec"])
    if (
        receipt["status"] != "completed_fixed_joint_command_training"
        or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
        or sha256(source["checkpoint"]) != source["checkpoint_sha256"]
        or sha256(spec["graph_file"]) != spec["graph_sha256"]
        or sha256(spec["annotations_file"]) != spec["annotations_sha256"]
    ):
        raise ValueError("Original fixed graph and annotations required")
    data_root = Path(plan["data"])
    entries = [
        e
        for e in read(data_root / "dataset.json")["episodes"]
        if e["role"] == "train" and e["file"] == "episode-06.npz"
    ]
    if len(entries) != 1 or sha256(data_root / entries[0]["file"]) != entries[0]["sha256"]:
        raise ValueError("Exact original training context required")
    with load_array(data_root / entries[0]["file"], allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    if not torch.cuda.is_available():
        raise ValueError("Original GPU required")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    checkpoint = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["spec"] != model.spec or checkpoint["schema"] != model.schema:
        raise ValueError("Original circuit specification mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    with load_array(spec["graph_file"], allow_pickle=False) as graph:
        graph_ids, types = graph["ids"].tolist(), graph["types"].copy()
    annotations = {str(row["bodyId"]): row for row in read(spec["annotations_file"])}
    groups = {}
    for family in ("HS", "VS"):
        for side in ("L", "R"):
            positions = [
                graph_ids.index(str(identity))
                for identity in spec["vision_input_ids"]
                if types[graph_ids.index(str(identity))].startswith(family)
                and annotations[str(identity)]["somaSide"] == side
            ]
            groups[family + side] = torch.tensor(positions, device="cuda")
    if [len(groups[k]) for k in ("HSL", "HSR", "VSL", "VSR")] != [4, 4, 16, 18]:
        raise ValueError("Expected original identified visual ports and sides required")
    cases = (
        "original",
        "all_zero",
        "all_one",
        "left_high_right_low",
        "left_low_right_high",
        "hs_left_high",
        "hs_right_high",
        "vs_left_high",
        "vs_right_high",
    )
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "episode_sha256": entries[0]["sha256"],
            "dataset_sha256": sha256(data_root / "dataset.json"),
            "conditions": cases,
            "intervention": "replace original model vision-port input current by bounded 0 or 1",
            "direct_motion": "held at TRAIN mean in all conditions",
            "scope": "sampled counterfactual; not physiological stimulation or reachable-set bound",
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
    t = {k: torch.tensor(arrays[k], device="cuda") for k in names}
    if t["visual_valid"].any():
        raise ValueError("Original unavailable visual6 mask must remain false")
    with torch.no_grad():
        base = model._drive_tensor(*(t[k] for k in names))
        scene = t["scene_features"].clone()
        mask = torch.zeros(576, dtype=torch.bool, device="cuda").reshape(4, 9, 16)
        mask[:, 1:] = True
        mask = mask.flatten()
        scene[:, mask] = model.backbone.scene_command.mean[mask]
        mean = model._drive_tensor(*(t[k] for k in names[:5]), scene)
        direct = torch.cat((model.backbone.steering_ports, model.backbone.goal_ports))
        base[:, direct] = mean[:, direct]
        if torch.isin(model.backbone.vision_ports, direct).any():
            raise ValueError("Visual and direct ports must remain disjoint")
        drives = [base.clone() for _ in cases]
        ports = model.backbone.vision_ports
        drives[1][:, ports] = 0
        drives[2][:, ports] = 1
        for name in ("HS", "VS"):
            drives[3][:, groups[name + "L"]], drives[3][:, groups[name + "R"]] = 1, 0
            drives[4][:, groups[name + "L"]], drives[4][:, groups[name + "R"]] = 0, 1
        drives[5][:, groups["HSL"]], drives[5][:, groups["HSR"]] = 1, 0
        drives[6][:, groups["HSL"]], drives[6][:, groups["HSR"]] = 0, 1
        drives[7][:, groups["VSL"]], drives[7][:, groups["VSR"]] = 1, 0
        drives[8][:, groups["VSL"]], drives[8][:, groups["VSR"]] = 0, 1
        for drive in drives:
            if not torch.equal(drive[:, direct], base[:, direct]):
                raise ValueError("Direct ports changed")
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(arrays["observation_time"])]
        state, held, outputs = model.initial_state(len(cases)), None, []
        for i, intervals in enumerate(schedule):
            if i:
                for dt in intervals:
                    state = model._advance_elapsed_tensor(held, state, dt)
            outputs.append(model._read_state_tensor(state)[0].cpu().numpy())
            held = torch.cat([drive[i : i + 1] for drive in drives])
    outputs = np.stack(outputs)
    np.savez_compressed(
        run / "predictions.npz", output=outputs, observation_time=arrays["observation_time"]
    )
    reports = {}
    for j, name in enumerate(cases):
        y = outputs[:, j, 0]
        reports[name] = {
            "min_steer": float(y.min()),
            "max_steer": float(y.max()),
            "positive_count": int((y > 0).sum()),
            "negative_count": int((y < 0).sum()),
            "mean_abs_delta_from_original": float(np.abs(y - outputs[:, 0, 0]).mean()),
        }
    write_json(
        run / "result.json",
        {
            "status": "completed_visual_port_yaw_capacity_probe",
            "reports": reports,
            "predictions_sha256": sha256(run / "predictions.npz"),
            "all_cases_artificial_except_original": True,
            "not_physiology_or_reachable_bound": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
