"""Counterfactual individual incoming-edge removals in saved visual histories."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--contributions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    training = Path(args.training)
    receipt = read(training / "result.json")
    manifest = read(training / "manifest.json")
    plan = manifest["config"]["plan"]
    if (
        receipt["status"] != "completed_fixed_joint_command_training"
        or sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
    ):
        raise ValueError("Original fixed joint training checkpoint required")
    source = read(Path(plan["source"]) / "result.json")
    if sha256(source["checkpoint"]) != source["checkpoint_sha256"]:
        raise ValueError("Original source checkpoint changed")
    if not torch.cuda.is_available():
        raise ValueError("GPU required for original cached graph")
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(
        plan["source_spec"], torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    ).to("cuda")
    checkpoint = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["spec"] != model.spec or checkpoint["schema"] != model.schema:
        raise ValueError("Original graph/spec mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    candidate = Path(args.candidate)
    trained = read(candidate / "result.json")
    if (
        trained["status"] != "completed_one_pass_hsvs_boundary_train"
        or trained["source_checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != trained["checkpoint_sha256"]
    ):
        raise ValueError("Exact completed HS/VS boundary candidate required")
    boundary = torch.load(candidate / "visual-boundary.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        model.backbone.vision.weight.copy_(boundary["weight"].to("cuda"))
        model.backbone.vision.bias.copy_(boundary["bias"].to("cuda"))
    spec = read(plan["source_spec"])
    with load_array(spec["graph_file"], allow_pickle=False) as graph:
        types = graph["types"].copy()
        graph_ids = graph["ids"].copy()
    comparison = read(Path(args.comparison) / "result.json")
    if (
        comparison["status"] != "completed_four_view_isolated_motion_interventions"
        or sha256(Path(args.comparison) / "predictions.npz") != comparison["predictions_sha256"]
        or sha256(spec["graph_file"]) != spec["graph_sha256"]
    ):
        raise ValueError("Pinned graph and previous prediction evidence required")
    positions = torch.tensor(np.flatnonzero(types == "DNb06"), device="cuda")
    with load_array(spec["graph_file"], allow_pickle=False) as graph:
        if list(graph["ids"][positions.cpu().numpy()]) != ["10888", "11067"]:
            raise ValueError("Exact DNb06 left/right identities required")
    matrix = model.backbone.circuit.matrix.coalesce()
    index, values = matrix.indices(), matrix.values()
    rows = torch.zeros((2, model.backbone.neurons), device="cuda")
    for j, position in enumerate(positions):
        selected = index[0] == position
        rows[j, index[1, selected]] = values[selected]
    prior_root = Path(args.prior)
    old_receipt = read(prior_root / "result.json")
    if (
        old_receipt["status"] != "completed_dnb06_original_history_state_audit"
        or sha256(prior_root / "predictions.npz") != old_receipt["predictions_sha256"]
    ):
        raise ValueError("Unchanged prior natural history audit required")
    root = Path(plan["data"])
    dataset = read(root / "dataset.json")
    entries = dataset["episodes"]
    if (
        sum(e["role"] == "train" for e in entries) != 6
        or sum(e["role"] == "development" for e in entries) != 2
    ):
        raise ValueError("Original six TRAIN and two known DEVELOPMENT histories required")
    evidence = Path(args.contributions)
    ev = read(evidence / "result.json")
    if (
        ev["status"] != "completed_dnb06_natural_edge_contributions"
        or sha256(evidence / "predictions.npz") != ev["predictions_sha256"]
    ):
        raise ValueError("Completed contribution evidence required")
    source_ids = ("514540", "12406", "11458", "15422")
    source_types = ("PS352", "PS213", "DNge043", "VST2")
    source_positions = [list(graph_ids).index(s) for s in source_ids]
    if tuple(types[i] for i in source_positions) != source_types:
        raise ValueError("Prespecified graph source identities changed")
    target = int(positions[1])
    weights = rows[1, source_positions]
    if torch.any(weights == 0):
        raise ValueError("Each selected source must have a nonzero edge to right DNb06")
    conditions = ("intact", *source_ids)
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "training_result_sha256": sha256(training / "result.json"),
            "candidate_checkpoint_sha256": trained["checkpoint_sha256"],
            "dataset_sha256": sha256(root / "dataset.json"),
            "prior_predictions_sha256": old_receipt["predictions_sha256"],
            "contribution_result_sha256": sha256(evidence / "result.json"),
            "conditions": conditions,
            "source_ids": source_ids,
            "edge_weights": weights.cpu().tolist(),
            "intervention": "remove one chosen source-to-right-DNb06 graph message each substep",
            "direct_ports": "motion channels at train mean in both scenes",
            "posthoc_source_selection": True,
            "known_development_not_blind": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    direct_ports = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    reports, saved = [], {}
    with torch.no_grad():
        for entry in entries:
            path = root / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Original episode changed")
            with load_array(path, allow_pickle=False) as archive:
                arrays = {k: archive[k].copy() for k in archive.files}
            names = ("visual", "visual_valid", "body", "body_valid", "goals", "scene_features")
            t = {k: torch.tensor(arrays[k], device="cuda") for k in names}
            if t["visual_valid"].any():
                raise ValueError("Original visual mask must be preserved")
            mean_scene = t["scene_features"].clone()
            select = torch.zeros(576, dtype=torch.bool, device="cuda").reshape(4, 9, 16)
            select[:, 1:] = True
            select = select.flatten()
            mean_scene[:, select] = model.backbone.scene_command.mean[select]
            intact = model._drive_tensor(*(t[k] for k in names))
            mean = model._drive_tensor(*(t[k] for k in names[:5]), mean_scene)
            intact[:, direct_ports] = mean[:, direct_ports]
            schedule = [[]] + [substeps(float(x)) for x in np.diff(arrays["observation_time"])]
            state = model.initial_state(10)
            outputs, neurons = [], []
            held = None
            for index, intervals in enumerate(schedule):
                if index:
                    for dt in intervals:
                        corrected = held.clone()
                        # Two scene conditions, each with four distinct single-edge removals.
                        for scene in range(2):
                            for j, source_position in enumerate(source_positions):
                                k = scene * 5 + j + 1
                                corrected[k, target] -= state[k, source_position] * weights[j]
                        state = model._advance_elapsed_tensor(corrected, state, dt)
                outputs.append(model._read_state_tensor(state)[0])
                neurons.append(state[:, positions])
                held = torch.cat(
                    [
                        intact[index : index + 1].expand(5, -1),
                        mean[index : index + 1].expand(5, -1),
                    ],
                    dim=0,
                )
            outputs, neurons = (
                torch.stack(outputs).cpu().numpy(),
                torch.stack(neurons).cpu().numpy(),
            )
            prefix = entry["file"].removesuffix(".npz")
            with load_array(prior_root / "predictions.npz", allow_pickle=False) as old:
                for k, name in ((0, "intact"), (5, "motion_mean")):
                    np.testing.assert_allclose(
                        outputs[:, k], old[prefix + "_" + name + "_intent"], rtol=0, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        neurons[:, k], old[prefix + "_" + name + "_state"], rtol=0, atol=1e-6
                    )
            saved[prefix + "_outputs"] = outputs
            saved[prefix + "_dnb06"] = neurons
            motion_effect = outputs[:, 0] - outputs[:, 5]
            dn_effect = neurons[:, 0, 1] - neurons[:, 5, 1]
            results = {}
            for j, source in enumerate(source_ids):
                ablated_motion = outputs[:, j + 1] - outputs[:, j + 6]
                ablated_dn = neurons[:, j + 1, 1] - neurons[:, j + 6, 1]
                results[source] = {
                    "right_dn_motion_effect_change_mean_abs": float(
                        np.abs(ablated_dn - dn_effect).mean()
                    ),
                    "output_motion_effect_change_mean_abs": np.abs(ablated_motion - motion_effect)
                    .mean(0)
                    .tolist(),
                    "output_motion_effect_change_max_abs": np.abs(ablated_motion - motion_effect)
                    .max(0)
                    .tolist(),
                }
            reports.append(
                {
                    "file": entry["file"],
                    "role": entry["role"],
                    "observations": len(schedule),
                    "edge_removals": results,
                }
            )
            print(json.dumps(reports[-1]), flush=True)
    np.savez_compressed(run / "replay.npz", **saved)
    if (
        sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]
        or sha256(candidate / "visual-boundary.pt") != trained["checkpoint_sha256"]
    ):
        raise ValueError("Fixed checkpoint changed")
    write_json(
        run / "result.json",
        {
            "status": "completed_single_edge_counterfactual_replay",
            "reports": reports,
            "replay_sha256": sha256(run / "replay.npz"),
            "posthoc_source_selection": True,
            "known_development_not_independent": True,
            "intervention_not_biological_lesion": True,
            "actor_deployed": False,
            "optimizer_updates": 0,
            "goal_achieved": False,
        },
    )


if __name__ == "__main__":
    main()
