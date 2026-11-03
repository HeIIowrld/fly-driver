"""Replay visual speed hypothesis through original physical flight graph, offline only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from flybrain_drive.visual_preview_speed import PreviewSpeedCommand
from scripts.prepare_flyvis_preview_folds import registered_cpu
from scripts.replay_registered_visual_yaw import predictor
from scripts.train_pedal_b2 import OriginalGraphAdapter
from scripts.train_visual_body_preview import direct_body_reference


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(args.runtime)
    folds = root / "20261030-t1692-fold-setup"
    fold_result = read(folds / "result.json")
    plan = read(folds / "manifest.json")["config"]["plan"]
    if fold_result["status"] != "completed_cpu_train_history_fold_setup":
        raise ValueError("Original fold setup required")
    entry = next(
        e
        for e in read(Path(plan["source_data"]) / "dataset.json")["episodes"]
        if e["id"] == "episode-04-train"
    )
    source = next(
        e
        for e in read(Path(plan["sensor_replay"]) / "result.json")["episodes"]
        if e["id"] == entry["id"]
    )
    if entry["role"] != "train" or source["role"] != "train":
        raise ValueError("Original internal holdout retains TRAIN role")
    checkpoint = folds / entry["id"] / "vision_body.pt"
    fold = next(f for f in fold_result["folds"] if f["heldout_history"] == entry["id"])
    if sha256(checkpoint) != fold["models"]["vision_body"]["sha256"]:
        raise ValueError("Original fold checkpoint hash mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        saved["condition"] != "vision_body"
        or saved["heldout_history"] != entry["id"]
        or saved["fitted_history_ids"] != ["episode-05-train", "episode-06-train"]
        or saved["actuator_policy"] is not False
        or saved["original_author_checkpoint_sha256"] != plan["original_author_checkpoint_sha256"]
    ):
        raise ValueError("Original disjoint fold checkpoint required")
    fit_dir = root / "20261030-t1913-b2-whole-history-load/firstfit-b2-interface"
    fit_plan_path = fit_dir / "fit-plan-SEALED.json"
    fit_result_path = fit_dir / "actual-fit/result.json"
    pedal_path = fit_dir / "actual-fit/final-two-DNg-scalars.json"
    fit_plan, fit_result, pedal = map(read, (fit_plan_path, fit_result_path, pedal_path))
    if (
        fit_result["status"] != "completed_one_pass_pedal_component_fit"
        or pedal["bindings"] != fit_result["bindings"]
    ):
        raise ValueError("Original fixed physical graph pedal fit required")
    previous = root / "20261030-t1917-registered-visual-yaw-replay/result.json"
    original = read(previous)
    if (
        sha256(previous) != "51803efe0a2f8786ff0cab34a2e576115d9dd0f24d95b9f50d1099dc431e0bb8"
        or original["status"] != "completed_offline_visual_yaw_causal_replay"
    ):
        raise ValueError("Original independent replay reference required")
    vision = Path(source["output"])
    if sha256(vision / "responses.npz") != source["responses_sha256"]:
        raise ValueError("Original FlyVis response changed")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "fold_checkpoint_sha256": sha256(checkpoint),
            "fold_result_sha256": sha256(folds / "result.json"),
            "vision_sha256": source["responses_sha256"],
            "prior_result_sha256": sha256(previous),
            "fit_plan_sha256": sha256(fit_plan_path),
            "fit_result_sha256": sha256(fit_result_path),
            "pedal_checkpoint_sha256": sha256(pedal_path),
            "gain_m": 100.0,
            "external_speed_ceiling_kmh": 60.0,
            "recorded_teacher_speed_hint_kmh": 60.0,
            "role": "internal_train_fold_holdout",
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    adapter = OriginalGraphAdapter(fit_plan, lambda: None)
    with torch.no_grad():
        for name, parameter in adapter.parameters().items():
            parameter.copy_(torch.tensor(pedal["values"][name], device=parameter.device))
        quiet = adapter.initialize()
    fixed = fit_result["fit"]["final_fixed_rollout"]["fixed_fingerprint"]
    if adapter.fixed_fingerprint() != fixed:
        raise ValueError("Fixed motor graph fingerprint mismatch")
    with load_array(vision / "responses.npz", allow_pickle=False) as arrays:
        if arrays["additional_cell_types"].tolist() != ["L1", "L2", "Mi1", "Tm3", "Tm1", "Tm9"]:
            raise ValueError("Original six retinal channels required")
        clocks = arrays["decision_monotonic_seconds"].copy()
        activity = np.concatenate(
            [arrays["additional_activity"], arrays["motion_activity"]], axis=2
        )
    body, valid, _ = direct_body_reference(entry, clocks, sha256(plan["body_contract"]))
    with load_array(vision / "retina-registration.npz", allow_pickle=False) as arrays:
        rays, origins = arrays["body_directions"].copy(), arrays["body_origins_m"].copy()
        if arrays["view_order"].tolist() != ["front", "left", "right", "rear"]:
            raise ValueError("Original view order required")
    episode = dict(
        activity=activity, clock=clocks, body=body, body_valid=valid, rays=rays, origins=origins
    )
    features = registered_cpu(episode, saved["channel_mean"], saved["channel_scale"], batch_size=8)[
        "intact"
    ]
    n = len(clocks)
    if n != 739 or np.any((np.diff(clocks) <= 0) | (np.diff(clocks) > 0.08)):
        raise ValueError("Complete original history required")
    body_t, valid_t = torch.tensor(body, dtype=torch.float32), torch.tensor(valid)
    with load_array(Path(source["recording"]) / "observations.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(clocks, arrays["observation_time"])
        views = torch.tensor(arrays["obs_view_valid"].all(axis=2))
    if not views.all():
        raise ValueError("Original complete view masks required")
    goals = torch.tensor(racing_goal(60, n))
    model = PreviewSpeedCommand(predictor(saved), curvature_gain_m=100.0).eval()
    with torch.no_grad():
        current = model(torch.tensor(features, dtype=torch.float32), body_t, valid_t, views, goals)
        removed = model(
            torch.tensor(features, dtype=torch.float32),
            body_t,
            valid_t,
            views,
            goals,
            intervention="motion_channels",
        )
    prior_yaw = np.asarray(original["yaw_left_rad_s"])[:, 0]
    reference_error = float(np.max(np.abs(current["command"][:, 0].numpy() - prior_yaw)))
    if reference_error > 1e-6:
        raise ValueError("Visual yaw must reproduce original internal holdout")
    commands = torch.stack(
        (
            current["command"],
            removed["command"],
            torch.stack((current["command"][:, 0], goals[:, 1] * 100 / 3.6), 1),
        ),
        1,
    ).cuda()
    body_gpu, valid_gpu, goals_gpu = body_t.cuda(), valid_t.cuda(), goals.cuda()
    state, outputs, currents = quiet.repeat(3, 1), [], []
    with torch.no_grad():
        for i in range(n):
            if i:
                state = adapter.model.advance(
                    commands[i - 1],
                    body_gpu[i - 1 : i].expand(3, -1),
                    valid_gpu[i - 1 : i].expand(3, -1),
                    goals_gpu[i - 1 : i].expand(3, -1),
                    state,
                    float(clocks[i] - clocks[i - 1]),
                    command_contract=MANEUVER_CONTRACT,
                    goal_contract=RACING_GOAL_CONTRACT,
                )[2]
            outputs.append(adapter.predict(state)["canonical"].cpu().numpy())
            currents.append(
                adapter.model.command(
                    commands[i],
                    body_gpu[i : i + 1].expand(3, -1),
                    valid_gpu[i : i + 1].expand(3, -1),
                    goals_gpu[i : i + 1].expand(3, -1),
                    command_contract=MANEUVER_CONTRACT,
                    goal_contract=RACING_GOAL_CONTRACT,
                )[1]
                .cpu()
                .numpy()
            )
    if adapter.fixed_fingerprint() != fixed:
        raise ValueError("Fixed physical graph changed")
    outputs, currents = np.stack(outputs), np.stack(currents)
    np.savez_compressed(
        run / "predictions.npz",
        time=clocks,
        commands=commands.cpu().numpy(),
        canonical=outputs,
        power_current=currents,
    )
    result = {
        "status": "completed_visual_preview_speed_graph_replay",
        "observations": n,
        "conditions": ["visual", "motion_mean", "constant_60kmh"],
        "mean_target_kmh": (commands[:, :, 1].mean(0).cpu().numpy() * 3.6).tolist(),
        "mean_abs_power_change_vs_constant": float(
            np.mean(np.abs(currents[:, 0] - currents[:, 2]))
        ),
        "mean_abs_pedal_change_vs_constant": float(
            np.mean(np.abs(outputs[:, 0, 1:] - outputs[:, 2, 1:]))
        ),
        "yaw_reference_max_error": reference_error,
        "predictions_sha256": sha256(run / "predictions.npz"),
        "recorded_body_is_fixed_counterfactual": True,
        "actual_wind_unverified": True,
        "physiology_or_driving_verified": False,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
