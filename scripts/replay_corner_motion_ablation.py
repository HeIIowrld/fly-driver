"""Offline recurrent four-view motion-mean interventions on exploratory transit."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_airflow import ExperimentalVisualBodyAirflow
from flybrain_drive.flight_visual_body_temporal_context import (
    PerViewTemporalContextOpponentFlightStream,
    PerViewVisualBodyContextBoundary,
)
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_camera_group import require_matching_camera_geometry
from flybrain_drive.relative_airflow_sensor import missing_airflow
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from flybrain_drive.view_motion_interventions import mean_motion_views


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "recording",
        "assignment",
        "raw-audit",
        "vision",
        "body",
        "common",
        "per-view",
        "training",
        "opponent",
        "config",
        "output",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    root, audit, vision, body, common, per_view, training, opponent = map(
        Path,
        (a.recording, a.raw_audit, a.vision, a.body, a.common, a.per_view, a.training, a.opponent),
    )
    rr, ar, vr, br, cr, pr, tr, op = (
        read(x / "result.json")
        for x in (root, audit, vision, body, common, per_view, training, opponent)
    )
    manifests = [
        read(x / "manifest.json")["config"] for x in (audit, vision, body, common, per_view)
    ]
    event_hash = sha256(root / "events.jsonl")
    assignment = read(a.assignment)
    if (
        assignment["lease"]["role"] != "transit"
        or rr["status"] != "completed"
        or rr.get("training_admissible") is not False
        or ar["status"] != "completed"
        or not ar["camera_aid_state_verified"]
        or any(ar["actor_replay_max_errors"].values())
        or vr["status"] != "completed"
        or br["status"] != "completed"
        or cr["status"] != "completed_temporal_body_context"
        or pr["status"] != "completed_per_view_body_receipt_audit"
        or any(m["events_sha256"] != event_hash for m in manifests)
        or sha256(vision / "responses.npz") != vr["responses_sha256"]
        or sha256(body / "virtual-body.npz") != br["virtual_body_sha256"]
        or sha256(common / "body-pairs.npz") != cr["body_pairs_sha256"]
        or sha256(per_view / "per-view-body.npz") != pr["arrays_sha256"]
        or manifests[-1]["common_context_sha256"] != cr["body_pairs_sha256"]
        or sha256(tr["checkpoint"]) != op["source_checkpoint_sha256"]
        or sha256(opponent / "temporal-boundary.pt") != op["checkpoint_sha256"]
        or not torch.cuda.is_available()
    ):
        raise ValueError("Original transit role, raw sensor audit and exact model lineage required")
    root_manifest = read(root / "manifest.json")["config"]
    model_cfg = read(a.config)
    if sha256(model_cfg["camera_config"]) != model_cfg["camera_config_sha256"]:
        raise ValueError("Pinned model camera contract required")
    require_matching_camera_geometry(
        root_manifest["sensory_configuration"], read(model_cfg["camera_config"])
    )
    events = [
        json.loads(x) for x in (root / "events.jsonl").read_text(encoding="utf8").splitlines()
    ]
    with np.load(vision / "responses.npz", allow_pickle=False) as d:
        retinal, motion, times, captures = (
            d[k].copy()
            for k in (
                "retina_intensity",
                "motion_activity",
                "decision_monotonic_seconds",
                "capture_ids",
            )
        )
    with np.load(body / "virtual-body.npz", allow_pickle=False) as d:
        body_values, body_valid, body_times, packets = (
            d[k].copy()
            for k in ("values", "valid", "decision_monotonic_seconds", "native_packet_ids")
        )
    with np.load(common / "body-pairs.npz", allow_pickle=False) as d:
        common_values, common_valid, host, common_packets = (
            d[k].copy() for k in ("values", "valid", "original_host_timing", "packets")
        )
    with np.load(per_view / "per-view-body.npz", allow_pickle=False) as d:
        views = {
            k: d[k].copy()
            for k in ("values", "valid", "packets", "receipts", "render_start_lower", "decisions")
        }
    n = len(events)
    if (
        n != rr["decisions"]
        or n != pr["observations"]
        or retinal.shape != (n, 4, 721)
        or motion.shape != (n, 4, 8, 721)
        or body_values.shape != (n, 21)
        or views["values"].shape != (n, 4, 21)
        or not np.array_equal(times, body_times)
        or not np.array_equal(times, views["decisions"])
        or not np.array_equal(times, host[:, 4])
        or not np.array_equal(packets, common_packets[:, 0])
        or not np.array_equal(body_valid, common_valid[:, 0])
        or not np.allclose(body_values, common_values[:, 0], atol=1e-6, rtol=0)
    ):
        raise ValueError("Complete original chronological body and visual evidence required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "events_sha256": event_hash,
            "assignment_sha256": sha256(a.assignment),
            "raw_audit_sha256": sha256(audit / "result.json"),
            "vision_sha256": vr["responses_sha256"],
            "body_sha256": br["virtual_body_sha256"],
            "common_sha256": cr["body_pairs_sha256"],
            "per_view_sha256": pr["arrays_sha256"],
            "training_checkpoint_sha256": sha256(tr["checkpoint"]),
            "opponent_sha256": op["checkpoint_sha256"],
            "seed": 863009,
            "role": "exploratory transit, no fit or independent performance claim",
            "intervention": "pooled motion channels replaced by training mean across entire history; retinal intensity preserved",
            "branches": ["intact", "intact_repeat", "front", "left", "right", "rear", "all"],
            "yaw_context_boundary": "zero initialized, no engineering probe",
            "fixed_task_speed_kmh": 15,
        },
    )
    output = {"status": "failed", "training_admissible": False, "actor_deployed": False}
    try:
        torch.set_num_threads(1)
        torch.manual_seed(863009)
        plan = read(training / "manifest.json")["config"]["plan"]
        source = read(Path(plan["source"]) / "result.json")
        model = EmbodiedVisualFlightCircuit(
            plan["source_spec"],
            torch.load(source["checkpoint"], map_location="cpu", weights_only=True),
        ).to("cuda")
        model.load_state_dict(
            torch.load(tr["checkpoint"], map_location="cpu", weights_only=True)["model"],
            strict=True,
        )
        model.eval().requires_grad_(False)
        temporal = OpponentTemporalMotionBoundary(
            model.backbone.scene_command.mean,
            model.backbone.scene_command.scale,
            len(model.backbone.vision_ports),
        ).to("cuda")
        temporal.load_state_dict(
            torch.load(opponent / "temporal-boundary.pt", map_location="cpu", weights_only=True)[
                "model"
            ],
            strict=True,
        )
        temporal.eval().requires_grad_(False)
        joint = ExperimentalVisualBodyAirflow(model).to("cuda").eval().requires_grad_(False)
        boundary = PerViewVisualBodyContextBoundary(len(model.backbone.vision_ports), device="cuda")
        streams = [
            PerViewTemporalContextOpponentFlightStream(
                joint, temporal, boundary, optimize_held=True
            )
            for _ in range(7)
        ]
        contract, episode = sha256("configs/ac-virtual-navigation-body-v2.json"), "transit-original"
        visual, valid = (
            torch.zeros(1, 6, device="cuda"),
            torch.zeros(1, 6, dtype=torch.bool, device="cuda"),
        )
        goals = torch.tensor([[0.0, 0.5]], device="cuda")
        scene_valid, records = torch.ones((1, 4), dtype=torch.bool, device="cuda"), []
        with torch.no_grad():
            for i, event in enumerate(events):
                camera = event["camera"]
                if (
                    event["index"] != i
                    or captures[i] != camera["capture_id"]
                    or times[i] != event["decision_monotonic_seconds"]
                    or packets[i] != event["native"]["packet_id"]
                ):
                    raise ValueError("Original event identity changed")
                camera_views = {x["view_id"]: x for x in camera["views"]}
                if set(camera_views) != set(VIEW_ORDER):
                    raise ValueError("All four original views required")
                lower = tuple(camera_views[v]["render_started_lower"] for v in VIEW_ORDER)
                prior_time, first_lower, receipt, body_received, now = map(float, host[i])
                if (
                    first_lower != min(lower)
                    or receipt != camera["received"]
                    or not np.array_equal(views["render_start_lower"][i], lower)
                ):
                    raise ValueError("Original per-view timing changed")
                current = NavigationBodyObservation(
                    episode,
                    int(packets[i]),
                    body_received,
                    now,
                    body_values[i].copy(),
                    body_valid[i].copy(),
                    contract,
                )
                previous = (
                    None
                    if common_packets[i, 1] < 0
                    else NavigationBodyObservation(
                        episode,
                        int(common_packets[i, 1]),
                        prior_time,
                        receipt,
                        common_values[i, 1].copy(),
                        common_valid[i, 1].copy(),
                        contract,
                    )
                )
                separate = tuple(
                    None
                    if views["packets"][i, j] < 0
                    else NavigationBodyObservation(
                        episode,
                        int(views["packets"][i, j]),
                        float(views["receipts"][i, j]),
                        receipt,
                        views["values"][i, j].copy(),
                        views["valid"][i, j].copy(),
                        contract,
                    )
                    for j in range(4)
                )
                image = np.concatenate((retinal[i, :, None], motion[i]), axis=1)
                scene = model.backbone.scene_pool(
                    torch.as_tensor(image[None], dtype=torch.float32, device="cuda")
                )
                intents = []
                for stream, selected in zip(streams, ((), (), (0,), (1,), (2,), (3,), (0, 1, 2, 3)), strict=True):
                    contexts = separate
                    branch_scene = mean_motion_views(scene, model.backbone.scene_command.mean, selected)
                    body_context = per_view_temporal_body_context(
                        contexts,
                        current,
                        render_start_lowers=lower,
                        camera_receipt_time=receipt,
                        decision_time=now,
                    )
                    intent, _ = stream.decision(
                        decision_time=now,
                        visual=visual,
                        visual_valid=valid,
                        body_context=body_context,
                        goals=goals,
                        scene_features=branch_scene,
                        scene_valid=scene_valid,
                        airflow_observation=missing_airflow(now),
                    )
                    intents.append(intent.cpu().numpy()[0].tolist())
                records.append(
                    {
                        "index": i,
                        "packet": int(packets[i]),
                        "time": now,
                        "yaw": float(body_values[i, 4]),
                        "intents": intents,
                    }
                )
        write_json(run / "events.json", records)
        intent = np.asarray([x["intents"] for x in records])
        differences = intent[:, :1, :] - intent[:, 2:, :]
        repeat = np.abs(intent[:, 0] - intent[:, 1])
        interaction = differences[:, :4].sum(axis=1) - differences[:, 4]
        output.update(
            status="completed_corner_four_view_motion_ablation",
            observations=n,
            branches=["intact", "intact_repeat", *VIEW_ORDER, "all_motion_mean"],
            repeat_max_abs=repeat.max(axis=0).tolist(),
            intervention_mean_abs=differences.__abs__().mean(axis=0).tolist(),
            intervention_max_abs=differences.__abs__().max(axis=0).tolist(),
            groups_exceeding_2e_6=(np.abs(differences) > 2e-6).any(axis=2).sum(axis=0).tolist(),
            interaction_mean_abs=np.abs(interaction).mean(axis=0).tolist(),
            interaction_max_abs=np.abs(interaction).max(axis=0).tolist(),
            interaction_definition="sum of four individual intact-minus-ablated histories minus intact-minus-all-ablated history",
            events_sha256=sha256(run / "events.json"),
            original_episode_resets=0,
            temporal_state_independent_per_branch=True,
            synthetic_mean_intervention_not_camera_loss=True,
            physical_exposure_time_verified=False,
            physiological_or_closed_loop_benefit_verified=False,
            optimizer_updates=0,
            vehicle_control_sent=False,
        )
    except Exception as error:
        output.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run / "result.json", output)
        print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
