"""Replay retained pace images and body-context counterfactuals; no fit or control."""

import argparse
import json
from dataclasses import replace
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
from flybrain_drive.relative_airflow_sensor import missing_airflow
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "recording",
        "vision",
        "body",
        "context",
        "brake-audit",
        "training",
        "opponent",
        "output",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--per-view-context")
    p.add_argument("--fixed-yaw-probe", action="store_true")
    p.add_argument("--intact-only", action="store_true")
    a = p.parse_args()
    root, vision, body, context, brake, training, opponent = map(
        Path, (a.recording, a.vision, a.body, a.context, a.brake_audit, a.training, a.opponent)
    )
    rr, vr, br, cr, ar, tr, op = (
        read(x / "result.json") for x in (root, vision, body, context, brake, training, opponent)
    )
    vm, bm, cm, am = (read(x / "manifest.json")["config"] for x in (vision, body, context, brake))
    event_hash = sha256(root / "events.jsonl")
    if (
        rr["status"] != "completed"
        or vr["status"] != "completed"
        or br["status"] != "completed"
        or cr["status"] != "completed_temporal_body_context"
        or ar["status"] != "completed_pace_brake_diagnostic_audit"
        or cr.get("training_admissible") is not False
        or ar.get("training_admissible") is not False
        or any(m["events_sha256"] != event_hash for m in (vm, bm, cm, am))
        or sha256(vision / "responses.npz") != vr["responses_sha256"]
        or sha256(body / "virtual-body.npz") != br["virtual_body_sha256"]
        or sha256(context / "body-pairs.npz") != cr["body_pairs_sha256"]
        or sha256(brake / "moving-brake-receipts.json") != ar["receipt_rows_sha256"]
        or sha256(tr["checkpoint"]) != op["source_checkpoint_sha256"]
        or sha256(opponent / "temporal-boundary.pt") != op["checkpoint_sha256"]
        or not torch.cuda.is_available()
    ):
        raise ValueError("Exact diagnostic sensor, brake, checkpoint and GPU lineage required")
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf8").splitlines()
    ]
    brake_rows = read(brake / "moving-brake-receipts.json")
    brake_indices = {row["index"] for row in brake_rows}
    if (
        len(events) != 731
        or len(brake_rows) != 62
        or len(brake_indices) != 62
        or any(
            events[row["index"]]["native"]["packet_id"] != row["native_packet"]
            for row in brake_rows
        )
    ):
        raise ValueError("All original decisions and separately audited moving-brake rows required")
    with np.load(vision / "responses.npz", allow_pickle=False) as v:
        retinal, motion, clocks, capture = (
            v[k].copy()
            for k in (
                "retina_intensity",
                "motion_activity",
                "decision_monotonic_seconds",
                "capture_ids",
            )
        )
    with np.load(body / "virtual-body.npz", allow_pickle=False) as b:
        values, valid, body_clock, packet = (
            b[k].copy()
            for k in ("values", "valid", "decision_monotonic_seconds", "native_packet_ids")
        )
    with np.load(context / "body-pairs.npz", allow_pickle=False) as c:
        pairs, pair_valid, times, pair_packet = (
            c[k].copy() for k in ("values", "valid", "original_host_timing", "packets")
        )
    per_view = None
    if a.per_view_context:
        view_root = Path(a.per_view_context)
        view_result = read(view_root / "result.json")
        view_manifest = read(view_root / "manifest.json")["config"]
        if (
            view_result.get("status") not in ("completed_pace_per_view_body_receipt_audit",
                                               "completed_per_view_body_receipt_audit")
            or view_result.get("training_admissible") is not False
            or view_manifest["events_sha256"] != event_hash
            or view_manifest["common_context_sha256"] != cr["body_pairs_sha256"]
            or sha256(view_root / "per-view-body.npz") != view_result["arrays_sha256"]
        ):
            raise ValueError("Exact audited per-view original context required")
        with np.load(view_root / "per-view-body.npz", allow_pickle=False) as data:
            per_view = {
                k: data[k].copy()
                for k in (
                    "values",
                    "valid",
                    "packets",
                    "receipts",
                    "render_start_lower",
                    "decisions",
                )
            }
        if per_view["values"].shape != (731, 4, 21) or not np.array_equal(
            per_view["decisions"], clocks
        ):
            raise ValueError("Complete same-clock per-view history required")
    if (
        retinal.shape != (731, 4, 721)
        or motion.shape != (731, 4, 8, 721)
        or values.shape != (731, 21)
        or pairs.shape != (731, 2, 21)
        or not np.array_equal(clocks, body_clock)
        or not np.array_equal(clocks, times[:, 4])
        or not np.array_equal(packet, pair_packet[:, 0])
        or not np.array_equal(valid, pair_valid[:, 0])
        or not np.allclose(values, pairs[:, 0], atol=1e-6, rtol=0)
    ):
        raise ValueError("Complete same-clock original visual and body evidence required")
    run = begin_run(
        a.output,
        {
            "program_sha256": sha256(__file__),
            "events_sha256": event_hash,
            "vision_sha256": vr["responses_sha256"],
            "body_sha256": br["virtual_body_sha256"],
            "context_sha256": cr["body_pairs_sha256"],
            "brake_sha256": ar["receipt_rows_sha256"],
            "flight_sha256": sha256(tr["checkpoint"]),
            "opponent_sha256": op["checkpoint_sha256"],
            "seed": 863009,
            "fixed_task_speed_kmh": 15,
            "role": "pace diagnostic only, no fit",
            "camera_domain_shift": "exterior mount differs from trained yaw-camera contract",
            "historical_body": (
                "independent prior for each view"
                if per_view
                else "one proven prior before earliest view, shared by four views"
            ),
            "conditions": (
                ["intact"]
                if a.intact_only
                else ["intact", "current_tilt_zero", "current_load_zero"]
            ),
            "per_view_context_sha256": (view_result["arrays_sha256"] if per_view else None),
            "fixed_yaw_probe": a.fixed_yaw_probe,
            "fixed_yaw_probe_weight": 0.01 / 256 if a.fixed_yaw_probe else 0.0,
            "intact_only": a.intact_only,
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
        if a.fixed_yaw_probe:
            with torch.no_grad():
                boundary.projection.weight.fill_(0.01 / 256)
        streams = [
            PerViewTemporalContextOpponentFlightStream(
                joint, temporal, boundary, optimize_held=True
            )
            for _ in range(1 if a.intact_only else 3)
        ]
        contract = sha256("configs/ac-virtual-navigation-body-v2.json")
        episode = "pace-diagnostic-original"
        visual, visual_valid = (
            torch.zeros(1, 6, device="cuda"),
            torch.zeros(1, 6, dtype=torch.bool, device="cuda"),
        )
        goals = torch.tensor([[0.0, 0.5]], device="cuda")
        scene_valid = torch.ones((1, 4), dtype=torch.bool, device="cuda")
        records = []
        with torch.no_grad():
            for i, event in enumerate(events):
                camera = event["camera"]
                if (
                    event["index"] != i
                    or capture[i] != camera["capture_id"]
                    or event["decision_monotonic_seconds"] != clocks[i]
                    or event["native"]["packet_id"] != packet[i]
                ):
                    raise ValueError("Original camera/native sequence changed")
                views = {v["view_id"]: v for v in camera["views"]}
                if set(views) != set(VIEW_ORDER):
                    raise ValueError("Original four views required")
                lower = tuple(views[v]["render_started_lower"] for v in VIEW_ORDER)
                prior_time, first_lower, receipt, body_received, now = map(float, times[i])
                if first_lower != min(lower) or receipt != camera["received"]:
                    raise ValueError("Original conservative prior/render timing changed")
                current = NavigationBodyObservation(
                    episode,
                    int(packet[i]),
                    body_received,
                    now,
                    values[i].copy(),
                    valid[i].copy(),
                    contract,
                )
                prior = (
                    None
                    if pair_packet[i, 1] < 0
                    else NavigationBodyObservation(
                        episode,
                        int(pair_packet[i, 1]),
                        prior_time,
                        receipt,
                        pairs[i, 1].copy(),
                        pair_valid[i, 1].copy(),
                        contract,
                    )
                )
                prior_views = (prior,) * 4
                if per_view is not None:
                    if not np.array_equal(per_view["render_start_lower"][i], lower):
                        raise ValueError("Original per-view render bounds changed")
                    prior_views = tuple(
                        None
                        if per_view["packets"][i, j] < 0
                        else NavigationBodyObservation(
                            episode,
                            int(per_view["packets"][i, j]),
                            float(per_view["receipts"][i, j]),
                            receipt,
                            per_view["values"][i, j].copy(),
                            per_view["valid"][i, j].copy(),
                            contract,
                        )
                        for j in range(4)
                    )
                scene_data = np.concatenate((retinal[i, :, None], motion[i]), axis=1)
                scene = model.backbone.scene_pool(
                    torch.as_tensor(scene_data[None], dtype=torch.float32, device="cuda")
                )
                intents = []
                interventions = ((),) if a.intact_only else ((), (0, 1), (8, 9, 10, 11))
                for stream, indices in zip(streams, interventions, strict=True):
                    changed = current.values.copy()
                    changed[list(indices)] = 0
                    decision_body = replace(current, values=changed)
                    ctx = per_view_temporal_body_context(
                        prior_views,
                        decision_body,
                        render_start_lowers=lower,
                        camera_receipt_time=receipt,
                        decision_time=now,
                    )
                    intent, _ = stream.decision(
                        decision_time=now,
                        visual=visual,
                        visual_valid=visual_valid,
                        body_context=ctx,
                        goals=goals,
                        scene_features=scene,
                        scene_valid=scene_valid,
                        airflow_observation=missing_airflow(now),
                    )
                    intents.append(intent.cpu().numpy()[0].tolist())
                records.append(
                    {
                        "index": i,
                        "time": now,
                        "packet": int(packet[i]),
                        "moving_brake": i in brake_indices,
                        "intents": intents,
                    }
                )
        write_json(run / "events.json", records)
        intent = np.asarray([x["intents"] for x in records])
        braking = np.asarray([x["moving_brake"] for x in records], dtype=bool)
        summary = {}
        for column, name in () if a.intact_only else ((1, "tilt_zero"), (2, "load_zero")):
            delta = np.abs(intent[:, column] - intent[:, 0])
            summary[name] = {
                "all_mean_abs": delta.mean(axis=0).tolist(),
                "all_max_abs": delta.max(axis=0).tolist(),
                "braking_mean_abs": delta[braking].mean(axis=0).tolist(),
                "braking_max_abs": delta[braking].max(axis=0).tolist(),
            }
        output.update(
            status="completed_pace_visual_body_counterfactual",
            observations=len(records),
            moving_braking_observations=int(braking.sum()),
            interventions=summary,
            per_view_context_used=per_view is not None,
            fixed_yaw_probe=a.fixed_yaw_probe,
            fixed_probe_is_neither_measured_nor_trained=True,
            events_sha256=sha256(run / "events.json"),
            original_episode_resets=0,
            optimizer_updates=0,
            camera_mount_domain_shift=True,
            original_pace_task_not_actor_goal=True,
            physical_source_time_or_exposure_verified=False,
            biological_or_closed_loop_benefit_verified=False,
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
