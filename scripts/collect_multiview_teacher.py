"""Collect synchronized multi-view sensory episodes with separate teacher-only labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.imitation import teacher_action
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import mixed_legacy_control
from flybrain_drive.multiview_sensory import MultiViewPacker, ViewMap
from scripts.initialize_vehicle_circuit import organs


def runtime_inputs(config):
    if (
        sha256(config["source_checkpoint"]) != config["source_checkpoint_sha256"]
        or sha256(config["view_maps"]) != config["view_maps_sha256"]
    ):
        raise ValueError("Checkpoint or view-map source hash differs")
    payload = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=True)
    source = json.loads(Path(config["source_config"]).read_text(encoding="utf8"))
    scalar_specs = json.loads(Path(config["scalar_calibration"]).read_text(encoding="utf8"))[
        "scalar_organs"
    ]
    views = [
        ViewMap.from_manifest(m)
        for m in json.loads(Path(config["view_maps"]).read_text(encoding="utf8"))
    ]
    environment = {**source["environment"], "control_dt": config["control_dt_seconds"]}
    if (
        environment.get("mode") != "V"
        or environment.get("actuation_mode") != "legacy"
        or not np.isclose(
            config["brain_ticks_per_decision"] * payload["constructor"]["dt"],
            config["control_dt_seconds"],
        )
    ):
        raise ValueError("Declared V/legacy pilot and matched control/circuit periods required")
    return payload, environment, views, source["sensory_preprocessing"], scalar_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiview-steering-pilot.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    beta = config.get("teacher_mixture_beta", 1.)
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("Explicit teacher mixture probability must be in [0,1]")
    if config["action_profile"] != "mujoco_curvature_proxy_only":
        raise ValueError("This collector produces only explicitly labelled MuJoCo steering proxies")
    from flybrain_drive.env import FlyBrainEnv
    from flybrain_drive.multiview_runtime import MultiViewMuJoCoRuntime
    from flybrain_drive.sensory_imitation import fixed_speed_action

    run = begin_run(
        args.output,
        {"configuration": config, "collection": "expert_only" if beta == 1 else "dagger_mixture",
         "teacher_mixture_beta": beta, "actor_route_truth": False},
    )
    env, runtime = None, None
    try:
        payload, environment, views, settings, scalars = runtime_inputs(config)
        student = None
        if beta < 1:
            if payload.get("action_profile") != config["action_profile"]:
                raise ValueError("DAgger source must have a trained matching action profile")
            torch.set_num_threads(1)
            graph = {k: np.asarray(v, dtype={"pre": np.int64, "post": np.int64,
                                            "weights": np.float32}.get(k))
                     for k, v in payload["graph"].items()}
            student = MultiViewCircuitController(graph, **payload["constructor"])
            student.load_state_dict(payload["model"])
            student.eval()
        rng = np.random.default_rng(config["data_seed"])
        episodes = config["training_episodes"]
        if (
            not episodes
            or len({e["name"] for e in episodes}) != len(episodes)
            or any(type(e["steps"]) is not int or e["steps"] < 1 for e in episodes)
            or any(not np.isfinite(e.get("teacher_mixture_beta", beta))
                   or not 0 <= e.get("teacher_mixture_beta", beta) <= 1 for e in episodes)
        ):
            raise ValueError("Unique training episode names and positive budgets required")
        env = FlyBrainEnv(**{**environment, "max_steps": max(e["steps"] for e in episodes) + 1})
        runtime = MultiViewMuJoCoRuntime(
            env,
            views,
            organs(settings, scalars),
            periods=config["period_decisions"],
            source_size=config["source_size"],
        )
        if runtime.packer.calibration != payload["constructor"]["sensory_calibration"]:
            raise ValueError("Runtime sensory contract differs from source checkpoint")
        records, decisions = [], 0
        for i, spec in enumerate(episodes):
            episode_id = spec["name"]
            episode_beta = spec.get("teacher_mixture_beta", beta)
            if student is None and episode_beta < 1:
                raise ValueError("Per-episode student execution needs a configured student mixture")
            obs, info = env.reset(
                seed=config["data_seed"] + i,
                options={
                    k: spec[k]
                    for k in ("start_x", "lateral_offset", "heading_offset_degrees")
                    if k in spec
                },
            )
            packed_rows, labels, native_controls, times = [], [], [], []
            executed, chosen = [], []
            state = None if student is None else student.initial_state()
            initial_progress = info["truth"]["progress_3d"]
            for tick in range(spec["steps"]):
                actor = runtime.observe(obs, episode_id, tick)
                packed = runtime.packer.pack(actor)
                oracle = fixed_speed_action(
                    teacher_action(env, info, config["fixed_speed_mps"]),
                    env,
                    config["fixed_speed_mps"],
                )
                prediction = None
                if student is not None:
                    with torch.no_grad():
                        tensors = {k: torch.tensor(v)[None] for k, v in packed.items()}
                        drive = student.prepare_drive(tensors)
                        # Advance even when the teacher executes, preserving numerical state.
                        for _ in range(config["brain_ticks_per_decision"]):
                            prediction, state = student.advance(drive, state)
                        prediction = prediction[0].numpy()
                control, teacher_chosen = mixed_legacy_control(
                    oracle, prediction, episode_beta, rng)
                packed_rows.append(packed)
                labels.append([float(oracle[1]), 0.0, 0.0])
                native_controls.append(oracle.copy())
                executed.append(control.copy())
                chosen.append(teacher_chosen)
                times.append(actor.simulation_seconds)
                obs, _, terminated, truncated, info = env.step(control)
                if tick % 32 == 0:
                    write_json(
                        run / "heartbeat.json", {"episode": episode_id, "decision": tick + 1}
                    )
                if terminated or truncated:
                    break
            reason = info["reason"] if not env.active else "collection_segment_budget"
            file = run / f"episode-{i:03d}.npz"
            mask = np.zeros((len(labels), 3), dtype=bool)
            mask[:, 0] = True
            np.savez_compressed(
                file,
                **{
                    "obs_" + k: np.stack([r[k] for r in packed_rows])
                    for k in MultiViewPacker.fields
                },
                action=np.array(labels, dtype=np.float32),
                action_valid=mask,
                observation_time=np.array(times),
                teacher_native_control=np.stack(native_controls),
                executed_native_control=np.stack(executed),
                teacher_chosen=np.array(chosen, dtype=bool),
            )
            records.append(
                {
                    "episode_id": episode_id,
                    "file": file.name,
                    "sha256": sha256(file),
                    "steps": len(labels),
                    "seed": config["data_seed"] + i,
                    "teacher_executed_decisions": int(sum(chosen)),
                    "student_executed_decisions": len(chosen) - int(sum(chosen)),
                    "teacher_mixture_beta": episode_beta,
                    "termination": reason,
                    "diagnostic_progress_m": float(info["truth"]["progress_3d"] - initial_progress),
                }
            )
            decisions += len(labels)
            write_json(run / "progress.json", {"episodes": records, "decisions": decisions})
            print(f"collected {episode_id}: {len(labels)} decisions reason={reason}", flush=True)
            if episode_beta == 1 and reason not in (
                    "running", "time_limit", "completion", "collection_segment_budget"):
                raise ValueError("Teacher failed its declared component collection: " + reason)
        manifest = {
            "schema": "flybrain.multiview.teacher.v1",
            "action_profile": config["action_profile"],
            "sensory_calibration": runtime.packer.calibration,
            "actor_fields": sorted(MultiViewPacker.fields),
            "decision_dt_seconds": env.dt,
            "brain_ticks_per_decision": config["brain_ticks_per_decision"],
            "episodes": records,
            "decisions": decisions,
            "teacher_kind": "privileged geometric MuJoCo oracle",
            "native_AC_training_data": False,
            "pedal_labels_available": False,
            "collection": "expert_only" if beta == 1 else "dagger_mixture",
            "teacher_mixture_beta": beta,
            "student_checkpoint_sha256": config["source_checkpoint_sha256"] if student else None,
        }
        write_json(run / "dataset.json", manifest)
        write_json(
            run / "result.json",
            {
                "status": "completed",
                "decisions": decisions,
                "episodes": len(records),
                "goal_achieved": False,
            },
        )
    except Exception as error:
        write_json(
            run / "result.json",
            {
                "status": "failed",
                "error": type(error).__name__,
                "message": str(error),
                "goal_achieved": False,
            },
        )
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
