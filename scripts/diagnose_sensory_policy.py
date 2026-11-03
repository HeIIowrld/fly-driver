"""Run the new sensory-to-policy path in real MuJoCo; no training or success claim."""

import argparse
import time
from dataclasses import asdict

import numpy as np
import torch

from flybrain_drive.body_organs import BodyOrganEncoder, PositionOrganSpec
from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import demo_graph, sha256
from flybrain_drive.learning import restore_policy, save_checkpoint
from flybrain_drive.mujoco_sensory import MuJoCoSensoryBridge
from flybrain_drive.policy import policy_from_config, tensor_observation
from flybrain_drive.sensory_policy import ObservationPacker, SensoryPolicyRunner


def diagnose(output, *, steps=100, seed=17, kind="gru"):
    if type(steps) is not int or steps < 1 or kind not in ("gru", "graph"):
        raise ValueError("Diagnostic needs positive steps and gru/graph policy")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    settings = {
        "mode": "V", "max_steps": steps + 1, "camera_hz": 25,
        "vehicle": {"max_speed": 4},
        "camera": {"fovy": 120, "source_size": 96, "forward": .95,
                   "pitch_degrees": -15, "near_m": .03, "far_m": 250},
    }
    env = FlyBrainEnv(**settings)
    run = None
    try:
        observation, info = env.reset(seed=seed)
        bridge = MuJoCoSensoryBridge(env)
        specs = [PositionOrganSpec("steering_angle", 2, -.5, .5, .15, 1),
                 PositionOrganSpec("suspension_position", 4, -.07, .07, .02, 1)]
        organs = BodyOrganEncoder(specs, rotation_scale=5)
        packer = ObservationPacker(bridge.eyes, organs,
                                   specific_force_scale=20, wheel_speed_scale=50)
        config = {
            "environment": settings, "policy": kind, "seed": seed, "torch_threads": 1,
            "driver": "untrained_random_sensor_policy", "training_updates": 0,
            "graph_source": "synthetic_demo" if kind == "graph" else None,
            "sensory_encoder": {"organ_size": packer.organ_size,
                                "eye_map_hashes": {e.side: e.sha256 for e in bridge.eyes}},
            "sensory_calibration": {
                "position_organs": [asdict(spec) for spec in specs], "rotation_scale": 5,
                "specific_force_scale": 20, "wheel_speed_scale": 50, "max_sample_age": .15,
            },
            "scope": "rendered MuJoCo integration, not anatomical fly/AC/performance evidence",
        }
        run = begin_run(output, config)
        write_json(run / "eye-maps.json", {e.side: e.manifest() for e in bridge.eyes})
        graph = demo_graph() if kind == "graph" else None
        policy = policy_from_config(config, graph)
        runner = SensoryPolicyRunner(policy, packer)
        save_checkpoint(run / "untrained-checkpoint.pt", policy,
                        torch.optim.Adam(policy.parameters()), config, graph, 0)
        trace, packed_rows, frames, frame_times = [], [], [], []
        initial_progress = info["truth"]["progress_3d"]
        previous_exposure = None
        for tick in range(steps):
            actor = bridge.observe(env, observation, episode_id="diagnostic-0", sequence_id=tick,
                                   received_monotonic_seconds=time.monotonic())
            action, packed = runner.step(actor)
            packed_rows.append(packed)
            if actor.eyes[0].frame_id != previous_exposure:
                frames.append(env.camera_source_images.copy())
                frame_times.append(actor.eyes[0].time.simulation_seconds)
                previous_exposure = actor.eyes[0].frame_id
            observation, _, terminated, truncated, info = env.step(action)
            trace.append({
                "decision_simulation_seconds": actor.simulation_seconds,
                "eye_exposure_ids": [e.frame_id for e in actor.eyes],
                "sample_ages": actor.ages(), "action": action.tolist(),
                "ray_valid_fraction": packed["ray_valid"].mean(axis=1).tolist(),
                "organ_valid_fraction": float(packed["organ_valid"].mean()),
                "training_only_diagnostic": {
                    "reason": info["reason"], "progress_3d": info["truth"]["progress_3d"],
                    "lateral_error": info["truth"]["lateral_error"],
                },
            })
            if terminated or truncated:
                break
        np.savez_compressed(run / "actor-sequence.npz", **{
            key: np.stack([row[key] for row in packed_rows]) for key in packed_rows[0]
        })
        np.savez_compressed(run / "source-exposures.npz", rgb=np.stack(frames),
                            simulation_seconds=np.array(frame_times))
        restored, _ = restore_policy(run / "untrained-checkpoint.pt")
        state, replay_error = restored.initial_state(), 0.0
        for packed, row in zip(packed_rows, trace, strict=True):
            with torch.no_grad():
                mean, _, state = restored(tensor_observation(packed), state)
            replay_error = max(replay_error, float(np.max(np.abs(
                torch.tanh(mean)[0].numpy() - row["action"]
            ))))
        if replay_error > 1e-6:
            raise RuntimeError("Saved sensory-policy checkpoint does not replay recorded actions")
        write_json(run / "trace.json", trace)
        write_json(run / "result.json", {
            "status": "completed", "steps": len(trace), "training_updates": 0,
            "reason": info["reason"] if not env.active else "diagnostic_budget",
            "diagnostic_progress_m": info["truth"]["progress_3d"] - initial_progress,
            "source_exposures": len(frames), "restored_action_max_abs_error": replay_error,
            "sequence_sha256": sha256(run / "actor-sequence.npz"),
            "scope": config["scope"], "target_goal_achieved": False,
        })
        return run
    except Exception as error:
        if run is not None:
            write_json(run / "result.json", {"status": "failed", "reason": type(error).__name__,
                                             "message": str(error)})
        raise
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--kind", choices=["gru", "graph"], default="gru")
    args = parser.parse_args()
    print(diagnose(args.output, steps=args.steps, seed=args.seed, kind=args.kind))


if __name__ == "__main__":
    main()
