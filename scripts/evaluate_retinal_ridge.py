"""Closed-loop vision-only ridge diagnostic; not a fly circuit or final driving model."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vision", choices=("normal", "missing"), default="normal")
    args = parser.parse_args()
    from flybrain_drive.env import FlyBrainEnv
    from flybrain_drive.evidence import begin_run, write_json
    from flybrain_drive.graph import sha256
    from flybrain_drive.sensory_imitation import checked_environment, fixed_speed_action
    from flybrain_drive.sensory_runtime import make_sensory_runtime

    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    source = Path(args.probe)
    with load_array(source / "poses.npz", allow_pickle=False) as data:
        valid = data["obs_ray_valid"]
        features = (data["obs_ray_values"] * valid[..., None]).reshape(len(valid), -1)
        train = data["train_mask"]
        # Labels are accessed only while fitting, never by the live actor.
        target = data["teacher_only_labels"][train, 2]
        directions = data["obs_ray_directions"][0].copy()
    mean = features[train].mean(0)
    scale = np.maximum(features[train].std(0), .05)
    x = (features[train] - mean) / scale
    bias = target.mean()
    weight = x.T @ np.linalg.solve(x @ x.T + .1 * np.eye(len(x)), target - bias)
    prediction = ((features[~train] - mean) / scale) @ weight + bias
    original = load_array(source / "visual-test-predictions.npy", allow_pickle=False)[:, 2]
    # Float32 feature kernels are reassociated into a compact primal readout.
    # Bound the numerical change far below the observed ~0.05 steering RMSE.
    np.testing.assert_allclose(prediction, original, atol=1e-4, rtol=0)
    run = begin_run(args.output, {
        "configuration": config, "probe_poses_sha256": sha256(source / "poses.npz"),
        "script_sha256": sha256(__file__), "training_rows": int(train.sum()),
        "scope": "vision-only linear diagnostic, not neural-circuit driving",
        "teacher_labels_used_live": False, "body_features_used_by_actor": False,
        "vision_condition": args.vision,
        "offline_replay_max_error": float(np.max(np.abs(prediction - original))),
        "offline_replay_absolute_tolerance": 1e-4,
    })
    np.savez_compressed(run / "ridge-model.npz", mean=mean, scale=scale, weight=weight, bias=bias,
                        directions=directions, training_rows=np.flatnonzero(train))
    results = []
    try:
        for case in config["evaluation"]["cases"]:
            environment = checked_environment(config)
            environment["max_steps"] = config["evaluation"]["steps"]
            environment["road"] = {**environment.get("road", {}), **case.get("road", {})}
            env = FlyBrainEnv(**environment)
            try:
                obs, info = env.reset(seed=int(case["seed"]), options={k: case[k] for k in
                    ("start_x", "lateral_offset", "heading_offset_degrees") if k in case})
                bridge, packer, sealed = make_sensory_runtime(env, config)
                write_json(run / "calibration.json", sealed["sensory_encoder"])
                initial = info["truth"]["progress_3d"]
                trace = []
                for tick in range(config["evaluation"]["steps"]):
                    actor = bridge.observe(env, obs, episode_id=case["name"], sequence_id=tick,
                                           received_monotonic_seconds=time.monotonic())
                    packed = packer.pack(actor)
                    np.testing.assert_array_equal(packed["ray_directions"], directions)
                    eye = (packed["ray_values"] * packed["ray_valid"][..., None]).reshape(-1)
                    if args.vision == "missing":
                        eye = np.zeros_like(eye)
                    steering = float(np.clip(((eye - mean) / scale) @ weight + bias, -1, 1))
                    command = fixed_speed_action([0, steering], env, config["fixed_speed_mps"])
                    obs, _, terminated, truncated, info = env.step(command)
                    trace.append([actor.simulation_seconds, steering,
                                  info["truth"]["progress_3d"], info["truth"]["lateral_error"]])
                    if tick % 1000 == 0:
                        write_json(run / "heartbeat.json", {"case": case["name"],
                                                              "decisions": tick + 1})
                    if terminated or truncated:
                        break
                reason = info["reason"] if not env.active else "evaluation_budget"
                np.savez_compressed(run / f"trace-{len(results):02d}.npz", trace=np.array(trace),
                                    columns=np.array(["observation_time", "steering", "progress_m",
                                                      "lateral_error_m"]))
                row = {"case": case["name"], "steps": len(trace), "reason": reason,
                       "completion": reason == "completion",
                       "progress_delta": info["truth"]["progress_3d"] - initial,
                       "lateral_rms": float(np.sqrt(np.mean(np.array(trace)[:, 3]**2)))}
                results.append(row)
                write_json(run / "progress.json", {"cases": results})
                print(json.dumps(row), flush=True)
            finally:
                env.close()
        summary = {"cases": results, "completion_rate": float(np.mean([
            r["completion"] for r in results])), "scope": "V-only ridge, not full fly/AC driving"}
        write_json(run / "evaluation.json", summary)
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "reason": type(error).__name__,
                                         "message": str(error)})
        raise


if __name__ == "__main__":
    main()
