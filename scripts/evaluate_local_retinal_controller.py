"""Persistent closed-loop evaluation of a local retinal->MeTu circuit checkpoint."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.local_retinal_circuit import LocalRetinalController
from scripts.evaluate_routed_controller import read_suite


def read_training_calibration(config, payload, poses):
    source = Path(poses).parent
    calibration = json.loads((source / "sensory-calibration.json").read_text(encoding="utf8"))
    original = json.loads((source / "manifest.json").read_text(encoding="utf8"))
    if (sha256(poses) != payload["source_sha256"]["poses"]
            or original["config"]["configuration"] != config):
        raise ValueError("Pose source or observation configuration differs from training")
    return calibration


def apply_vision_condition(packed, condition, frozen=None):
    """Declared post-validation ablation; preserve current body observations."""
    if condition == "vision_missing":
        packed["ray_valid"].zero_()
    elif condition == "vision_frozen":
        if frozen is None:
            frozen = {k: packed[k].detach().clone() for k in ("ray_values", "ray_valid")}
        for key, value in frozen.items():
            packed[key] = value.clone()
    return packed, frozen


def apply_body_condition(packed, condition, position_size):
    if condition == "body_missing":
        for key in ("organs", "organ_valid", "auxiliary", "auxiliary_valid",
                    "applied_control", "applied_valid"):
            packed[key].zero_()
    elif condition in ("position_missing", "rotation_missing"):
        region = (slice(0, position_size) if condition == "position_missing"
                  else slice(position_size, None))
        packed["organs"][:, region] = 0
        packed["organ_valid"][:, region] = 0
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "poses", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--suite",
                        help="Evaluation-only scenario override; checkpoint config unchanged")
    parser.add_argument("--condition", choices=("normal", "vision_missing", "vision_frozen",
                                               "circuit_cut",
                                               "EPG_silenced", "body_missing", "position_missing",
                                               "rotation_missing"), default="normal")
    args = parser.parse_args()
    from flybrain_drive.env import FlyBrainEnv
    from flybrain_drive.sensory_imitation import checked_environment, fixed_speed_action
    from flybrain_drive.sensory_runtime import make_sensory_runtime

    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    suite = read_suite(config, args.suite)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source = Path(args.poses).parent
    calibration = read_training_calibration(config, payload, args.poses)
    graph = {k: np.asarray(v, dtype={"pre": np.int64, "post": np.int64,
                                      "weights": np.float32}.get(k))
             for k, v in payload["graph"].items()}
    model = LocalRetinalController(graph, **payload["constructor"])
    model.load_state_dict(payload["model"])
    model.eval()
    if args.condition == "circuit_cut":
        model.circuit.matrix = model.circuit.matrix * 0
    if args.condition == "EPG_silenced":
        model.circuit.state_mask[np.flatnonzero(graph["types"] == "EPG")] = 0
    run = begin_run(args.output, {"configuration": config, "condition": args.condition,
        "checkpoint_sha256": sha256(args.checkpoint), "poses_sha256": sha256(args.poses),
        "calibration_sha256": sha256(source / "sensory-calibration.json"),
        "script_sha256": sha256(__file__), "teacher_used_live": False,
        "evaluation_suite": suite, "suite_sha256": sha256(args.suite) if args.suite else None,
        "body_feedback_active": any(bool(p.detach().any())
            for layer in model.adapter.organ_adapters for p in layer.parameters()),
        "organ_feedback_calibration": payload.get("organ_feedback"),
        "steps_per_decision": 1, "artificial_settling_at_episode_start": False,
        "frozen_vision_stage": "packed RGB/validity after sensor validation; current body retained",
        "scope": "development MuJoCo driving, not biological validation or AC"})
    results = []
    try:
        with torch.no_grad():
            for case in suite["cases"]:
                environment = checked_environment(config)
                environment["max_steps"] = suite["steps"]
                environment["road"] = {**environment.get("road", {}), **case.get("road", {})}
                env = FlyBrainEnv(**environment)
                try:
                    obs, info = env.reset(seed=int(case["seed"]), options={k: case[k] for k in
                        ("start_x", "lateral_offset", "heading_offset_degrees") if k in case})
                    bridge, packer, sealed = make_sensory_runtime(env, config)
                    if sealed["sensory_encoder"] != calibration:
                        raise ValueError("Runtime eye/organ calibration differs from training")
                    write_json(run / "calibration.json", sealed["sensory_encoder"])
                    state = model.initial_state()
                    frozen = None
                    initial = info["truth"]["progress_3d"]
                    trace, timings = [], []
                    for tick in range(suite["steps"]):
                        started = time.perf_counter()
                        actor = bridge.observe(env, obs, episode_id=case["name"], sequence_id=tick,
                                               received_monotonic_seconds=time.monotonic())
                        packed = {k: torch.tensor(v).unsqueeze(0)
                                  for k, v in packer.pack(actor).items()}
                        packed, frozen = apply_vision_condition(packed, args.condition, frozen)
                        packed = apply_body_condition(packed, args.condition,
                                                       model.adapter.organ_size - 6)
                        encoded = time.perf_counter()
                        intent, state = model(packed, state)
                        steering = float(intent[0].clamp(-1, 1))
                        decided = time.perf_counter()
                        command = fixed_speed_action([0, steering], env, config["fixed_speed_mps"])
                        obs, _, terminated, truncated, info = env.step(command)
                        stepped = time.perf_counter()
                        timings.append([encoded - started, decided - encoded,
                                        stepped - decided, stepped - started])
                        trace.append([actor.simulation_seconds, float(intent[0]), steering,
                                      float(state.abs().max()), info["truth"]["progress_3d"],
                                      info["truth"]["lateral_error"]])
                        if tick % 1000 == 0:
                            write_json(run / "heartbeat.json", {"case": case["name"],
                                                                 "tick": tick + 1})
                        if terminated or truncated:
                            break
                    trace, timings = np.array(trace), np.array(timings)
                    np.savez_compressed(run / f"trace-{len(results):02d}.npz", trace=trace,
                        timing_seconds=timings, columns=np.array(["observation_time",
                            "unclipped_intent", "steering", "max_abs_neural_state",
                            "diagnostic_progress", "diagnostic_lateral"]),
                        timing_columns=np.array(["encoding", "decision", "environment", "total"]))
                    reason = info["reason"] if not env.active else "evaluation_budget"
                    row = {"case": case["name"], "steps": len(trace), "reason": reason,
                           "completion": reason == "completion",
                           "progress_delta": info["truth"]["progress_3d"] - initial,
                           "lateral_rms": float(np.sqrt(np.mean(trace[:, -1]**2)))}
                    results.append(row)
                    write_json(run / "progress.json", {"cases": results})
                    print(json.dumps(row), flush=True)
                finally:
                    env.close()
        write_json(run / "evaluation.json", {"cases": results,
            "completion_rate": float(np.mean([r["completion"] for r in results])),
            "condition": args.condition,
            "scope": "local retinal circuit with declared organ calibration when present"})
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error)})
        raise


if __name__ == "__main__":
    main()
