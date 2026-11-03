"""Held-out closed-loop evaluation of the explicitly fixed-speed MuJoCo proxy."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.imitation import teacher_action
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from scripts.collect_multiview_teacher import runtime_inputs
from scripts.initialize_vehicle_circuit import organs


def evaluation_environment(environment, case, steps):
    """Declared scenario geometry is evaluator configuration, never an actor feature."""
    if type(steps) is not int or steps < 1 or not isinstance(case.get("road", {}), dict):
        raise ValueError("Positive scenario budget and explicit road mapping required")
    selected = copy.deepcopy(environment)
    selected["max_steps"] = steps
    selected["road"] = {**selected.get("road", {}), **case.get("road", {})}
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multiview-steering-pilot.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from flybrain_drive.env import FlyBrainEnv
    from flybrain_drive.multiview_runtime import MultiViewMuJoCoRuntime
    from flybrain_drive.sensory_imitation import fixed_speed_action

    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("action_profile") != "mujoco_curvature_proxy_only":
        raise ValueError("Only an explicitly trained MuJoCo curvature proxy may run here")
    graph = {
        k: np.asarray(v, dtype={"pre": np.int64, "post": np.int64, "weights": np.float32}.get(k))
        for k, v in payload["graph"].items()
    }
    model = MultiViewCircuitController(graph, **payload["constructor"])
    model.load_state_dict(payload["model"])
    model.eval()
    _, environment, views, settings, scalars = runtime_inputs(config)
    suite = config["evaluation"]
    conditions = suite["conditions"]
    if (
        not set(conditions) <= {"normal", "vision_missing", "body_missing", "front_only"}
        or type(suite["steps"]) is not int
        or suite["steps"] < 1
    ):
        raise ValueError("Known ablations and a positive evaluation budget required")
    run = begin_run(
        args.output,
        {
            "configuration": config,
            "checkpoint_sha256": sha256(args.checkpoint),
            "teacher_used_by_actor": False,
            "scope": "fixed-speed MuJoCo curvature intent; unsupervised pedal heads ignored",
            "ablations": "after sensor validation; recurrent state carried between decisions",
        },
    )
    results = []
    try:
        with torch.no_grad():
            for condition in ["teacher_reference", *conditions]:
                for case in suite["cases"]:
                    env, runtime = None, None
                    try:
                        env = FlyBrainEnv(**evaluation_environment(
                            environment, case, suite["steps"]))
                        obs, info = env.reset(
                            seed=case["seed"],
                            options={
                                k: case[k]
                                for k in ("start_x", "lateral_offset", "heading_offset_degrees")
                                if k in case
                            },
                        )
                        if condition != "teacher_reference":
                            runtime = MultiViewMuJoCoRuntime(
                                env,
                                views,
                                organs(settings, scalars),
                                periods=config["period_decisions"],
                                source_size=config["source_size"],
                            )
                            if runtime.packer.calibration != model.sensory_calibration:
                                raise ValueError("Evaluation sensory calibration differs")
                        state = model.initial_state()
                        initial = info["truth"]["progress_3d"]
                        trace = []
                        for tick in range(suite["steps"]):
                            if condition == "teacher_reference":
                                steering = float(
                                    teacher_action(env, info, config["fixed_speed_mps"])[1]
                                )
                            else:
                                actor = runtime.observe(obs, case["name"], tick)
                                packed = {
                                    k: torch.tensor(v)[None]
                                    for k, v in runtime.packer.pack(actor).items()
                                }
                                if condition == "vision_missing":
                                    packed["view_valid"].zero_()
                                    packed["contrast_valid"].zero_()
                                elif condition == "body_missing":
                                    packed["organs"].zero_()
                                    packed["organ_valid"].zero_()
                                elif condition == "front_only":
                                    packed["view_valid"][:, 1:].zero_()
                                    packed["contrast_valid"][:, 1:].zero_()
                                drive = model.prepare_drive(packed)
                                for _ in range(config["brain_ticks_per_decision"]):
                                    prediction, state = model.advance(drive, state)
                                steering = float(prediction[0, 0])
                            command = fixed_speed_action(
                                [0, steering], env, config["fixed_speed_mps"]
                            )
                            obs, _, terminated, truncated, info = env.step(command)
                            trace.append(
                                [
                                    float(env.data.time),
                                    steering,
                                    info["truth"]["progress_3d"],
                                    info["truth"]["lateral_error"],
                                ]
                            )
                            if tick % 32 == 0:
                                write_json(
                                    run / "heartbeat.json",
                                    {
                                        "condition": condition,
                                        "case": case["name"],
                                        "decision": tick + 1,
                                    },
                                )
                            if terminated or truncated:
                                break
                        values = np.array(trace)
                        reason = info["reason"] if not env.active else "evaluation_budget"
                        row = {
                            "condition": condition,
                            "case": case["name"],
                            "seed": case["seed"],
                            "steps": len(trace),
                            "reason": reason,
                            "completion": reason == "completion",
                            "progress_delta_m": float(info["truth"]["progress_3d"] - initial),
                            "lateral_rms_m": float(np.sqrt(np.mean(values[:, 3] ** 2))),
                            "steering_rms": float(np.sqrt(np.mean(values[:, 1] ** 2))),
                        }
                        np.savez_compressed(
                            run / f"trace-{len(results):02d}.npz",
                            trace=values,
                            columns=np.array(
                                [
                                    "simulation_time",
                                    "steering_proxy",
                                    "diagnostic_progress",
                                    "diagnostic_lateral",
                                ]
                            ),
                        )
                        results.append(row)
                        write_json(run / "progress.json", {"cases": results})
                        print(json.dumps(row), flush=True)
                    finally:
                        if runtime is not None:
                            runtime.close()
                        if env is not None:
                            env.close()
        write_json(
            run / "evaluation.json",
            {
                "cases": results,
                "closed_loop_evaluated": True,
                "AC_ready": False,
                "goal_achieved": False,
                "note": "Surviving this short segment is not a completed circuit lap",
            },
        )
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False})
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


if __name__ == "__main__":
    main()
