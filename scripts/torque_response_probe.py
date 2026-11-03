"""Bounded low-speed torque/yaw coupling diagnostic, not a tire calibration."""

import argparse

import numpy as np

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def run_probe(output):
    config = {
        "steps": 200, "seed": 1, "speed_action": 0,
        "curvatures": [0, .1], "residual_actions": [-.5, 0, .5],
        "environment": {
            "mode": "O", "actuation_mode": "residual", "vehicle": {"max_speed": 4},
            "road": {"turn_amplitude": 0, "hill_amplitude": 0,
                     "bank_degrees": 0, "bump_height": 0, "grade": 0},
        },
    }
    run = begin_run(output, config)
    results = []
    for curvature in config["curvatures"]:
        for residual in config["residual_actions"]:
            env = FlyBrainEnv(**config["environment"])
            trace = []
            try:
                _, info = env.reset(seed=config["seed"])
                for _ in range(config["steps"]):
                    action = np.array([0, curvature / env.vehicle.max_curvature, residual])
                    _, _, done, truncated, info = env.step(action)
                    control = info["control"]
                    trace.append({
                        "time": info["time"], "position_y": float(env.data.qpos[1]),
                        "gyro_z": float(env.data.sensor("gyro").data[2]),
                        "requested_residual_Nm": control["residual_yaw_moment_requested"],
                        "commanded_torque_yaw_Nm": control["yaw_moment_from_commanded_torques"],
                        "motor_torques_Nm": control["rear_torque"],
                        "encoder_speed_m_s": control["speed_measured"],
                        "constraint_active": control["torque_constraint_active"],
                    })
                    if done or truncated:
                        break
                path = run / f"probe-{len(results)}.json"
                write_json(path, trace)
                results.append({
                    "curvature_1_m": curvature, "residual_action": residual,
                    "steps": len(trace), "reason": info["reason"],
                    "last": trace[-1],
                    "mean_commanded_torque_yaw_Nm": float(np.mean(
                        [row["commanded_torque_yaw_Nm"] for row in trace]
                    )), "trace_sha256": sha256(path),
                })
            finally:
                env.close()
    write_json(run / "result.json", {
        "status": "completed", "cases": results,
        "limits": "2 s low-speed contact-model probe; commanded torque yaw is not tire yaw",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run_probe(parser.parse_args().output)
