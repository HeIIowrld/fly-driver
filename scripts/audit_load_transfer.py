"""Run preregistered no-render mechanical checks; contact truth never enters a policy."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.load_transfer_audit import diagnostic_step


def execute_case(environment, case, seed, output):
    env = FlyBrainEnv(**environment)
    rows = []
    try:
        env.reset(seed=seed, options={"start_x": 20})
        start = env.data.time
        for duration, speed, curvature in case["segments"]:
            steps = round(duration / env.dt)
            if not np.isclose(steps * env.dt, duration):
                raise ValueError("Segment must have an integer number of control steps")
            action = [2 * speed / env.vehicle.max_speed - 1,
                      curvature / env.vehicle.max_curvature, 0]
            if np.max(np.abs(action)) > 1:
                raise ValueError("Diagnostic intent outside actuator domain")
            for _ in range(steps):
                row = diagnostic_step(env, action)
                row["relative_time"] = row["time"] - start
                rows.append(row)
                if (not all(np.isfinite(value).all() for value in row.values())
                        or row["chassis_up_world"][2] < .5
                        or abs(row["com_world_m"][1]) > 19):
                    raise RuntimeError("Diagnostic left bounded upright flat terrain")
        arrays = {name: np.array([r[name] for r in rows]) for name in rows[0]}
        path = output / (case["id"] + ".npz")
        np.savez_compressed(path, **arrays)
        lo, hi = case["window_seconds"]
        selected = (arrays["relative_time"] >= lo) & (arrays["relative_time"] <= hi)
        if selected.sum() < 10:
            raise ValueError("Insufficient diagnostic window")
        force = arrays["mean_ground_force_world_N"][:, :, 2]
        weight = float(arrays["mass_kg"][0] * np.linalg.norm(env.model.opt.gravity))
        return {"file": path.name, "sha256": sha256(path), "steps": len(rows),
            "configuration": env.configuration(), "weight_N": weight,
            "mean_window_wheel_vertical_N": force[selected].mean(0).tolist(),
            "mean_window_suspension_proxy_N":
                arrays["mean_suspension_proxy_N"][selected].mean(0).tolist(),
            "window_com_speed_rms_mps": float(np.sqrt(np.mean(np.sum(
                arrays["com_velocity_world_mps"][selected]**2, axis=1)))),
            "mean_window_acceleration_body_mps2":
                arrays["acceleration_body_mps2"][selected].mean(0).tolist(),
            "linear_balance_rms_over_weight": float(np.sqrt(np.mean(np.sum(
                arrays["linear_balance_residual_N"]**2, axis=1))) / weight),
            "angular_balance_rms_over_weight_wheelbase": float(np.sqrt(np.mean(np.sum(
                arrays["angular_balance_residual_Nm"]**2, axis=1)))
                / (weight * env.vehicle.wheelbase)),
            "unexpected_contacts": int(arrays["unexpected_contacts_during_step"].sum()),
            "termination": "fixed_budget_completed"}
    except Exception as error:
        if rows:
            np.savez_compressed(output / (case["id"] + ".partial.npz"),
                                **{key: np.array([r[key] for r in rows]) for key in rows[0]})
        write_json(output / (case["id"] + ".failure.json"), {
            "status": "failed", "error": type(error).__name__, "message": str(error),
            "recorded_steps": len(rows), "configuration": env.configuration()})
        raise
    finally:
        env.close()


def evaluate_checks(results, limits):
    checks = {}
    rest = np.array(results["rest"]["mean_window_wheel_vertical_N"])
    weight = results["rest"]["weight_N"]
    checks["static_weight"] = abs(rest.sum() / weight - 1) <= limits[
        "static_weight_relative_error_max"]
    for name, result in results.items():
        checks[name + ":linear_balance"] = result["linear_balance_rms_over_weight"] <= limits[
            "linear_momentum_rms_residual_over_weight_max"]
        checks[name + ":angular_balance"] = result[
            "angular_balance_rms_over_weight_wheelbase"] <= limits[
                "angular_momentum_rms_residual_over_weight_wheelbase_max"]
        checks[name + ":contact_domain"] = result["unexpected_contacts"] <= limits[
            "unexpected_contact_count_max"]
    for name, axis, sign in [("acceleration", 0, 1), ("braking", 0, -1),
                              ("left", 1, 1), ("right", 1, -1)]:
        result = results[name]
        delta = np.array(result["mean_window_wheel_vertical_N"]) - rest
        difference = (delta[2:].sum() - delta[:2].sum() if axis == 0
                      else delta[[1, 3]].sum() - delta[[0, 2]].sum())
        key = "longitudinal" if axis == 0 else "lateral"
        checks[name + ":acceleration_sign"] = sign * result[
            "mean_window_acceleration_body_mps2"][axis] >= limits[
                "minimum_" + key + "_acceleration_magnitude_mps2"]
        key = "front_rear" if axis == 0 else "left_right"
        checks[name + ":load_shift_sign"] = sign * difference >= limits[
            "minimum_" + key + "_load_shift_N"]
    left = np.array(results["left"]["mean_window_wheel_vertical_N"])
    right = np.array(results["right"]["mean_window_wheel_vertical_N"])[[1, 0, 3, 2]]
    checks["left_right_mirror"] = np.max(np.abs(left - right)) / (weight / 4) <= limits[
        "left_right_mirror_load_relative_error_max"]
    return {name: bool(value) for name, value in checks.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/load-transfer-audit.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    run = begin_run(args.output, config)
    results = {}
    try:
        for index, case in enumerate(config["cases"]):
            results[case["id"]] = execute_case(config["environment"], case,
                                                config["seed"] + index, run)
            write_json(run / "cases.json", results)
            print(case["id"], json.dumps(results[case["id"]]), flush=True)
        checks = evaluate_checks(results, config["checks_declared_before_run"])
        write_json(run / "result.json", {"status": "completed", "checks": checks,
            "all_mechanical_checks_passed": all(checks.values()), "goal_achieved": False,
            "scope": "synthetic mechanical component only; no actor or learned driving",
            "tyre_thermal_model_implemented": False})
        print("CHECKS", json.dumps(checks), flush=True)
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error), "completed_cases": list(results)})
        raise


if __name__ == "__main__":
    main()
