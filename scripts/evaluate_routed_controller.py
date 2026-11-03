"""Evaluate engineering-calibrated measured-circuit steering; no direct visual motor bypass."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array


def restore(payload):
    from flybrain_drive.routed_controller import RoutedController

    graph = {key: np.asarray(value, dtype={"pre": np.int64, "post": np.int64,
                                          "weights": np.float32}.get(key))
             for key, value in payload["graph"].items()}
    model = RoutedController(graph, payload["ports"], **payload["constructor"])
    model.load_state_dict(payload["model"])
    return model


def read_suite(config, path=None):
    """Evaluation-only override; checkpoint configuration stays unchanged."""
    suite = config["evaluation"] if path is None else json.loads(
        Path(path).read_text(encoding="utf8"))
    cases = suite.get("cases")
    if (type(suite.get("steps")) is not int or suite["steps"] < 1
            or not isinstance(cases, list) or not cases):
        raise ValueError("Suite needs a positive integer budget and cases")
    allowed = {"name", "seed", "start_x", "lateral_offset", "heading_offset_degrees", "road"}
    if any(not isinstance(c, dict) or set(c) - allowed or not isinstance(c.get("name"), str)
           or not c["name"] or type(c.get("seed")) is not int or c["seed"] < 0 for c in cases):
        raise ValueError("Invalid evaluation case schema")
    if len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Evaluation case names must be unique")
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "graph", "ports", "retina", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--suite",
                        help="Evaluation-only JSON; never changes model weights or config")
    parser.add_argument("--position-gain", type=float, default=.2)
    parser.add_argument("--yaw-gain", type=float, default=.1)
    parser.add_argument("--condition", choices=("normal", "body_missing", "vision_missing",
                                               "circuit_cut"), default="normal")
    args = parser.parse_args()
    import torch

    from flybrain_drive.env import FlyBrainEnv
    from flybrain_drive.evidence import begin_run, write_json
    from flybrain_drive.graph import load_graph, sha256
    from flybrain_drive.routed_controller import RoutedController
    from flybrain_drive.sensory_imitation import (
        checked_environment,
        fixed_speed_action,
        policy_digest,
    )
    from flybrain_drive.sensory_runtime import make_sensory_runtime

    torch.set_num_threads(1)
    torch.manual_seed(710101)
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    suite = read_suite(config, args.suite)
    sources = {name: sha256(getattr(args, name)) for name in ("graph", "ports", "retina")}
    run = begin_run(args.output, {"configuration": config, "source_sha256": sources,
        "script_sha256": sha256(__file__), "condition": args.condition, "seed": 710101,
        "requested_position_gain": args.position_gain, "requested_yaw_gain": args.yaw_gain,
        "gain_flags_apply_only_without_checkpoint": True,
        "evaluation_suite": suite, "suite_sha256": sha256(args.suite) if args.suite else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "scope": "cut-boundary measured-structure rate model, not full fly biology or AC",
        "direct_visual_to_command_connection": False,
        "output_sign_hypothesis": {"10360": -1, "523769": 1}})
    try:
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if payload["source_sha256"] != sources or payload["configuration"] != config:
                raise ValueError("Checkpoint source/calibration/configuration differs")
            model = restore(payload)
        else:
            graph = load_graph(args.graph)
            ports = json.loads(Path(args.ports).read_text(encoding="utf8"))
            if graph["ids"][graph["roles"] == "output"].tolist() != ["10360", "523769"]:
                raise ValueError("New graphs require an explicit output-ID sign hypothesis")
            env = FlyBrainEnv(**checked_environment(config))
            try:
                _, packer, sealed = make_sensory_runtime(env, config)
                constructor = {"position_size": sum(p.channels * (p.populations + 2)
                                                     for p in packer.organs.positions),
                               "organ_size": packer.organ_size, "dt": env.dt,
                               "output_signs": [-1, 1]}
                model = RoutedController(graph, ports, **constructor)
                calibration = model.calibrate(packer.organs.positions, env.vehicle,
                    position_gain=args.position_gain, yaw_gain=args.yaw_gain,
                    rotation_scale=packer.organs.rotation_scale)
                calibration["vehicle"] = asdict(env.vehicle)
            finally:
                env.close()
            with load_array(args.retina, allow_pickle=False) as retina:
                retinal = {key: torch.tensor(retina[key]) for key in retina.files}
            payload = {"model": model.state_dict(), "constructor": constructor,
                       "graph": {k: v.tolist() for k, v in graph.items()}, "ports": ports,
                       "retina": retinal, "source_sha256": sources,
                       "configuration": config, "sensory_encoder": sealed["sensory_encoder"],
                       "calibration": calibration}
        torch.save(payload, run / "checkpoint.pt")
        reloaded = restore(torch.load(run / "checkpoint.pt", weights_only=True))
        state_a, state_b = model.initial_state(), reloaded.initial_state()
        with torch.no_grad():
            for _ in range(8):
                inputs = (torch.tensor([[.3]]), torch.ones(1, 1),
                          torch.ones(1, model.organ_size), torch.ones(1, model.organ_size))
                a, state_a, _ = model.advance(*inputs, state_a)
                b, state_b, _ = reloaded.advance(*inputs, state_b)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(state_a, state_b, rtol=0, atol=0)
        write_json(run / "calibration.json", payload["calibration"])
        if args.condition == "circuit_cut":
            model.circuit.structural_weight.zero_()
        write_json(run / "effective-model.json", {
            "state_sha256": policy_digest(model), "condition": args.condition,
            "checkpoint_replay_max_error": 0, "command_source": "differential output neurons only"})
        retinal = {key: value.cpu().numpy() for key, value in payload["retina"].items()}
        results = []
        for case in suite["cases"]:
            environment = checked_environment(config)
            environment["max_steps"] = suite["steps"]
            environment["road"] = {**environment.get("road", {}), **case.get("road", {})}
            env = FlyBrainEnv(**environment)
            try:
                obs, info = env.reset(seed=int(case["seed"]), options={k: case[k] for k in
                    ("start_x", "lateral_offset", "heading_offset_degrees") if k in case})
                bridge, packer, effective = make_sensory_runtime(env, config)
                if effective["sensory_encoder"] != payload["sensory_encoder"]:
                    raise ValueError("Eye/organ calibration changed")
                initial = info["truth"]["progress_3d"]
                state, trace, timings = model.initial_state(), [], []
                for tick in range(suite["steps"]):
                    started = time.perf_counter()
                    actor = bridge.observe(env, obs, episode_id=case["name"], sequence_id=tick,
                                           received_monotonic_seconds=time.monotonic())
                    packed = packer.pack(actor)
                    np.testing.assert_array_equal(packed["ray_directions"], retinal["directions"])
                    image = (packed["ray_values"] * packed["ray_valid"][..., None]).reshape(-1)
                    signal = float(np.clip(((image - retinal["mean"]) / retinal["scale"])
                                             @ retinal["weight"] + retinal["bias"], -1, 1))
                    usable = float(np.any(packed["ray_valid"]))
                    if args.condition == "vision_missing":
                        signal, usable = 0., 0.
                    organ_valid = packed["organ_valid"].copy()
                    if args.condition == "body_missing":
                        organ_valid[:] = 0
                    encoded = time.perf_counter()
                    with torch.no_grad():
                        steering, state, outputs = model.advance(
                            torch.tensor([[signal]], dtype=torch.float32), torch.tensor([[usable]]),
                            torch.tensor(packed["organs"])[None],
                            torch.tensor(organ_valid)[None], state)
                    decided = time.perf_counter()
                    if args.condition == "circuit_cut":
                        assert not steering.any(), "Output bypass detected"
                    command = fixed_speed_action([0, float(steering[0])], env,
                                                  config["fixed_speed_mps"])
                    obs, _, terminated, truncated, info = env.step(command)
                    stepped = time.perf_counter()
                    timings.append([encoded - started, decided - encoded,
                                    stepped - decided, stepped - started])
                    trace.append([actor.simulation_seconds, signal, float(steering[0]),
                                  *outputs[0].tolist(), float(state.abs().max()),
                                  info["truth"]["progress_3d"], info["truth"]["lateral_error"]])
                    if tick % 1000 == 0:
                        write_json(run / "heartbeat.json", {"case": case["name"], "tick": tick + 1})
                    if terminated or truncated:
                        break
                trace = np.array(trace)
                timings = np.array(timings)
                np.savez_compressed(run / f"trace-{len(results):02d}.npz", trace=trace,
                    timing_seconds=timings,
                    timing_columns=np.array(["sensor_encoding", "neural_decision",
                                              "environment_step", "total"]),
                    columns=np.array(["observation_time", "retinal_signal", "steering",
                        "output_10360", "output_523769", "max_abs_neural_state",
                        "diagnostic_progress", "diagnostic_lateral"]))
                reason = info["reason"] if not env.active else "evaluation_budget"
                row = {"case": case["name"], "steps": len(trace), "reason": reason,
                       "completion": reason == "completion",
                       "progress_delta": info["truth"]["progress_3d"] - initial,
                       "lateral_rms": float(np.sqrt(np.mean(trace[:, -1]**2))),
                       "timing_quantiles_ms_p50_p95_p99": np.quantile(
                           timings * 1000, [.5, .95, .99], axis=0).T.tolist(),
                       "total_over_control_period_fraction": float(np.mean(timings[:, 3] > env.dt)),
                       "actor_over_control_period_fraction": float(np.mean(
                           timings[:, :2].sum(1) > env.dt))}
                results.append(row)
                write_json(run / "progress.json", {"cases": results})
                print(json.dumps(row), flush=True)
            finally:
                env.close()
        write_json(run / "evaluation.json", {"cases": results,
            "completion_rate": float(np.mean([r["completion"] for r in results])),
            "condition": args.condition, "scope": "engineering-calibrated circuit component"})
        write_json(run / "result.json", {"status": "completed", "goal_achieved": False,
                                         "checkpoint_sha256": sha256(run / "checkpoint.pt")})
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "reason": type(error).__name__,
                                         "message": str(error)})
        raise


if __name__ == "__main__":
    main()
