"""Explicit commands; planned backends never silently fall back."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from .env import BODY_CHANNELS, BODY_SCALE, FlyBrainEnv
from .evidence import begin_run, write_json
from .graph import import_csv, load_graph, rewire, save_graph, sha256
from .road import RoadConfig, SyntheticRoad


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def rollout(args):
    if args.steps < 1:
        raise ValueError("steps must be positive")
    env_config = {**read_config(args.environment), "mode": args.mode}
    run = begin_run(
        args.output,
        {
            "environment": env_config,
            "seed": args.seed,
            "steps": args.steps,
            "driver": "oracle_route" if args.mode == "O" else "fixed_intent_camera_smoke",
        },
    )
    env = None
    try:
        env = FlyBrainEnv(**env_config)
        obs, info = env.reset(seed=args.seed)
        trace, durations = [], []
        initial_images = obs["images"].copy()
        for _ in range(args.steps):
            if args.mode == "O":
                physical = obs["body"] * BODY_SCALE
                curvature = np.clip(
                    physical[-1] - 0.3 * physical[-5] - 0.6 * physical[-4],
                    -env.vehicle.max_curvature,
                    env.vehicle.max_curvature,
                )
                action = np.array([-0.5, curvature / env.vehicle.max_curvature], np.float32)
            else:
                action = np.array([-0.75, 0.0], np.float32)
            start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            durations.append(time.perf_counter() - start)
            trace.append(
                {
                    **info,
                    "action": action.tolist(),
                    "reward": reward,
                    "body": obs["body"].tolist(),
                    "body_valid": obs["body_valid"].tolist(),
                    "camera_valid": obs["camera_valid"].tolist(),
                    "camera_age": obs["camera_age"].tolist(),
                }
            )
            if terminated or truncated:
                break
        write_json(run / "trace.json", trace)
        write_json(run / "environment.json", env.configuration())
        write_json(
            run / "rollout.json",
            {
                "steps": len(trace),
                "reason": info["reason"] if not env.active else "rollout_budget",
                "final": info,
                "reward_sum": sum(row["reward"] for row in trace),
                "sensor_layout": BODY_CHANNELS,
                "step_wall_ms_p50_p95_p99": (
                    np.quantile(durations, [0.5, 0.95, 0.99]) * 1000
                ).tolist(),
                "camera_mean": np.mean(obs["images"], axis=(1, 2, 3)).tolist(),
            },
        )
        if args.mode == "V":
            np.savez_compressed(
                run / "camera_samples.npz", initial=initial_images, final=obs["images"]
            )
        write_json(run / "result.json", {"status": "completed"})
        print(f"{run}: {len(trace)} steps, {info['reason']}")
    except Exception as error:
        write_json(
            run / "result.json",
            {"status": "failed", "reason": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    ac_probe = commands.add_parser("ac-probe")
    ac_probe.add_argument("--steam-root", action="append")
    ac_probe.add_argument("--game-path", action="append", default=[])
    ac_probe.add_argument("--output", required=True)
    generate = commands.add_parser("track")
    generate.add_argument("--config")
    generate.add_argument("--output", required=True)
    roll = commands.add_parser("rollout")
    roll.add_argument("--mode", choices=["O", "V"], default="O")
    roll.add_argument("--environment")
    roll.add_argument("--steps", type=int, default=300)
    roll.add_argument("--seed", type=int, default=1)
    roll.add_argument("--output", required=True)
    train_command = commands.add_parser("train")
    train_command.add_argument("--config", required=True)
    train_command.add_argument("--output", required=True)
    evaluate_command = commands.add_parser("evaluate")
    evaluate_command.add_argument("--checkpoint", required=True)
    evaluate_command.add_argument("--steps", type=int, default=300)
    evaluate_command.add_argument("--output", required=True)
    suite_command = commands.add_parser("evaluate-suite")
    suite_command.add_argument("--checkpoint", required=True)
    suite_command.add_argument("--suite", required=True)
    suite_command.add_argument("--output", required=True)
    suite_command.add_argument("--record", action="store_true")
    suite_command.add_argument("--record-every", type=int, default=4)
    graph_command = commands.add_parser("import-graph")
    graph_command.add_argument("--nodes", required=True)
    graph_command.add_argument("--edges", required=True)
    graph_command.add_argument("--dataset", default="user_reviewed_unlabeled")
    graph_command.add_argument("--output", required=True)
    rewiring = commands.add_parser("rewire")
    rewiring.add_argument("--graph", required=True)
    rewiring.add_argument("--seed", type=int, required=True)
    rewiring.add_argument("--swaps", type=int)
    rewiring.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        result = {
            "backend": "mujoco_only",
            "modes": ["O", "V"],
            "calibration": "illustrative_unvalidated",
        }
        for name in ("numpy", "gymnasium", "mujoco", "torch"):
            try:
                result[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result[name] = "not_installed"
        print(json.dumps(result, indent=2))
    elif args.command == "ac-probe":
        from .ac_probe import inspect_installation

        run = begin_run(args.output, {"probe": "read_only_ac_installation"})
        result = inspect_installation(args.steam_root, args.game_path)
        write_json(run / "capabilities.json", result)
        write_json(run / "result.json", {
            "status": "completed", "scope": "installation discovery, not runtime integration",
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "track":
        road = SyntheticRoad(RoadConfig(**read_config(args.config)))
        with Path(args.output).open("x", encoding="utf-8") as stream:
            json.dump(road.export(), stream)
    elif args.command == "rollout":
        rollout(args)
    elif args.command == "train":
        from .learning import train

        config = read_config(args.config)
        if int(config.get("num_envs", 1)) > 1:
            from .parallel_learning import train_parallel

            train_parallel(config, args.output)
        else:
            train(config, args.output)
    elif args.command == "evaluate":
        from .learning import evaluate

        evaluate(args.checkpoint, args.output, args.steps)
    elif args.command == "evaluate-suite":
        from .evaluation import evaluate_suite

        evaluate_suite(
            args.checkpoint,
            args.output,
            read_config(args.suite),
            record=args.record,
            record_every=args.record_every,
        )
    elif args.command == "import-graph":
        save_graph(
            args.output,
            import_csv(args.nodes, args.edges),
            {
                "dataset": args.dataset,
                "nodes_csv_sha256": sha256(args.nodes),
                "edges_csv_sha256": sha256(args.edges),
                "ports": "user_assigned_not_auto_anatomy",
            },
        )
    elif args.command == "rewire":
        graph, metadata = rewire(load_graph(args.graph), args.seed, args.swaps)
        save_graph(args.output, graph, {**metadata, "source_sha256": sha256(args.graph)})


if __name__ == "__main__":
    main()
