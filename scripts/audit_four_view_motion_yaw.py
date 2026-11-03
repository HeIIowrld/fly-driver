"""Compare original fly visual motion responses with native body yaw across histories."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.train_joint_view_expanded_histories import read


def correlation(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y))
    return np.divide(
        (x * y[:, None]).sum(axis=0), denom, out=np.full(x.shape[1], np.nan), where=denom > 1e-12
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-sensors", required=True)
    parser.add_argument("--new-sensors", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = [args.old_sensors, args.new_sensors]
    pins = {p: sha256(p) for p in inputs}
    entries = read(args.old_sensors)["episodes"] + read(args.new_sensors)["episodes"]
    if len(entries) != 14 or len({e["id"] for e in entries}) != 14:
        raise ValueError("Original distinct 14 episodes required")
    records = []
    for e in entries:
        visual = Path(e["output"]) / "responses.npz"
        body_dir = Path(e["body_replay"])
        body = body_dir / "virtual-body.npz"
        body_result = body_dir / "result.json"
        receipt = read(body_result)
        if (
            sha256(visual) != e["responses_sha256"]
            or receipt["status"] != "completed"
            or sha256(body) != receipt["virtual_body_sha256"]
        ):
            raise ValueError("Original stored vision or body checksum differs")
        pins.update({str(p): sha256(p) for p in (visual, body, body_result)})
        with load_array(visual, allow_pickle=False) as a:
            motion = np.asarray(a["motion_activity"], dtype=np.float64).mean(axis=3).reshape(-1, 32)
            clock = a["decision_monotonic_seconds"].copy()
        with load_array(body, allow_pickle=False) as a:
            if (
                not np.array_equal(clock, a["decision_monotonic_seconds"])
                or a["feature_names"][4] != "angular_body.yaw_left_rad_s"
            ):
                raise ValueError("Original decision-clock and yaw convention required")
            yaw, valid = a["values"][:, 4].copy(), a["valid"][:, 4].copy()
        if (
            len(clock) != e["observations"]
            or motion.shape != (len(clock), 32)
            or not np.isfinite(motion).all()
            or not np.isfinite(yaw).all()
            or np.any(np.diff(clock) <= 0)
        ):
            raise ValueError("Finite original motion and increasing decisions required")
        results = []
        for lag in (0.0, 0.2):
            # Latest earlier vision decision at or before yaw decision minus lag.
            source = np.searchsorted(clock, clock - lag, side="right") - 1
            chosen = (source >= 0) & valid
            actual_age = clock[chosen] - clock[source[chosen]]
            chosen[chosen] &= actual_age <= lag + 0.06
            if chosen.sum() < 50:
                raise ValueError("Insufficient original timed valid yaw samples")
            c = correlation(motion[source[chosen]], yaw[chosen])
            if not np.isfinite(c).all():
                raise ValueError("Nonconstant motion and yaw needed for this diagnostic")
            results.append(c)
        records.append(
            {
                "id": e["id"],
                "role": e["role"],
                "correlations": np.stack(results),
                "yaw_valid_count": int(valid.sum()),
            }
        )
        print(json.dumps({"loaded": e["id"]}), flush=True)
    train = [r for r in records if r["role"] == "train"]
    dev = [r for r in records if r["role"] == "development"]
    if len(train) != 11 or len(dev) != 3:
        raise ValueError("Original 11 TRAIN / 3 known DEV roles required")
    mean = np.mean(np.stack([r["correlations"] for r in train]), axis=0)
    lag_index, channel = np.unravel_index(np.argmax(np.abs(mean)), mean.shape)
    selection = {
        "lag_seconds": [0.0, 0.2][lag_index],
        "view": ["front", "left", "right", "rear"][channel // 8],
        "cell_index": int(channel % 8),
        "train_episode_mean_correlation": float(mean[lag_index, channel]),
    }
    reports = [
        {
            "id": r["id"],
            "role": r["role"],
            "yaw_valid_count": r["yaw_valid_count"],
            "selected_correlation": float(r["correlations"][lag_index, channel]),
        }
        for r in records
    ]
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError("Original input changed")
    run = begin_run(
        args.output,
        {
            "program_sha256": sha256(__file__),
            "source_hashes": pins,
            "model": "original 8 motion populations averaged over 721 receptors per camera",
            "selection": "largest absolute mean signed Pearson correlation over 11 TRAIN episodes",
            "lags": [0.0, 0.2],
            "late_selection_or_fit_on_dev": False,
            "actor_deployed": False,
        },
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(
        run / "correlations.npz",
        train_mean=mean,
        episode_correlations=np.stack([r["correlations"] for r in records]),
    )
    out = {
        "status": "completed_four_view_motion_yaw_audit",
        "selection": selection,
        "reports": reports,
        "correlations_sha256": sha256(run / "correlations.npz"),
        "host_decision_clock_not_physics_exposure": True,
        "correlation_not_direction_selectivity": True,
        "known_dev_not_independent_final_test": True,
        "optimizer_updates": 0,
        "actor_deployed": False,
        "goal_achieved": False,
    }
    write_json(run / "result.json", out)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
