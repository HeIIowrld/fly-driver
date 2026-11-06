"""Look for local pedal regressions hidden by aggregate high-speed RMSE."""
import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.fit_current_flight_launch_pedals import datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
                                     seed=None, vehicle_control_sent=False))
    result = dict(status="failed", used_for_fit_or_selection=False, fits={})
    try:
        for fit in args.fit:
            path = Path(fit)
            cfg = read(path / "manifest.json")["config"]
            artifact = path / "predictions.npz"
            if sha256(artifact) != read(path / "result.json")["predictions_sha256"]:
                raise ValueError("Sealed fit predictions required")
            rows = result["fits"][fit] = []
            with np.load(artifact, allow_pickle=False) as prediction:
                for i, item in enumerate(datasets(Path(cfg["evaluation"]))):
                    speed = item["data"]["body"][:, 16]
                    labels = item["data"]["labels"]
                    old = item["original"]
                    new = prediction[f"prediction_{i}"]
                    masks = {"below_25m_s": speed < 25, "25_to_40m_s": (speed >= 25) & (speed < 40),
                             "at_least_40m_s": speed >= 40, "teacher_brake_above_point1": labels[:, 2] > .1,
                             "teacher_throttle_above_point8": labels[:, 1] > .8}
                    row = dict(role=item["receipt"]["role"], bands={})
                    for band, mask in masks.items():
                        mask = mask & item["high"]
                        if not mask.any():
                            row["bands"][band] = dict(labels=0)
                            continue
                        original = np.sqrt(np.mean((old[mask, 1:] - labels[mask, 1:])**2, axis=0))
                        learned = np.sqrt(np.mean((new[mask, 1:] - labels[mask, 1:])**2, axis=0))
                        row["bands"][band] = dict(labels=int(mask.sum()), original_rmse=original.tolist(),
                                                   learned_rmse=learned.tolist(),
                                                   rmse_diff=(learned-original).tolist())
                    rows.append(row)
        result["status"] = "completed"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out / "result.json", result)


if __name__ == "__main__":
    main()
