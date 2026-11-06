"""Independently evaluate the sealed joint pedal coefficients with NumPy."""
import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from scripts.fit_current_flight_launch_pedals import datasets


def audit(path):
    config = read(path / "manifest.json")["config"]
    result = read(path / "result.json")
    checkpoint = path / "pedals.pt"
    if (sha256(checkpoint) != result["checkpoint_sha256"] or
            sha256(path / "predictions.npz") != result["predictions_sha256"] or
            sha256(config["fixed_decoder"]) != config["fixed_decoder_sha256"]):
        raise ValueError("Fit artifact hash differs")
    cp = torch.load(checkpoint, weights_only=True, map_location="cpu")["state"]
    parent, _ = load_muscle_interaction_head(config["fixed_decoder"])
    ref = parent.base.training_reference.detach().numpy()
    scale = parent.base.training_scale.detach().numpy()
    centers = cp["centers"].numpy()
    weights = cp["pedal_projection"]["weight"].numpy()[1:]
    bias = cp["pedal_projection"]["bias"].numpy()[1:]
    width = float(cp["width"])
    offset = cp["offset"]["weight"].numpy()
    if not ((weights[0, 4:] > 0).all() and (weights[1, 4:] < 0).all()):
        raise ValueError("Signed power derivatives differ")
    items = datasets(Path(config["evaluation"]))
    rows = []
    with np.load(path / "predictions.npz") as stored:
        for i, item in enumerate(items):
            z = ((item["muscles"] - ref) / scale).astype(np.float64)
            phi = np.exp(-.5 * np.sum((z[:, None, :4] - centers)**2, axis=-1) / width**2)
            logits = z @ weights.T + bias + phi @ offset.T
            pedals = 1 / (1 + np.exp(-logits))
            expected = stored[f"prediction_{i}"]
            diff = float(np.max(np.abs(pedals - expected[:, 1:])))
            if diff > 3e-6 or not np.array_equal(expected[:, 0], item["original"][:, 0]):
                raise ValueError("Reconstructed pedals or preserved steering differ")
            row = dict(role=item["receipt"]["role"], decisions=len(z), max_pedal_diff=diff)
            for name in ("low", "high"):
                mask = item[name]
                labels = item["data"]["labels"][mask, 1:]
                mse = np.mean((pedals[mask] - labels)**2, axis=0)
                original = np.mean((item["original"][mask, 1:] - labels)**2, axis=0)
                if row["role"] == "train" and name == "high" and (mse / original > .98001).any():
                    raise ValueError("Independent TRAIN MSE constraint exceeded")
                row[name] = dict(labels=int(mask.sum()), rmse=np.sqrt(mse).tolist(),
                                 mse_ratio=(mse / original).tolist())
            rows.append(row)
    return dict(checkpoint_sha256=sha256(checkpoint), episodes=rows, decisions=sum(row["decisions"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
                                     seed=None, vehicle_control_sent=False))
    result = dict(status="failed", fits={}, vehicle_control_sent=False)
    try:
        for fit in args.fit:
            result["fits"][fit] = audit(Path(fit))
        result["status"] = "completed"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out / "result.json", result)


if __name__ == "__main__":
    main()
