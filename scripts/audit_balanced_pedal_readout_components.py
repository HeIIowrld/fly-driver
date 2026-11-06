"""Separate affine and balance-RBF logit contributions on immutable recorded muscles."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from flybrain_drive.sac_unified_student import read
from scripts.fit_current_flight_launch_pedals import datasets


def audit(fit):
    cfg = read(fit / "manifest.json")["config"]
    result = read(fit / "result.json")
    cp = fit / "pedals.pt"
    if sha256(cp) != result["checkpoint_sha256"] or sha256(cfg["fixed_decoder"]) != cfg["fixed_decoder_sha256"]:
        raise ValueError("Sealed fit and original decoder required")
    state = torch.load(cp, weights_only=True, map_location="cpu")["state"]
    parent, _ = load_muscle_interaction_head(cfg["fixed_decoder"])
    ref, scale = (getattr(parent.base, "training_" + key).detach().numpy()
                  for key in ("reference", "scale"))
    orig_w = parent.base.projection.weight.detach().numpy()[1:]
    orig_b = parent.base.projection.bias.detach().numpy()[1:]
    w = state["pedal_projection"]["weight"].numpy()[1:]
    b = state["pedal_projection"]["bias"].numpy()[1:]
    centers, width = state["centers"].numpy(), float(state["width"])
    rows = []
    for item in datasets(Path(cfg["evaluation"])):
        z = ((item["muscles"] - ref) / scale).astype(np.float64)
        phi = np.exp(-.5*np.sum((z[:, None, :4]-centers)**2, axis=-1) / width**2)
        linear, original, offset = z@w.T+b, z@orig_w.T+orig_b, phi@state["offset"]["weight"].numpy().T
        if not np.allclose(expit(original), item["original"][:, 1:], atol=2e-6, rtol=0):
            raise ValueError("Original pedal registration differs")
        prediction = {key: expit(value) for key, value in (
            ("original", original), ("learned_affine_only", linear),
            ("learned_rbf_only", original+offset), ("both", linear+offset))}
        y = item["data"]["labels"][:, 1:]
        masks = dict(low=item["low"], high=item["high"],
                     brake=item["high"] & (y[:, 1] > .1),
                     fastest=item["high"] & (item["data"]["body"][:, 16] >= 40))
        groups = {}
        for name, mask in masks.items():
            groups[name] = dict(labels=int(mask.sum()),
                rmse={key: np.sqrt(np.mean((value[mask]-y[mask])**2, axis=0)).tolist()
                      for key, value in prediction.items()},
                peak_center_activation_median=float(np.median(np.max(phi[mask], axis=1))))
        rows.append(dict(role=item["receipt"]["role"], groups=groups))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", action="append", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    out = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
                                     seed=None, vehicle_control_sent=False))
    result = dict(status="failed", fits={}, scope="Offline algebra on saved muscles only")
    try:
        for fit in args.fit:
            result["fits"][fit] = audit(Path(fit))
        result["status"] = "completed"
    except Exception as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(out / "result.json", result)


if __name__ == "__main__":
    main()
