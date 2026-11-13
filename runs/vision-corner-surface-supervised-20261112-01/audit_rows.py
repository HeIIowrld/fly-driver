"""Measure held-out row estimation and physical-section errors before curve selection."""
import json
from pathlib import Path

import numpy as np

from verify import OUT, RUNS, ROOT, SILVER, digest, read


def main():
    predicted = OUT / "image-only-predictions.npz"
    result_path = OUT / "result.json"
    result = read(result_path)
    assert digest(predicted) == result["predictions_sha256"]
    source = RUNS / "vision-corner-physics-surface-20261112-01" / f"{SILVER}-train-labels.npz"
    mount_path = ROOT / "configs/ac-csp-exterior-front-v3-retina.json"
    camera = read(mount_path)["views"]["front"]
    assert digest(source) == read(source.with_suffix(".json"))["labels_sha256"]
    rows_path = OUT / "silverstone-frames.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf8").splitlines()]
    with np.load(predicted, allow_pickle=False) as raw, np.load(source, allow_pickle=False) as truth:
        np.testing.assert_array_equal(raw["frame"], truth["frame"])
        np.testing.assert_array_equal(raw["run"], truth["run"])
        assert [(r["run"],r["frame"]) for r in rows] == list(zip(raw["run"],raw["frame"],strict=True))
        keep = truth["stations"] >= 10.
        station = truth["stations"][keep]
        pixels = truth["image_px"][:,0,keep]
        target = truth["lateral_m"][:,keep]
        witness = truth["adjacent_invalid"][:,keep]
        values = raw["values"]
    mask = witness & np.isfinite(pixels).all(axis=-1)
    mask &= (pixels[...,1] >= 73.) & (pixels[...,1] < 151.)
    mask &= (station >= 16.)[None,:,None]
    flat = (.5+.5*camera["projection_y"]*camera["origin_body_m"][2]/(station-camera["origin_body_m"][0]))*179.
    error = abs(values[...,1]-pixels[...,1])
    error_flat = abs(flat[None,:,None]-pixels[...,1])
    section_error = abs(values[...,0]-target)
    summaries = {}
    for name, group in (("all", np.ones(len(rows),bool)),
                        ("right_approach", np.array([r["truth"]=="approach" and r["label"]=="2" for r in rows]))):
        stats = {}
        for side, label in enumerate(("right","left")):
            valid = (mask & group[:,None,None])[...,side]
            both = valid & np.isfinite(error[...,side])
            stats[label] = dict(
                physical_targets=int(valid.sum()), model_outputs=int(both.sum()),
                baseline_row_mae_px=float(error_flat[...,side][both].mean()) if both.any() else None,
                learned_row_mae_px=float(error[...,side][both].mean()) if both.any() else None,
                raw_lateral_mae_m=float(section_error[...,side][both].mean()) if both.any() else None,
            )
        summaries[name] = stats
    (OUT / "row-audit.json").write_text(json.dumps(
        dict(status="verified", seed=result["seed"], source_revision=result["source_revision"],
             metrics=summaries, source_sha256={str(p):digest(p) for p in
               (Path(__file__), predicted, source, mount_path, rows_path, result_path)}),
        indent=2)+"\n", encoding="utf8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
