"""Package two existing observation streams and verify the label-free executable."""
from pathlib import Path
import sys
import json

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from replay_corner_boundary_states import ROUTING, ROOT, load_observer, observe, old, main as cli
from routing import SILVER, VAL
from directional_features import PARENT
from verify_directional import frames
import alternatives as alt


def inputs(track, size):
    sources = {Path(__file__)}
    features = alt.load_features(track, sources)
    path = PARENT/f"{track}-observations.npz"
    receipt = path.with_suffix(".json")
    assert old.digest(path) == old.read(receipt)["arrays_sha256"]
    keys = [(r["run"], r["frame"]) for r in features["rows"]]
    with np.load(path, allow_pickle=False) as data:
        assert keys == list(zip(data["run"], data["frame"], strict=True))
        sensor = {name: data[name] for name in ("run", "frame", "sections", "motion", "heads")}
    for name in ("generation", "received_s"):
        sensor[name] = np.array([row[name] for row in features["rows"]])
    by_key = {(r["run"], r["frame"]): v for r, v in zip(features["selected"], features["x"], strict=True)}
    sensor["stations"] = old.STATIONS
    sensor["t4t5"] = np.stack([by_key.get(k, np.full((2, size), np.nan)) for k in keys])
    target = OUT/f"{track}-sensors.npz"
    np.savez_compressed(target, **sensor)
    return target, sensor, sources | {path, receipt}


def observed_status(track, sensor, records, expected):
    assert len(expected) == len(records) == 1200
    missing = [i for i, row in enumerate(records) if not row["motion_vision_available"]]
    moving_missing = sum(np.isfinite(sensor["motion"][i]).all() and sensor["motion"][i, 1] >= old.MIN_SPEED_MPS
                         for i in missing)
    deltas = []
    for i, (actual, baseline) in enumerate(zip(records, expected, strict=True)):
        assert (actual["run"], actual["frame"]) == (baseline["run"], baseline["frame"])
        if actual["forecasts"] != baseline["forecasts"]:
            deltas.append(dict(frame=actual["frame"], motion_vision_available=actual["motion_vision_available"],
                               earlier=baseline["forecasts"], new=actual["forecasts"]))
    assert all(not row["motion_vision_available"] for row in deltas), deltas[:6]
    return dict(frames=len(records), t4t5_missing=len(missing), moving_t4t5_missing=int(moving_missing),
                parity_frames=len(records)-len(deltas), differences=deltas)


def verify_missing_vision(path, sensor, models, settings, records, original_sections):
    first = next(i for i, r in enumerate(records) if r["motion_vision_available"] and
                 r["forecasts"]["temporal"]["state"] != "abstain")
    stop = min(first+20, len(records))
    subset = {k: v[:stop].copy() if k != "stations" else v.copy() for k, v in sensor.items()}
    with_source = path.with_name(path.stem+"-prefix.npz")
    np.savez_compressed(with_source, **subset)
    with np.load(with_source, allow_pickle=False) as source:
        prefix, boundaries, _ = observe(source, models, settings)
    assert json.loads(json.dumps(prefix)) == records[:stop]
    np.testing.assert_array_equal(boundaries, original_sections[:stop])
    subset["t4t5"][:] = np.nan
    target = path.with_name(path.stem+"-no-motion-vision.npz")
    np.savez_compressed(target, **subset)
    with np.load(target, allow_pickle=False) as source:
        ablated, boundaries, _ = observe(source, models, settings)
    np.testing.assert_array_equal(boundaries, original_sections[:stop])
    for old_row, new_row in zip(records[:stop], ablated, strict=True):
        assert new_row["forecasts"]["temporal"] == old_row["forecasts"]["temporal"]
        assert new_row["forecasts"]["t4t5_confirm"] == dict(state="abstain", distance_m=None)
    assert any(r["motion_vision_available"] for r in records[:stop])
    assert any(r["forecasts"]["temporal"]["state"] != "abstain" for r in records[:stop])
    subset["motion"] = np.full_like(subset["motion"], np.nan)
    subset["t4t5"] = sensor["t4t5"][:stop]
    body_path = path.with_name(path.stem+"-no-body-motion.npz")
    np.savez_compressed(body_path, **subset)
    with np.load(body_path, allow_pickle=False) as source:
        lost_body, unfilled, restored = observe(source, models, settings)
    np.testing.assert_array_equal(unfilled, sensor["sections"][:stop])
    assert not restored.any()
    assert all(r["forecasts"] == dict.fromkeys(("temporal", "t4t5_confirm"),
               dict(state="abstain", distance_m=None)) and not r["body_motion_available"] for r in lost_body)
    return dict(frames=stop, input_sha256=old.digest(target), body_loss_sha256=old.digest(body_path),
                prefix_sha256=old.digest(with_source))


def main():
    models, settings, model_sources = load_observer()
    witness = ROUTING/"straight-verification.json"
    verified = old.read(witness)
    reports = {}
    for track in (SILVER, VAL):
        source, sensor, sources = inputs(track, settings["motion_features"])
        destination = OUT/track
        # Invoke the same entry point used by callers and verify its emitted artifacts.
        sys.argv = ["replay_corner_boundary_states.py", "--input", str(source), "--output", str(destination)]
        cli()
        run = old.read(destination/"result.json")
        assert run["input_sha256"] == old.digest(source)
        records = frames(destination/"forecasts.jsonl", run["forecasts_sha256"])
        prior = ROUTING/f"{track}-state-distance.jsonl"
        expected = frames(prior, verified["tracks"][track]["rows_sha256"])
        stats = observed_status(track, sensor, records, expected)
        path = destination/"boundaries.npz"
        assert old.digest(path) == run["boundaries_sha256"]
        with np.load(path, allow_pickle=False) as data:
            earlier = old.read(ROOT/"runs/vision-corner-aligned-sections-20261112-01/result.json")
            aligned = ROOT/"runs/vision-corner-aligned-sections-20261112-01"/f"{track}-sections.npz"
            assert old.digest(aligned) == earlier["tracks"][track]["sections_sha256"]
            with np.load(aligned, allow_pickle=False) as reference:
                np.testing.assert_array_equal(data["estimated"], reference["aligned"])
                np.testing.assert_array_equal(data["restored"], reference["added"])
            stats["motion_vision_loss"] = verify_missing_vision(source, sensor, models, settings, records,
                                                                 data["estimated"])
        sources.update((source, witness, prior, aligned, destination/"result.json", *model_sources))
        stats["source_sha256"] = {str(p): old.digest(p) for p in sorted(sources)}
        reports[track] = stats
        print(track, {k: v for k, v in stats.items() if k not in ("source_sha256", "differences")}, flush=True)
    old.write(OUT/"verification.json", dict(status="verified", source_revision=old.revision(), tracks=reports,
              termination="two_label_free_replays_and_motion_vision_loss_verified"))


if __name__ == "__main__":
    main()
