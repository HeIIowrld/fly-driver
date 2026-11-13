"""Replay Silverstone with frozen image predictions and identical T4/T5 candidates."""
from collections import Counter, defaultdict
from pathlib import Path
import json

import cv2
import numpy as np

from model import STATIONS, SUPPORT_CUT, camera_mask
from predict import HEAT_CUT, REFINE_RADIUS_PX, infer
from prepare import OUT, ROOT, SILVER, extract, labels, old

MODES = ("base", "far_wide", "direct_fill", "direct_replace", "learned_row", "plane_row")
COLUMNS = {"direct_fill": 0, "direct_replace": 0, "learned_row": 2, "plane_row": 3}
SECTIONS = old.STATIONS
CUT = .25


def combine(base, results):
    arrays = {"far_wide": base}
    selection = SECTIONS >= STATIONS[0]
    for name, j in COLUMNS.items():
        candidate = np.full_like(base, np.nan)
        candidate[selection] = results[..., j]
        arrays[name] = old.combine(base, candidate, fill=name == "direct_fill")
    return arrays


def score_boundaries(keys, archive, rows):
    truth, source = labels(SILVER)
    assert keys == list(zip(truth["run"], truth["frame"], strict=True))
    reference = truth["lateral_m"]
    visible = camera_mask(truth["image_px"][:, 0], truth["adjacent_invalid"])
    far = (SECTIONS >= 16.)[None, :, None]
    approach = np.array([r["truth"] == "approach" and r["label"] == "2" for r in rows])
    modes = list(archive)
    base = archive["far_wide"]
    errors = {}
    for group, include in (("all", np.ones(len(keys), dtype=bool)), ("right_approach", approach)):
        mask = visible & far & include[:, None, None]
        reports = {}
        for mode in modes:
            array = archive[mode]
            support = np.isfinite(array)
            same = mask & support & np.isfinite(base)
            sampled = mask & support
            abs_base, abs_new = np.abs(base-reference), np.abs(array-reference)
            reports[mode] = dict(
                labelled=mask.sum(axis=(0,1)).tolist(),
                retained=sampled.sum(axis=(0,1)).tolist(),
                retained_pairs=int((mask & support).all(axis=-1).sum()),
                paired_mae_m=[dict(points=int(same[...,s].sum()),
                                   base=float(abs_base[...,s][same[...,s]].mean()) if same[...,s].any() else None,
                                   model=float(abs_new[...,s][same[...,s]].mean()) if same[...,s].any() else None)
                              for s in range(2)],
                extra_points=(sampled & ~np.isfinite(base)).sum(axis=(0,1)).tolist())
        errors[group] = reports
    return errors, source


def main():
    cv2.setNumThreads(1)
    freeze = OUT / "training.json"
    checkpoint = OUT / "boundary.pt"
    trained = old.read(freeze)
    assert SILVER not in trained["training_circuits"] and trained["excluded_circuits"] == [SILVER]
    assert trained["support_cut"] == SUPPORT_CUT and trained["checkpoint_sha256"] == old.digest(checkpoint)
    for path, sha in trained["source_sha256"].items():
        assert SILVER not in path and old.digest(ROOT / path) == sha
    cache = OUT / f"{SILVER}-front.npz"
    if not cache.is_file():
        extract(SILVER)
    cache_receipt = cache.with_suffix(".json")
    assert old.digest(cache) == old.read(cache_receipt)["images_sha256"]
    mount = old.read(old.MOUNT)
    keys, available, predictions = infer(cache, checkpoint, mount["views"]["front"])
    predictions_path = OUT / "image-only-predictions.npz"
    np.savez_compressed(predictions_path, run=np.array([k[0] for k in keys]),
                        frame=np.array([k[1] for k in keys]), values=predictions,
                        columns=np.array(["lateral_m", "predicted_row_px", "heat_at_predicted_row_m", "heat_at_flat_row_m"]))
    entry = old.context()
    runner = entry.import_replay()
    hashes = {Path(__file__), Path(infer.__code__.co_filename), Path(extract.__code__.co_filename),
              Path(runner.__file__), Path(entry.__file__), Path(old.selections.__code__.co_filename),
              Path(old.combine.__code__.co_filename), Path(old.signals.__code__.co_filename),
              freeze, checkpoint, cache, cache_receipt, old.MOUNT}
    rows, prior, mapping, bundles, profiles, training = runner.setup(SILVER, hashes)
    assert [runner.key(r) for r in rows] == keys
    original_path = old.OLD / f"{SILVER}-frames.jsonl"
    original_result = old.OLD / "result.json"
    assert old.digest(original_path) == old.read(original_result)["tracks"][SILVER]["rows_sha256"]
    hashes.update((original_path, original_result))
    observed = {runner.key(r): r for r in map(json.loads, original_path.read_text(encoding="utf8").splitlines())}
    expected = {key: i for i,key in enumerate(keys)}
    near, far = (old.SurfaceContrast(mount, near=value) for value in (True, False))
    far.stations = SECTIONS[3:]
    far.lookup = far._sampling(mount, far.stations)
    timelines = {f"{m}_{g}": defaultdict(list) for m in runner.MODELS for g in MODES}
    sections = {m: np.full((len(keys), len(SECTIONS),2), np.nan) for m in MODES[1:]}
    recorded, paired = [], Counter()
    for tag, folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[tag]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                key = tag, item["capture_id"]
                if key not in expected:
                    continue
                i = expected[key]
                t, control = rows[i], prior[i]
                name = Path(item["file"]).name
                assert archive.by_name[name]["sha256"] == item["file_sha256"]
                meta = archive.json(name)
                assert (meta["track_id"], meta["index"], meta["generation"]) == (SILVER, key[1], item["generation"])
                motion = old.read_motion(meta) if available[i] else None
                offers = {}
                if motion is not None and motion[1] >= old.MIN_SPEED_MPS:
                    images = {v["view_id"]: cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name), np.uint8),
                                                        cv2.IMREAD_COLOR) for v in item["views"]}
                    assert set(images) == {"front", "left", "right", "rear"} and all(v is not None for v in images.values())
                    lab = {n: cv2.cvtColor(image, cv2.COLOR_BGR2LAB) for n,image in images.items()}
                    base = old.selections(near, far, images, lab, near.detect(images), old.edge_scores(near, lab))
                    for mode, bounds in combine(base, predictions[i]).items():
                        sections[mode][i] = bounds
                        offers[mode] = entry.local(SECTIONS, bounds, 10.)
                known = observed[key]
                candidate = offers.get("far_wide")
                assert (list(candidate) if candidate else None) == known["offers"].get("far_wide"), key
                base = {m: control["proposals"][spec[1]] for m,spec in runner.MODELS.items()}
                decisions = {}
                for mode in MODES:
                    offer = offers.get(mode)
                    for m in runner.MODELS:
                        score = old.signals(profiles[m].get(key), *offer)["local"] if offer else None
                        decision = base[m] or (offer if score is not None and score >= CUT else None)
                        method = f"{m}_{mode}"
                        decisions[method] = decision
                        timelines[method][tag].append(old.publish(t, decision, control["baseline_observable"] or decision is not None))
                        if mode == "far_wide":
                            assert (list(decision) if decision else None) == known["final"][method], key
                    if mode != "base" and not any(base.values()):
                        a,b = (decisions[f"{m}_{mode}"] for m in runner.MODELS)
                        if a != b:
                            paired[mode, "t4t5" if b else "edge", old.category(t, a or b)] += 1
                recorded.append(dict(run=tag, frame=key[1], truth=t["truth"], label=t["label"],
                                     truth_endpoints_m=t["truth_endpoints_m"], offers=offers, decisions=decisions))
    assert [runner.key(r) for r in recorded] == keys
    errors, physical = score_boundaries(keys, sections, rows)
    hashes.add(physical)
    section_path = OUT / "observed-cross-sections.npz"
    np.savez_compressed(section_path, run=np.array([k[0] for k in keys]), frame=np.array([k[1] for k in keys]),
                        stations=SECTIONS, **sections)
    rows_path = OUT / "silverstone-frames.jsonl"
    old.jsonl(rows_path, recorded)
    metrics = {k: runner.totals(v) for k,v in timelines.items()}
    data = dict(status="silverstone_excluded_from_boundary_training_and_selection", seed=trained["seed"],
                source_revision=old.revision(), training_circuits=trained["training_circuits"],
                confirmation_training=training, frames=len(rows), metrics=metrics, boundary_metrics=errors,
                configuration=dict(support_cut=SUPPORT_CUT, heat_cut=HEAT_CUT,
                                   refinement_radius_px=REFINE_RADIUS_PX, confirm_cut=CUT, changed_from_m=old.START),
                paired=[dict(mode=k[0], accepted_by=k[1], category=k[2], frames=v) for k,v in sorted(paired.items())],
                predictions_sha256=old.digest(predictions_path), rows_sha256=old.digest(rows_path),
                sections_sha256=old.digest(section_path),
                source_sha256={str(p):old.digest(p) for p in sorted(hashes)},
                termination="silverstone_complete_with_unchanged_comparison_control")
    old.write(OUT / "result.json", data)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
