"""Replay a frozen appearance classifier on Silverstone and Spa, with paired alert confirmation."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from baseline import PROTOCOL, SILVER, fit_boundaries, publish, reference
from confirm import signals
from corner_fusion_boundaries import stations
from evaluate_fused_corner import frame_key, load_frozen
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from image_normal import FIELDS, choose, edge_scores
import learn_edges as learned
from paired_motion import inputs
from patches import GRID, PatchGrid
import profile_replay as profile
from recover import baseline_data
from relative_corner_boundaries import SurfaceContrast
from score_corner_captures import summarize
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
TEST_TRACKS = (SILVER, "spa")
MODES = ("original", "normal", "learned", "ranked", "ranked_boundary", "ranked_t4t5")
CONFIRM_THRESHOLD = .25


def training_set(track, hashes):
    path = learned.OUT / f"{track}-patches.npz"
    receipt = path.with_suffix(".json")
    proof = read(receipt)
    assert proof["track"] == track and digest(path) == proof["patches_sha256"]
    hashes.update((path, receipt))
    for filename, expected in proof["input_sha256"].items():
        source = Path(filename)
        assert digest(source) == expected, source
        hashes.add(source)
    with np.load(path, allow_pickle=False) as data:
        return learned.examples(data)


def probabilities(model, detector, lab):
    features, _ = detector.features(lab)
    valid = detector.valid
    result = np.zeros(valid.shape, dtype=np.float32)
    result[valid] = model.predict_proba(learned.encode(features[valid]))[:, 1]
    return result


def ranked_patches(raw, contrast, grid, probability):
    """Retain measured edges and select only among current image-supported replacements."""
    filtered = contrast.copy()
    for i, row in enumerate(raw):
        for side, sign in ((0, -1.), (1, 1.)):
            value = row[FIELDS[side]]
            other = row[FIELDS[1-side]]
            if value is not None or other is None:
                continue
            width = (grid-other)*sign
            mask = (width > 3.) & (width < 20.)
            score = np.interp(grid, GRID, probability[i, side], left=0., right=0.)
            # The original patch classifier's fixed probability cutoff is unchanged.
            supported = mask & (score >= learned.PATCH_THRESHOLD)
            filtered[i, mask] = np.where(supported[mask], score[mask], -np.inf)
    return choose(raw, filtered, grid, "normal_fill")


def confirm(profile_values, proposal):
    if proposal is None:
        return None
    side, distance = proposal
    score = signals(profile_values, side, distance)["local"]
    return proposal if score is not None and score >= CONFIRM_THRESHOLD else None


def replay(track, mount, model, profiles):
    hashes = {Path(__file__), Path(learned.__file__), Path(fit_boundaries.__code__.co_filename),
              Path(baseline_data.__code__.co_filename), Path(summarize.__code__.co_filename)}
    group, cache, mapping, bundles = baseline_data(track, hashes)
    near, far, grid = SurfaceContrast(mount, near=True), SurfaceContrast(mount), PatchGrid(mount)
    geometry = profile.TRACK_ROOT / track / "ai/fast_lane.ai"
    hashes.add(geometry)
    track_edges, _ = load_edges(geometry)
    expected = {frame_key(r): r for r in group["rows"]}
    reports = {mode: {} for mode in MODES}
    details = []
    for run, folder in read(mapping).items():
        by_mode = {mode: [] for mode in MODES}
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[run]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                key = run, item["capture_id"]
                if key not in expected:
                    continue
                row = expected[key]
                filename = Path(item["file"]).name
                assert archive.by_name[filename]["sha256"] == item["file_sha256"]
                meta = archive.json(filename)
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, key[1], item["generation"])
                views = {v["view_id"]: v for v in item["views"]}
                usable = set(views) == {"front","rear","left","right"} and not any(
                    obj.get(flag) for obj in (item, meta) for flag in ("ideal_line_shown", "aid_probe_only"))
                base = np.full((len(stations), 2), np.nan)
                if usable:
                    images = {side: cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name), np.uint8),
                                                 cv2.IMREAD_COLOR) for side, v in views.items()}
                    assert all(im is not None for im in images.values())
                    raw = near.detect(images) + far.detect({"front": images["front"]})[1:]
                    values = np.array([[s[f] if s[f] is not None else np.nan for f in FIELDS] for s in raw])
                    base = np.where(np.array([s["valid"] for s in raw])[:, None], values, np.nan)
                if track == SILVER:
                    assert np.array_equal(base, cache[key], equal_nan=True)
                observable, original = fit_boundaries(base, .002)["hinge"]
                arrays = {m: base for m in ("original", "normal", "learned", "ranked")}
                if usable and (not observable or row["truth"] == "approach"):
                    lab = {side: cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for side, im in images.items()}
                    contrast = np.r_[edge_scores(near, lab), edge_scores(far, lab)[1:]]
                    score = probabilities(model, grid, lab)
                    arrays.update(normal=choose(raw, contrast, near.lateral, "normal_fill"),
                                  learned=learned.sections(score),
                                  ranked=ranked_patches(raw, contrast, near.lateral, score))
                    assert np.array_equal(arrays["ranked"][np.isfinite(base).all(axis=1)],
                                          base[np.isfinite(base).all(axis=1)])
                choices = {name: original if observable else fit_boundaries(edges, .002)["hinge"][1]
                           for name, edges in arrays.items()}
                choices["ranked_boundary"] = original if observable else confirm(profiles[0].get(key), choices["ranked"])
                choices["ranked_t4t5"] = original if observable else confirm(profiles[1].get(key), choices["ranked"])
                for mode, proposal in choices.items():
                    by_mode[mode].append(publish(row, proposal, observable or proposal is not None))
                if row["truth"] == "approach" or not observable and any(choices.values()):
                    edges_eval = {}
                    if row["truth"] == "approach":
                        body = max(meta["views"], key=lambda v: v["render_call_finished_clock"]).get("source_body_after_render")
                        target = reference(track_edges, body)
                        for name, arr in arrays.items():
                            valid = np.isfinite(arr) & np.isfinite(target)
                            error = np.abs(arr-target)
                            edges_eval[name] = dict(paired=int(np.isfinite(arr).all(axis=1).sum()),
                                                   right_error_m=error[:,0][valid[:,0]].tolist(),
                                                   left_error_m=error[:,1][valid[:,1]].tolist())
                    details.append(dict(run=run, frame=key[1], truth=row["truth"], label=row["label"],
                                        truth_endpoints_m=row["truth_endpoints_m"],
                                        baseline_observable=observable, choices=choices, edge_evaluation=edges_eval))
        for mode, rows in by_mode.items():
            reports[mode][run] = summarize(rows)
    assert sum(r["frames"] for r in reports["original"].values()) == len(expected)
    path = OUT / f"{track}-frames.jsonl"
    jsonl(path, details)
    print(track, "replay complete", flush=True)
    return dict(methods=reports, rows_sha256=digest(path), source_sha256={str(p): digest(p) for p in sorted(hashes)})


def confirmation_models(training, hashes):
    names = ("boundaries", "boundaries_temporal_t4t5")
    train = [inputs(t, hashes) for t in training]
    models = [profile.fit([g[m] for g in train], 1) for m in names]
    profiles = {}
    for track in TEST_TRACKS:
        data = inputs(track, hashes)
        profiles[track] = [
            dict(zip((frame_key(r) for r in data[m]["selected"]), profile.predict(data[m], model), strict=True))
            for m, model in zip(names, models, strict=True)
        ]
    return profiles


def main():
    cv2.setNumThreads(1)
    training = tuple(profile.CALIBRATION_TRACKS)
    assert set(training).isdisjoint(TEST_TRACKS)
    config = Path(load_frozen(PROTOCOL.with_name("candidate.json"))["camera_config"])
    hashes = {Path(__file__), config, Path(learned.__file__), Path(inputs.__code__.co_filename),
              Path(profile.__file__), Path(signals.__code__.co_filename)}
    model = learned.train([training_set(t, hashes) for t in training])
    checkpoint = OUT/"patch-model.joblib"
    joblib.dump(model, checkpoint)
    print("Boundary model fit:", training, flush=True)
    profiles = confirmation_models(training, hashes)
    write(OUT/"frozen.json", dict(training=training, excluded=TEST_TRACKS, seed=learned.SEED,
          checkpoint_sha256=digest(checkpoint), patch_threshold=learned.PATCH_THRESHOLD,
          confirmation_threshold=CONFIRM_THRESHOLD, source_sha256={str(p):digest(p) for p in sorted(hashes)}))
    mount = read(config)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {t: pool.submit(replay, t, mount, model, profiles[t]) for t in TEST_TRACKS}
        results = {t: f.result() for t, f in futures.items()}
    write(OUT/"result.json", dict(status="evaluated", source_revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",
          seed=learned.SEED, termination="archive_end", frozen_sha256=digest(OUT/"frozen.json"), tracks=results))


if __name__ == "__main__":
    main()
