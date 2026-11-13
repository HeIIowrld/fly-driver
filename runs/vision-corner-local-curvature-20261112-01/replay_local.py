"""Replay local curvature and paired T4/T5 confirmation on the same road edges."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from corner_fusion_boundaries import curvature, decide, read_motion, stations, transform
from replay_learned import (
    CONFIRM_THRESHOLD, FIELDS, OUT as PREVIOUS, PROTOCOL, SILVER, TEST_TRACKS, CameraArchive,
    PatchGrid, SurfaceContrast, baseline_data, choose, confirmation_models, digest, edge_scores,
    fit_boundaries, frame_key, jsonl, load_frozen, probabilities, publish, ranked_patches,
    read, signals, summarize, write,
)
from score_corner_captures import MIN_SPEED_MPS

OUT = Path(__file__).parent
CUT = .002
MODES = ("baseline", "ranked_hinge", "ranked_local", "normal_local",
         "ranked_local_quiet_prefix", "ranked_local_boundary", "ranked_local_t4t5",
         "ranked_local_boundary_hold", "ranked_local_t4t5_hold")


def alert(value):
    return value if value is not None and value[0] and 4. <= value[1] <= 20. else None


def quiet_prefix(samples, proposal):
    if proposal is None:
        return None
    _, onset = proposal
    before = [k for x, k in samples if onset-4. <= x <= onset-2.]
    return proposal if len(before) >= 2 and all(abs(k) < CUT for k in before) else None


def held(previous, step, current):
    if previous is None or step is None or current is not None:
        return None
    return alert((previous[0], previous[1]-float(np.linalg.norm(step[1]))))


def support(values, proposal):
    if proposal is None:
        return None
    return signals(values, *proposal)["local"]


def replay(track, mount, model, profiles):
    hashes = {Path(__file__), Path(ranked_patches.__code__.co_filename), Path(curvature.__code__.co_filename),
              Path(baseline_data.__code__.co_filename), Path(summarize.__code__.co_filename),
              Path(SurfaceContrast.detect.__code__.co_filename), Path(edge_scores.__code__.co_filename)}
    group, cache, mapping, bundles = baseline_data(track, hashes)
    near, far, grid = SurfaceContrast(mount, near=True), SurfaceContrast(mount), PatchGrid(mount)
    truth = {frame_key(row): row for row in group["rows"]}
    reports = {mode: {} for mode in MODES}
    rows, arrays, keys = [], [], []
    for run, folder in read(mapping).items():
        by_mode = {mode: [] for mode in MODES}
        previous = None
        previous_confirmed = (None, None)
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[run]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                key = run, item["capture_id"]
                if key not in truth:
                    previous = None
                    previous_confirmed = (None, None)
                    continue
                row = truth[key]
                name = Path(item["file"]).name
                assert archive.by_name[name]["sha256"] == item["file_sha256"]
                meta = archive.json(name)
                gen = item["generation"]
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, key[1], gen)
                views = {v["view_id"]: v for v in item["views"]}
                usable = set(views) == {"front","rear","left","right"} and not any(
                    obj.get(flag) for obj in (item, meta) for flag in ("ideal_line_shown", "aid_probe_only"))
                base = np.full((len(stations), 2), np.nan)
                if usable:
                    images = {side: cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name), np.uint8),
                                                 cv2.IMREAD_COLOR) for side, v in views.items()}
                    assert all(im is not None for im in images.values())
                    raw = near.detect(images) + far.detect({"front": images["front"]})[1:]
                    pos = np.array([[r[f] if r[f] is not None else np.nan for f in FIELDS] for r in raw])
                    base = np.where(np.array([r["valid"] for r in raw])[:, None], pos, np.nan)
                if track == SILVER:
                    assert np.array_equal(base, cache[key], equal_nan=True)
                observable, original = fit_boundaries(base, CUT)["hinge"]
                normal, ranked = base, base
                if usable and not observable:
                    lab = {side: cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for side, im in images.items()}
                    contrast = np.r_[edge_scores(near, lab), edge_scores(far, lab)[1:]]
                    normal = choose(raw, contrast, near.lateral, "normal_fill")
                    ranked = ranked_patches(raw, contrast, near.lateral, probabilities(model, grid, lab))
                arrays.append(np.stack((base, normal, ranked)))
                keys.append(key)

                motion = read_motion(meta) if usable else None
                moving = motion is not None and motion[1] >= MIN_SPEED_MPS
                contiguous = previous is not None and moving and previous[:2] == (key[1]-1, gen)
                step = transform(motion, previous[2]) if contiguous else None
                samples, local, local_normal, hinge = [], None, None, None
                confirmations, continuation, scores, held_scores = [], [], [], []
                if not observable and moving:
                    samples = curvature(ranked)
                    local = alert(decide(samples, CUT))
                    local_normal = alert(decide(curvature(normal), CUT))
                    hinge = fit_boundaries(ranked, CUT)["hinge"][1]
                for values, old in zip(profiles, previous_confirmed, strict=True):
                    score = support(values.get(key), local)
                    direct = local if score is not None and score >= CONFIRM_THRESHOLD else None
                    candidate = held(old, step, local) if not observable else None
                    carry_score = support(values.get(key), candidate)
                    carried = candidate if carry_score is not None and carry_score >= CONFIRM_THRESHOLD else None
                    scores.append(score)
                    held_scores.append(carry_score)
                    confirmations.append(direct)
                    continuation.append(direct or carried)
                previous_confirmed = tuple(confirmations)
                previous = (key[1], gen, motion) if moving else None
                offers = (original, original or hinge, original or local, original or local_normal,
                          original or quiet_prefix(samples, local),
                          *(original or x for x in confirmations), *(original or x for x in continuation))
                for mode, proposal in zip(MODES, offers, strict=True):
                    by_mode[mode].append(publish(row, proposal, observable or proposal is not None))
                rows.append(dict(run=run, frame=key[1], truth=row["truth"], label=row["label"],
                                 physical_corner=row["physical_corner"], truth_endpoints_m=row["truth_endpoints_m"],
                                 baseline_observable=observable, proposals=dict(zip(MODES, offers, strict=True)),
                                 local_scores=scores, held_scores=held_scores))
        for mode, records in by_mode.items():
            reports[mode][run] = summarize(records)
    assert len(rows) == len(truth) and len(set(keys)) == len(truth)
    path = OUT / f"{track}-frames.jsonl"
    jsonl(path, rows)
    section_path = OUT / f"{track}-sections.npz"
    np.savez_compressed(section_path, run=np.array([k[0] for k in keys]), frame=np.array([k[1] for k in keys]),
                        modes=np.array(["baseline", "normal", "ranked"]), stations=stations, sections=np.stack(arrays))
    print(track, "all frames replayed", flush=True)
    return dict(methods=reports, rows_sha256=digest(path), sections_sha256=digest(section_path),
                source_sha256={str(p): digest(p) for p in sorted(hashes)})


def checks():
    assert quiet_prefix([(6., .001), (7., 0.)], (1, 10.)) == (1, 10.)
    assert quiet_prefix([(6., .003), (7., 0.)], (1, 10.)) is None
    assert held((2, 8.), (np.eye(2), np.array([3., 0.])), None) == (2, 5.)
    assert held((2, 6.), (np.eye(2), np.array([3., 0.])), None) is None


def main():
    cv2.setNumThreads(1)
    checks()
    freeze, checkpoint = PREVIOUS/"frozen.json", PREVIOUS/"patch-model.joblib"
    source = read(freeze)
    assert digest(checkpoint) == source["checkpoint_sha256"]
    assert set(source["training"]).isdisjoint(TEST_TRACKS)
    for path, sha in source["source_sha256"].items():
        assert digest(Path(path)) == sha, path
    model = joblib.load(checkpoint)
    config = Path(load_frozen(PROTOCOL.with_name("candidate.json"))["camera_config"])
    hashes = {Path(__file__), freeze, checkpoint, config, Path(signals.__code__.co_filename)}
    profiles = confirmation_models(tuple(source["training"]), hashes)
    mount = read(config)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {t: pool.submit(replay, t, mount, model, profiles[t]) for t in TEST_TRACKS}
        results = {t: f.result() for t, f in futures.items()}
    write(OUT/"result.json", dict(status="evaluated", source_revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",
          seed=source["seed"], termination="archive_end", tracks=results, curvature_cut=CUT,
          confirmation_cut=CONFIRM_THRESHOLD, source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
