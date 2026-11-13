"""Apply side-camera additions to the same frozen corner observer."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import cv2

from side_model import (OUT, ROOT, old, image, stream, sampling, observations, image_features,
                        SIDES, STATIONS, DISTANCES, TEST, DEV)
from audit_right import ARCHIVE, CAPTURE, ROUTING, MESH
from image_normal import across_normal
from replay_corner_boundary_states import load_observer, observe, joblib
from verify_directional import frames

VAL = "ks_vallelunga"
RAW_ARCHIVE = ARCHIVE.parent
METHODS = ("temporal", "t4t5_confirm")
VARIANTS = ("normal_anchor", "learned", "learned_anchor")
ANCHOR_STATION_M = 4.
ANCHOR_LATERAL_M = 2.


def select(p, grid, eligible, floor):
    index = np.flatnonzero(eligible)
    if not len(index):
        return np.nan
    best = index[np.argmax(p[index])]
    return grid[best] if p[best] >= floor else np.nan


def guard(current, points, station, side, grid, anchored):
    sign = (-1., 1.)[side]
    other = current[station, 1-side]
    allowed = np.ones(len(grid), bool)
    if np.isfinite(other):
        width = sign*(grid-other)
        allowed &= (width > 3.) & (width < 20.)
    if anchored:
        anchor = np.isfinite(current[:, side]) & (abs(points-points[station]) <= ANCHOR_STATION_M)
        if not anchor.any():
            return np.zeros(len(grid), bool)
        allowed &= np.any(abs(grid[:, None]-current[anchor, side]) <= ANCHOR_LATERAL_M, axis=1)
    return allowed


def predict_side(track, mapping, bundles, sensor, model, cut):
    cv2.setNumThreads(1)
    sample = sampling(old.read(old.MOUNT))
    keys = list(zip(sensor["run"], sensor["frame"], strict=True))
    index = {key: i for i, key in enumerate(keys)}
    observed = sensor["sections"]
    filled = {name: observed.copy() for name in VARIANTS}
    entries, xs = [], []
    for run, folder in old.read(mapping).items():
        for arc, item, _, views in stream(folder, bundles[run], track):
            i = index[run, item["capture_id"]]
            if not views or not np.isfinite(sensor["motion"][i]).all() or sensor["motion"][i, 1]<old.MIN_SPEED_MPS:
                continue
            for side, view in enumerate(SIDES):
                need = np.flatnonzero(~np.isfinite(observed[i, :len(STATIONS), side]))
                if not len(need):
                    continue
                bitmap = image(arc, views[view])
                features, valid = image_features(bitmap, sample[view])
                u, v, mask = sample[view]
                score = across_normal(cv2.cvtColor(bitmap, cv2.COLOR_BGR2LAB), u/319., v/179., mask)
                sign = (-1., 1.)[side]
                grid = sign*DISTANCES
                for s in need:
                    free = valid[s] & guard(observed[i], sensor["stations"], s, side, grid, False)
                    anchored = free & guard(observed[i], sensor["stations"], s, side, grid, True)
                    finite = np.isfinite(score[s])
                    adjacent = np.r_[False, finite[:-1]] | np.r_[finite[1:], False]
                    filled["normal_anchor"][i, s, side] = select(score[s], grid, anchored & finite & adjacent, 0.)
                    entries.append((i, s, side, grid, free, anchored))
                    xs.append(features[s])
    if xs:
        probability = model.predict_proba(np.concatenate(xs))[:, 1].reshape(-1, len(DISTANCES))
        for (i, s, side, grid, free, anchored), p in zip(entries, probability, strict=True):
            filled["learned"][i, s, side] = select(p, grid, free, cut)
            filled["learned_anchor"][i, s, side] = select(p, grid, anchored, cut)
    for variant in filled.values():
        width = variant[:, :, 1]-variant[:, :, 0]
        bad = np.isfinite(width) & ((width<=3.) | (width>=20.))
        variant[bad] = observed[bad]
        np.testing.assert_array_equal(variant[np.isfinite(observed)], observed[np.isfinite(observed)])
    return filled


def metrics(rows, reference):
    stats = {m: Counter() for m in METHODS}
    cases = {}
    for actual, truth in zip(rows, reference, strict=True):
        assert (actual["run"], actual["frame"]) == (truth["run"], truth["frame"])
        for method in METHODS:
            data = actual["forecasts"][method]
            state = data["state"]
            offer = ({"left": 1, "right": 2}[state], data["distance_m"]) if state in ("left", "right") else None
            category = old.category(truth, offer)
            stats[method][category] += 1
            if state=="right" and category=="joint_hit":
                stats[method]["right_joint_hit"] += 1
                cases.setdefault(method,[]).append(actual["frame"])
    return {m: dict(v) for m,v in stats.items()}, cases


def mesh_score(variant, base, reference, keys):
    assert old.digest(MESH)==old.read(MESH.with_suffix(".json"))["labels_sha256"]
    with np.load(MESH, allow_pickle=False) as data:
        assert keys==list(zip(data["run"],data["frame"],strict=True))
        valid = data["adjacent_invalid"] & np.isfinite(data["image_px"]).all(axis=-1).any(axis=1)
        truth = data["lateral_m"]
    mask = np.array([r["truth"]=="approach" and r["label"]=="2" for r in reference])
    right = valid[:,:,0] & mask[:,None]
    fresh = right & np.isfinite(variant[:,:,0]) & ~np.isfinite(base[:,:,0])
    error = abs(variant[:,:,0]-truth[:,:,0])
    return dict(labelled=int(right.sum()), previously_observed=int((right & np.isfinite(base[:,:,0])).sum()),
                new=int(fresh.sum()), new_mae_m=float(error[fresh].mean()) if fresh.any() else None,
                within_1_25_m=int((fresh & (error<=1.25)).sum()),
                points=[dict(frame=int(keys[i][1]), station=float(old.STATIONS[s]), estimate=float(variant[i,s,0]),
                             truth=float(truth[i,s,0]), absolute_error_m=float(error[i,s]))
                        for i,s in zip(*np.where(fresh),strict=True)])


def main():
    training = OUT/"side-training.json"
    model_path = OUT/"side.joblib"
    frozen = old.read(training)
    assert old.digest(model_path)==frozen["checkpoint_sha256"]
    assert frozen["development"]==list(DEV) and frozen["excluded"]==[TEST]
    for path, sha in frozen["source_sha256"].items():
        assert TEST not in path and VAL not in path and old.digest(Path(path))==sha
    model = joblib.load(model_path)
    models, settings, sources = load_observer()
    sources.update((Path(__file__), training, model_path, Path(across_normal.__code__.co_filename)))
    proof = old.read(ROUTING/"straight-verification.json")
    output = {}
    for track, name, mapping in (
            (TEST,"silverstone", CAPTURE),
            (VAL,"vallelunga", OUT.parent/"vision-vallelunga-local-corner-capture-20261112-01"/"archive-map.json")):
        original = RAW_ARCHIVE/name
        sensor_path = original/"sensors.npz"
        receipt = old.read(original/"result.json")
        assert old.digest(sensor_path)==receipt["inputs_sha256"]
        forecast_proof = old.read(original/"predictions/result.json")
        prediction = frames(original/"predictions/forecasts.jsonl", forecast_proof["forecasts_sha256"])
        prior = ROUTING/f"{track}-state-distance.jsonl"
        reference = frames(prior, proof["tracks"][track]["rows_sha256"])
        with np.load(sensor_path, allow_pickle=False) as data:
            raw = {n:data[n] for n in data.files}
        offers = predict_side(track, mapping, receipt["archive_bundles"], raw, model, frozen["cut"])
        base = np.load(original/"predictions/boundaries.npz",allow_pickle=False)
        baseline, base_cases = metrics(prediction, reference)
        methods = {"baseline":dict(metrics=baseline,right_hits=base_cases)}
        for variant, values in offers.items():
            path = OUT/f"{track}-{variant}-sensors.npz"
            np.savez_compressed(path, **dict(raw, sections=values))
            with np.load(path,allow_pickle=False) as data:
                rows, aligned, changed = observe(data,models,settings)
            report, cases = metrics(rows,reference)
            evidence = OUT/f"{track}-{variant}-frames.jsonl"
            old.jsonl(evidence,rows)
            methods[variant] = dict(metrics=report,right_hits=cases,frames_sha256=old.digest(evidence),
                                    sensor_sha256=old.digest(path),observed_additions=(np.isfinite(values)&~np.isfinite(raw["sections"])).sum(axis=(0,1)).tolist())
            if track==TEST:
                methods[variant]["mesh"] = mesh_score(aligned,base["estimated"],reference,
                                                list(zip(raw["run"],raw["frame"],strict=True)))
        base.close()
        hashes = {*sources, sensor_path, mapping, original/"result.json", original/"predictions/result.json",
                  original/"predictions/boundaries.npz", prior}
        if track==TEST:
            hashes.update((MESH,MESH.with_suffix(".json")))
        output[track] = dict(methods=methods,source_sha256={str(p):old.digest(p) for p in sorted(hashes)})
        print(track,{m:(v.get("observed_additions"),v["right_hits"],v.get("mesh",{}).get("new"))
                     for m,v in methods.items()},flush=True)
    old.write(OUT/"side-result.json",dict(status="evaluated_on_original_timelines", training_sha256=old.digest(training),
              tracks=output, termination="same_source_video_and_mesh_scored"))


if __name__ == "__main__":
    main()
