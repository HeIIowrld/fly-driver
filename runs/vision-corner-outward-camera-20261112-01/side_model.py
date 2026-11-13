"""Learn a symmetric, physically supervised boundary readout from the side cameras."""
from pathlib import Path
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]/"scripts"))
from replay_corner_camera_archive import ROOT, old, image, stream
from replay_corner_boundary_states import joblib
from collect import baseline_data
from prepare import labels
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from sklearn.ensemble import HistGradientBoostingClassifier

DEV = ("magione", "spa")
TEST = "ks_silverstone1967"
SEED = 20261112
STATIONS = np.arange(2., 13., 2.)
DISTANCES = np.arange(1.6, 14.01, .2)
SIDES = ("right", "left")
SHIFTS = (-12., -6., 0., 6., 12.)
TOLERANCE_M = 1.25


def sampling(mount):
    detector = PixelRoadBoundary(mount)
    if not np.allclose(detector.lateral, -detector.lateral[::-1]):
        raise ValueError("Expected symmetric lateral samples")
    lateral = detector.lateral
    lookup = detector._sampling(mount, STATIONS)
    samples = {}
    for name, sign in zip(SIDES, (-1., 1.), strict=True):
        indices = np.array([np.abs(lateral-sign*d).argmin() for d in DISTANCES])
        u, v, good = (x[:, indices] for x in lookup[name])
        u, v = u*319., v*179.
        eligible = good & (u >= 2.) & (u <= 317.) & (v >= 18.) & (v <= 161.)
        samples[name] = (u.astype("f"), v.astype("f"), eligible)
    return samples


def image_features(bitmap, lookup):
    if bitmap.shape != (180, 320, 3):
        raise ValueError("Side camera must provide its native 320 by 180 image")
    u, v, valid = lookup
    lab = cv2.cvtColor(bitmap, cv2.COLOR_BGR2LAB).astype("f")
    smooth = cv2.GaussianBlur(lab, (5, 5), 1.)
    strip = {}
    for offset in sorted({shift + step for shift in SHIFTS for step in (-3., 3.)}):
        strip[offset] = cv2.remap(smooth, u, v+offset, cv2.INTER_LINEAR)
    evidence = []
    for shift in SHIFTS:
        inside, outside = strip[shift+3.], strip[shift-3.]
        diff = outside-inside
        evidence.append(np.stack((diff[..., 0], np.linalg.norm(diff[..., 1:], axis=-1),
                                  np.linalg.norm(outside[..., 1:]-128., axis=-1)-
                                  np.linalg.norm(inside[..., 1:]-128., axis=-1)), axis=-1))
    # Distance and image row describe the projection of the contrast samples.
    grid = np.broadcast_to(DISTANCES, valid.shape)
    matrix = np.concatenate([*evidence, grid[..., None], v[..., None]], axis=-1).astype("f")
    return matrix, valid


def observations(track, inventory, bundles, keys, sample):
    index = {key: i for i, key in enumerate(keys)}
    seen = set()
    for run, folder in old.read(inventory).items():
        for archive, item, _, views in stream(folder, bundles[run], track):
            key = run, item["capture_id"]
            if key not in index:
                continue
            i = index[key]
            if i in seen:
                raise ValueError("Repeated camera frame")
            seen.add(i)
            if views:
                for side, view in enumerate(SIDES):
                    yield i, side, *image_features(image(archive, views[view]), sample[view])
    assert len(seen) == len(keys)


def development(track, sample, hashes):
    lab, label_path = labels(track)
    group, _, mapping, bundles = baseline_data(track, hashes)
    keys = [(r["run"], r["frame"]) for r in group["rows"]]
    assert list(zip(lab["run"], lab["frame"], strict=True)) == keys
    assert np.array_equal(lab["stations"][:len(STATIONS)], STATIONS)
    witness = lab["adjacent_invalid"][:, :len(STATIONS)]
    cam = np.stack((lab["image_px"][:, 2, :len(STATIONS), 0], lab["image_px"][:, 1, :len(STATIONS), 1]), axis=-2)
    usable = witness & np.isfinite(cam).all(axis=-1)
    metres = abs(lab["lateral_m"][:, :len(STATIONS)])
    result = []
    for i, side, x, valid in observations(track, mapping, bundles, keys, sample):
        for station in np.flatnonzero(usable[i, :, side]):
            distance = metres[i, station, side]
            support = valid[station]
            if not np.any(support & (abs(DISTANCES-distance) <= .4)):
                continue
            result.append((x[station, support], DISTANCES[support], distance))
    hashes.update((label_path, label_path.with_suffix(".json"), mapping))
    return result


def fit(points):
    matrices, targets, weights = [], [], []
    for x, grid, distance in points:
        delta = abs(grid-distance)
        hit = np.flatnonzero(delta <= .4)
        miss = np.flatnonzero(delta >= .9)
        if len(miss) == 0:
            continue
        samples = np.unique(np.r_[miss[np.linspace(0, len(miss)-1, min(8, len(miss))).astype(int)],
                                  miss[np.argsort(delta[miss])[:2]]])
        matrices.append(x[np.r_[hit, samples]])
        targets.extend([1]*len(hit) + [0]*len(samples))
        weights.extend([.5/len(hit)]*len(hit) + [.5/len(samples)]*len(samples))
    x = np.concatenate(matrices)
    y, w = np.asarray(targets), np.asarray(weights)
    w *= len(w)/w.sum()
    return HistGradientBoostingClassifier(max_iter=65, max_leaf_nodes=15, min_samples_leaf=35,
                    learning_rate=.1, l2_regularization=1., early_stopping=False, random_state=SEED).fit(x, y, sample_weight=w)


def best(points, model):
    rows = [x for x, _, _ in points]
    sizes = np.array([len(x) for x in rows])
    scores = np.split(model.predict_proba(np.concatenate(rows))[:, 1], np.cumsum(sizes)[:-1])
    result = []
    for (_, grid, distance), probability in zip(points, scores, strict=True):
        j = np.argmax(probability)
        result.append((probability[j], abs(grid[j]-distance)))
    return np.array(result)


def calibrate(folds):
    score, error = np.concatenate(list(folds.values())).T
    thresholds = np.r_[np.unique(score), np.nextafter(1., np.inf)]
    chosen = max((int((score >= p).sum()), float(p)) for p in thresholds
                 if not np.any(score >= p) or error[score >= p].mean() <= TOLERANCE_M)
    cut = chosen[1]
    return cut, {track: dict(points=len(a), accepted=int((a[:, 0]>=cut).sum()),
                     mae_m=float(a[a[:, 0]>=cut, 1].mean()) if (a[:, 0]>=cut).any() else None)
                 for track, a in folds.items()}


def main():
    cv2.setNumThreads(1)
    mount = old.read(old.MOUNT)
    sample = sampling(mount)
    hashes = {Path(__file__), old.MOUNT, Path(stream.__code__.co_filename), Path(labels.__code__.co_filename),
              ROOT/"src/flybrain_drive/pixel_road_boundary.py", Path(baseline_data.__code__.co_filename)}
    points = {}
    for track in DEV:
        points[track] = development(track, sample, hashes)
        print(track, len(points[track]), flush=True)
    folds = {track: best(points[track], fit([x for name, p in points.items() if name!=track for x in p]))
             for track in DEV}
    cut, calibration = calibrate(folds)
    frozen = OUT/"side.joblib"
    joblib.dump(fit([x for p in points.values() for x in p]), frozen)
    assert not any(TEST in str(path) for path in hashes)
    old.write(OUT/"side-training.json", dict(status="frozen_before_silverstone", development=DEV, excluded=[TEST],
              seed=SEED, source_revision=old.revision(), cut=cut, calibration=calibration,
              error_tolerance_m=TOLERANCE_M, checkpoint_sha256=old.digest(frozen),
              source_sha256={str(p):old.digest(p) for p in sorted(hashes)}, termination="side_crossfit_then_frozen"))
    print("Frozen", cut, calibration, flush=True)


if __name__ == "__main__":
    main()
