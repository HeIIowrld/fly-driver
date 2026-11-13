"""Learn a causal road-edge estimate from independent current and past observations."""
from pathlib import Path
import sys
from collections import Counter, deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import joblib

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
GATE = OUT.parent / "vision-corner-corrected-geometry-gate-20261112-01"
sys.path.insert(0, str(GATE))
from train_gate import ARCHIVE, DEVELOP, SILVER, VAL, MONZA, old, load_observer, reference, frames
from gate_inputs import EXTRA, read_development
from replay_corner_boundary_states import observe, sensor_rows, alt, valid_pairs
from aligned_sections import MAX_AGE_S, MAX_HISTORY, MAX_SAMPLE_GAP_M

SEED = 20261112
SIGNS = np.array([-1., 1.])


def project(stations, points):
    x, y = points.T
    result = np.full(len(stations), np.nan)
    if len(x) < 2 or np.any(np.diff(x) <= 0.):
        return result
    right = np.searchsorted(x, stations).clip(1, len(x)-1)
    left = right-1
    valid = (stations >= x[left]) & (stations <= x[right]) & (x[right]-x[left] <= MAX_SAMPLE_GAP_M)
    result[valid] = np.interp(stations[valid], x, y)
    return result


def history(sensors, settings):
    stations, raw, motion = (sensors[k] for k in ("stations", "sections", "motion"))
    rows = sensor_rows(sensors, settings)
    out = np.full((*raw.shape, MAX_HISTORY), np.nan)
    previous = deque(maxlen=MAX_HISTORY)
    moving = np.isfinite(motion).all(axis=1) & (motion[:, 1] >= old.MIN_SPEED_MPS)
    for i, current in enumerate(raw):
        step = alt.transform(motion[i], motion[i-1]) if i and moving[i] and moving[i-1] and alt.adjacent(rows[i-1], rows[i]) else None
        if step is None:
            previous.clear()
        else:
            previous = deque(((stamp, tuple(alt.advance(curve[:, None], *step)[:, 0] for curve in pair))
                              for stamp, pair in previous if 0 < motion[i, 0]-stamp <= MAX_AGE_S),
                             maxlen=MAX_HISTORY)
        for lag, (_, pair) in enumerate(reversed(previous)):
            for side, curve in enumerate(pair):
                out[i, :, side, lag] = project(stations, curve)
        if moving[i]:
            observed = np.isfinite(current.T)
            curves = tuple(np.column_stack((stations[ok], values[ok])) for ok, values in zip(observed, current.T, strict=True))
            previous.append((motion[i, 0], curves))
    return out


def shift(a, n):
    out = np.full_like(a, np.nan)
    if n < 0:
        out[:, -n:] = a[:, :n]
    else:
        out[:, :-n] = a[:, n:]
    return out


def features(sensors, settings):
    raw = sensors["sections"] * SIGNS
    past = history(sensors, settings) * SIGNS[None, None, :, None]
    before, after = shift(raw, -1), shift(raw, 1)
    bracket = (before+after)/2.
    supported = np.isfinite(raw) | np.isfinite(bracket) | np.isfinite(past).any(-1)
    station = np.broadcast_to(sensors["stations"][None, :, None], raw.shape)
    speed = np.broadcast_to(sensors["motion"][:, 1, None, None], raw.shape)
    columns = [raw, bracket, before, after, shift(raw, -2), shift(raw, 2), raw[:, :, ::-1],
               before[:, :, ::-1], after[:, :, ::-1], station, speed, *np.moveaxis(past, -1, 0)]
    return np.stack(columns, axis=-1), supported


class SensorArrays(dict):
    @property
    def files(self):
        return self.keys()


def load_labels(path, sensors, sources, camera_hashes=None):
    receipt = path.with_suffix(".json")
    proof = old.read(receipt)
    assert old.digest(path) == proof["labels_sha256"]
    if camera_hashes is not None:
        assert camera_hashes == proof["camera_bundle_sha256_by_run"]
    with np.load(path, allow_pickle=False) as data:
        for key in ("run", "frame", "stations"):
            np.testing.assert_array_equal(data[key], sensors[key])
        ok = data["adjacent_invalid"] & np.isfinite(data["image_px"]).all(axis=-1).any(axis=1)
        edge = data["lateral_m"].copy()
    assert np.isfinite(edge[ok]).all()
    sources.update((path, receipt))
    return edge, ok


def regress(folds):
    x, y, w = [], [], []
    for f, support, edge, valid in folds:
        use = support & valid
        x.append(f[use])
        y.append((edge * SIGNS)[use])
        w.append(np.full(use.sum(), 1/use.sum()))
    weights = np.concatenate(w)
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=100, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1., random_state=SEED)
    return model.fit(np.concatenate(x), np.concatenate(y), sample_weight=weights/weights.mean())


def estimate(model, f, support):
    result = np.full(support.shape, np.nan)
    result[support] = model.predict(f[support])
    return result


def combine(raw, signed, fill, replace):
    result = np.where(np.isfinite(raw), raw, np.where(fill, signed * SIGNS, np.nan))
    result[replace] = (signed * SIGNS)[replace]
    invalid = np.isfinite(result).all(-1) & ~valid_pairs(result)
    result[invalid] = raw[invalid]
    return result


def boundary_scores(raw, result, edge, valid):
    known = valid & np.isfinite(raw)
    new = valid & ~np.isfinite(raw) & np.isfinite(result)
    changed = known & (raw != result)
    def error(value, use):
        return float(np.mean(abs(value[use]-edge[use]))) if use.any() else None
    return dict(labels=int(valid.sum()), original=int(known.sum()), new=int(new.sum()),
                new_mae=error(result, new), changed=int(changed.sum()), old_mae=error(raw, known),
                common_mae=error(result, known), old_changed_mae=error(raw, changed),
                new_changed_mae=error(result, changed))


def quality(rows):
    x, y, w = [], [], []
    for f, use, edge, valid, predicted in rows:
        use = use & valid
        error = abs(predicted - edge*SIGNS)
        raw_error = abs(f[..., 0] - edge*SIGNS)
        observed = np.isfinite(f[..., 0])
        target = np.where(observed, error + .15 < raw_error, error <= 1.25)
        x.append(np.column_stack((f[use], predicted[use], (predicted-f[..., 0])[use])))
        y.append(target[use])
        w.append(np.full(use.sum(), 1/use.sum()))
    weights = np.concatenate(w)
    model = HistGradientBoostingClassifier(max_iter=65, max_leaf_nodes=12, min_samples_leaf=50,
                                            l2_regularization=1., random_state=SEED)
    return model.fit(np.concatenate(x), np.concatenate(y), sample_weight=weights/weights.mean())


def confidence(model, f, support, predicted):
    result = np.zeros(support.shape)
    columns = np.column_stack((f[support], predicted[support], (predicted-f[..., 0])[support]))
    result[support] = model.predict_proba(columns)[:, 1]
    return result


def freeze(dev):
    summary = {}
    for held, fold in dev.items():
        rest = [k for k in dev if k != held]
        assert len(rest) >= 3
        model = regress([dev[t] for t in rest])
        head = quality([(*dev[t], estimate(regress([dev[j] for j in rest if j != t]), *dev[t][:2]))
                        for t in rest])
        f, use, edge, valid = fold
        pred = estimate(model, f, use)
        confidence_score = confidence(head, f, use, pred)
        raw = f[..., 0] * SIGNS
        summary[held] = {}
        for cut in (.5, .6, .7, .8, .9, 1.01):
            accept = confidence_score >= cut
            updated = combine(raw, pred, accept, accept & np.isfinite(raw))
            summary[held][cut] = boundary_scores(raw, updated, edge, valid)
        print("Boundary held-out", held, summary[held], flush=True)

    def key(cut):
        values = [report[cut] for report in summary.values()]
        penalty = sum(r["new"]*max(0., (r["new_mae"] or 0.)-1.25) for r in values)
        common_harm = sum(r["original"]*max(0., r["common_mae"]-r["old_mae"]-.01) for r in values)
        coverage = sum(r["new"] for r in values)
        return penalty + common_harm, -coverage, cut

    cut = min(next(iter(summary.values())), key=key)
    head = quality([(*dev[t], estimate(regress([dev[j] for j in dev if j != t]), *dev[t][:2])) for t in dev])
    return regress(list(dev.values())), head, cut, summary


def run():
    models, settings, sources = load_observer()
    sources.update((Path(__file__), Path(read_development.__code__.co_filename)))
    dev = {}
    for track in DEVELOP:
        data, _, receipt = read_development(track, sources, settings)
        labels = load_labels(EXTRA/f"{track}-train-labels.npz", data, sources, receipt["camera_bundle_sha256_by_run"])
        dev[track] = (*features(SensorArrays(data), settings), *labels)
    reg, head, cut, dev_scores = freeze(dev)
    ckpt = OUT/"boundary.joblib"
    joblib.dump(dict(coordinate=reg, confidence=head, cut=cut), ckpt)
    frozen = OUT/"training.json"
    assert all(t not in str(p) for p in sources for t in (SILVER, VAL, MONZA))
    old.write(frozen, dict(development=DEVELOP, excluded=[SILVER, VAL, MONZA], threshold=cut,
              folds=dev_scores, seed=SEED, checkpoint_sha256=old.digest(ckpt), source_revision=old.revision(),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)}, termination="frozen_before_video_replay"))
    results = {}
    for track, name in ((SILVER, "silverstone"), (VAL, "vallelunga"), (MONZA, "monza")):
        folder = ARCHIVE/name
        receipt = folder/"result.json"
        sensor = folder/"sensors.npz"
        assert old.digest(sensor) == old.read(receipt)["inputs_sha256"]
        with np.load(sensor, allow_pickle=False) as d:
            data = SensorArrays({k:d[k] for k in d.files})
        f, support = features(data, settings)
        predicted = estimate(reg, f, support)
        accept = confidence(head, f, support, predicted) >= cut
        raw = data["sections"]
        updated = combine(raw, predicted, accept, accept & np.isfinite(raw))
        assert not np.any(np.isfinite(raw) & ~np.isfinite(updated))
        assert not np.any(~support & np.isfinite(updated))
        forecasts, aligned, _ = observe(SensorArrays(dict(data, sections=updated)), models, settings)
        archive = folder/"predictions/forecasts.jsonl"
        original = frames(archive, old.read(archive.with_name("result.json"))["forecasts_sha256"])
        assert [(r["run"], r["frame"]) for r in original] == [(r["run"], r["frame"]) for r in forecasts]
        truth, truth_path = reference(track)
        by_key = {(r["run"], r["frame"]):r for r in truth}
        proof_paths = {sensor, receipt, archive, truth_path}
        trace = []
        counts = {}
        for a,b in zip(original, forecasts, strict=True):
            ref = by_key.get((a["run"], a["frame"]))
            if ref is None:
                continue
            row = dict(run=a["run"], frame=a["frame"], truth=ref["truth"], label=ref["label"],
                       before_candidate=a["candidate"], after_candidate=b["candidate"], forecasts={})
            for mode, item in (("before", a), ("after", b)):
                for method, forecast in item["forecasts"].items():
                    direction = {"left":1, "right":2}.get(forecast["state"])
                    category = old.category(ref, (direction, forecast["distance_m"]) if direction else None)
                    tally = counts.setdefault(f"{mode}_{method}", Counter())
                    tally[category] += 1
                    if category == "joint_hit":
                        tally[f"joint_{ref['label']}"] += 1
                    row["forecasts"][f"{mode}_{method}"] = dict(**forecast, category=category)
            trace.append(row)
        assert len(trace) == len(truth)
        result = dict(frames=len(trace), counts=counts, added=int((~np.isfinite(raw)&np.isfinite(updated)).sum()),
                      adjusted=int((np.isfinite(raw)&(raw!=updated)).sum()))
        if track == SILVER:
            mesh = OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            edge, visible = load_labels(mesh, data, proof_paths)
            right = np.array([by_key.get((r["run"], r["frame"]), {}).get("label") == "2" and
                              by_key.get((r["run"], r["frame"]), {}).get("truth") == "approach" for r in forecasts])
            assert right.sum() == 13
            result["right_approach_edges"] = {side:boundary_scores(raw[right,:,j], updated[right,:,j], edge[right,:,j],
                                                                  visible[right,:,j])
                                               for j,side in enumerate(("right","left"))}
        witness = OUT/f"{track}-frames.jsonl"
        old.jsonl(witness, trace)
        np.savez_compressed(OUT/f"{track}-boundaries.npz", sections=updated, aligned=aligned, stations=data["stations"],
                            run=data["run"], frame=data["frame"])
        result.update(trace_sha256=old.digest(witness), boundaries_sha256=old.digest(OUT/f"{track}-boundaries.npz"),
                      source_sha256={str(p):old.digest(p) for p in sorted(proof_paths)})
        results[track] = result
        print(track, result["counts"], result.get("right_approach_edges"), flush=True)
    old.write(OUT/"result.json", dict(tracks=results, training_sha256=old.digest(frozen), termination="video_replay_scored"))


if __name__ == "__main__":
    run()
