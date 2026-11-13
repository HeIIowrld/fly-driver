"""Freeze a left-turn confirmation policy on development recordings and score complete timelines."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
sys.path.insert(0, str(RUNS / "vision-corner-onset-through-line-20261113-01"))
sys.path.insert(0, str(RUNS / "vision-corner-selected-confirmation-20261112-01"))
from dense_onset import CLI, DEVELOP, EVAL, MONZA, frames, label, load_observer, old, read_development, reference
from gate_inputs import SensorArrays, build, scores
from replay_corner_boundary_states import alt, observe, phase_scores
from replay_corner_camera_archive import image, stream
from corner_archive_recovery import recover_sections

METHOD = "t4t5_confirm"
RISK = {"outside": 8, "inside": 5, "wrong_direction": 2, "distance_error": 2}
PER_RECORDING = {"outside": 4, "inside": 3, "wrong_direction": 1, "distance_error": 1}
SEED = 20261113
SELECTED = RUNS / "vision-corner-selected-confirmation-20261112-01"
ROUTES = ("raw", "recovered", "either")
MODES = ("structured", "agreement", "logistic", "boosting")
NO_ALERT = {"straight", "abstain"}


def digests(paths):
    return {str(path): old.digest(path) for path in sorted(set(paths))}


def keys(rows):
    return [(r["run"], r["frame"]) for r in rows]


def load_sensor(path, digest):
    if old.digest(path) != digest:
        raise ValueError(f"Sensor input hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return SensorArrays({k: data[k] for k in data.files})


def describe(raw, recovered, observer, settings, structured):
    raw_built, recovered_built = [build(sensor, settings) for sensor in (raw, recovered)]
    raw_rows, baseline = [observe(sensor, observer, settings)[0] for sensor in (raw, recovered)]
    if keys(raw_rows) != keys(baseline) or keys(baseline) != keys(raw_built["rows"]) or keys(baseline) != keys(recovered_built["rows"]):
        raise ValueError("Raw and recovered timelines differ")
    results = []
    for origin, (item, rows, other) in enumerate(((raw_built, raw_rows, baseline),
                                                   (recovered_built, baseline, raw_rows))):
        group = item["group"]
        phase = phase_scores(group, observer["phase"])
        lookup = {(i, offer): j for j, (i, offer) in enumerate(zip(group["index"], group["offers"], strict=True))}
        eligible = []
        for i, row in enumerate(rows):
            offer = row["candidate"]
            if (offer is not None and offer[0] == 1 and baseline[i][METHOD_KEY][METHOD]["state"] in NO_ALERT
                    and baseline[i]["motion_vision_available"] and phase[i, 2] <= settings["phase_cut"]):
                eligible.append((i, lookup[i, tuple(offer)]))
        ix = np.asarray([j for _, j in eligible], dtype=int)
        probability = scores(group, ix, structured) if len(ix) else {}
        for k, (i, j) in enumerate(eligible):
            distance = group["offers"][j][1]
            counterpart = other[i]["candidate"]
            agreement = np.exp(-abs(distance - counterpart[1]) / 3.) if counterpart and counterpart[0] == 1 else 0.
            p = probability["t4t5"][k]
            t = probability["temporal"][k]
            history = group["history"][j]
            vector = np.r_[p, t, phase[i], origin, agreement, distance,
                           raw["motion"][i, 1:], history, group["edge"][j]]
            results.append((i, origin, float(distance), float(p), float(agreement), vector))
    return baseline, results


METHOD_KEY = "forecasts"


def features(offers):
    return np.stack([a[-1] for a in offers])


def fit(groups, mode):
    xs, ys, ws = [], [], []
    for offers, categories in groups:
        y = categories == "hit_1"
        if not y.any() or y.all():
            continue
        _, counts = np.unique([a[0] for a in offers], return_inverse=True)
        w = 1. / np.bincount(counts)[counts]
        for side in (False, True):
            mask = y == side
            w[mask] /= w[mask].sum()
        xs.append(features(offers))
        ys.append(y)
        ws.append(w)
    x, y, w = map(np.concatenate, (xs, ys, ws))
    if mode == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.08, max_iter=800, random_state=SEED))
        return model.fit(x, y, logisticregression__sample_weight=w / w.mean())
    model = HistGradientBoostingClassifier(max_iter=70, max_leaf_nodes=8, min_samples_leaf=20,
                                          l2_regularization=2., random_state=SEED)
    return model.fit(x, y, sample_weight=w / w.mean())


def score_offers(offers, mode, model=None):
    if not offers:
        return np.empty(0)
    if mode == "structured":
        return np.asarray([a[3] for a in offers])
    if mode == "agreement":
        return np.asarray([a[3] * (.8 + .2 * a[4]) for a in offers])
    return model.predict_proba(features(offers))[:, 1]


def choose(offers, probability, route):
    best = {}
    for j, ((i, origin, *_), p) in enumerate(zip(offers, probability, strict=True)):
        if route != "either" and ROUTES[origin] != route:
            continue
        if not np.isfinite(p):
            continue
        if i not in best or p > probability[best[i]]:
            best[i] = j
    return list(best.values())


def operating_point(folds):
    all_scores = np.concatenate([p for p, _ in folds])
    best = None
    for cut in np.r_[np.unique(all_scores), 1.01]:
        per = [Counter(y[p >= cut]) for p, y in folds]
        total = sum(per, Counter())
        if (any(total[k] > cap for k, cap in RISK.items())
                or any(count[k] > cap for count in per for k, cap in PER_RECORDING.items())):
            continue
        rank = (total["hit_1"], sum(c["hit_1"] > 0 for c in per), -sum(total[k] for k in RISK), float(cut))
        if best is None or rank > best[0]:
            best = rank, dict(total), [dict(c) for c in per]
    rank, total, per = best
    return dict(cut=rank[-1], hits=rank[0], development_recordings_improved=rank[1],
                risk=sum(total.get(k, 0) for k in RISK), total=total, folds=dict(zip(DEVELOP, per, strict=True)))


def archive_mapping():
    path = RUNS / "vision-corner-pixel-checked-boundary-20261112-01/pixel_checked.py"
    sys.path.insert(0, str(path.parent))
    from pixel_checked import MAP
    return MAP, path


def calibrate():
    receipt = OUT / "frozen.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    observer, settings, sources = load_observer()
    training = SELECTED / "training.json"
    checkpoint = SELECTED / "confirmers.joblib"
    if old.digest(checkpoint) != old.read(training)["checkpoint_sha256"]:
        raise ValueError("Structured confirmer changed")
    structured = old.joblib.load(checkpoint)["structured"] if hasattr(old, "joblib") else None
    if structured is None:
        import joblib
        structured = joblib.load(checkpoint)["structured"]
    sources.update((Path(__file__), training, checkpoint, SELECTED / "selected_head.py",
                    Path(build.__code__.co_filename), Path(read_development.__code__.co_filename),
                    Path(label.__code__.co_filename)))
    mapping, mapping_source = archive_mapping()
    sources.add(mapping_source)
    groups, baselines = {}, {}
    for track in DEVELOP:
        sensor, truth, proof = read_development(track, sources, settings)
        raw = SensorArrays(sensor)
        archive = mapping[track]
        folders = {name: Path(path) for name, path in old.read(archive).items()}
        repaired, _, dependencies = recover_sections(raw, folders, proof["camera_bundle_sha256_by_run"], stream, image, old)
        sources.update((*dependencies, archive))
        baseline, offers = describe(raw, SensorArrays(dict(raw, sections=repaired)), observer, settings, structured)
        if keys(baseline) != keys(truth):
            raise ValueError("Development labels and sensor timeline differ")
        groups[track] = offers, np.asarray([label(truth[i], 1, d) for i, _, d, *_ in offers])
        baselines[track] = score(baseline, truth)
        print(track, len(offers), dict(Counter(groups[track][1])), flush=True)
    trials = {}
    for mode in MODES:
        signals = {}
        for held, (offers, categories) in groups.items():
            model = fit([g for track, g in groups.items() if track != held], mode) if mode in MODES[2:] else None
            signals[held] = score_offers(offers, mode, model)
        for route in ROUTES:
            data = []
            for track, (offers, categories) in groups.items():
                probability = signals[track]
                picked = choose(offers, probability, route)
                data.append((probability[picked], categories[picked]))
            trials[mode, route] = operating_point(data)
    mode, route = max(trials, key=lambda pair: (trials[pair]["hits"], trials[pair]["development_recordings_improved"],
                                              -trials[pair]["risk"], -MODES.index(pair[0]), -ROUTES.index(pair[1])))
    import joblib
    fitted = OUT / "left-policy.joblib"
    joblib.dump(fit(list(groups.values()), mode) if mode in MODES[2:] else None, fitted)
    old.write(receipt, dict(status="frozen_before_comparison", development=list(DEVELOP),
              exclusions=[t for t, _ in EVAL], primary=METHOD, settings=settings, chosen=dict(mode=mode, route=route),
              cut=trials[mode, route]["cut"], trials={f"{m}/{r}": v for (m, r), v in trials.items()},
              risk_caps=RISK, per_recording_risk_caps=PER_RECORDING, development_baselines=baselines,
              checkpoint_sha256=old.digest(fitted), source_sha256=digests(sources), seed=SEED,
              source_revision=old.revision(), termination="development_calibration_complete"))
    print("Frozen", mode, route, trials[mode, route], flush=True)


def forecast(baseline, offers, policy, model):
    p = score_offers(offers, policy["chosen"]["mode"], model)
    output = list(baseline)
    accepted = {}
    for j in choose(offers, p, policy["chosen"]["route"]):
        if p[j] < policy["cut"]:
            continue
        i, origin, distance, *_ = offers[j]
        row = baseline[i]
        if row[METHOD_KEY][METHOD]["state"] not in NO_ALERT:
            raise ValueError("Supplement attempted to replace an existing turn")
        output[i] = dict(row, candidate=[1, distance], forecasts={**row[METHOD_KEY],
                         METHOD: dict(state="left", distance_m=distance)})
        accepted[i] = dict(origin=ROUTES[origin], distance_m=distance, probability=float(p[j]))
    return output, accepted


def score(output, truth):
    indexed = dict(zip(keys(output), output, strict=True))
    if len(indexed) != len(output) or len(set(keys(truth))) != len(truth):
        raise ValueError("Scoring requires unique forecast and truth frame identities")
    counts, states = Counter(), Counter()
    approach, errors = defaultdict(Counter), defaultdict(list)
    events = defaultdict(list)
    active = None
    previous = None
    for row in truth:
        key = row["run"], row["frame"]
        record = indexed.get(key)
        f = record[METHOD_KEY].get(METHOD) if record is not None else None
        state = f["state"] if f is not None else "no_output"
        side = {"left": 1, "right": 2}.get(state)
        distance = f["distance_m"] if f is not None else None
        if side and (distance is None or not 4. <= distance <= 20.):
            raise ValueError(f"Invalid turn distance: {key}")
        category = label(row, side, distance) if side else "quiet"
        counts[category] += 1
        states[state] += 1
        if row["truth"] != "approach":
            active = None
            previous = None
            continue
        direction = row["label"]
        group = approach[direction]
        group["frames"] += 1
        group[state] += 1
        if side == int(direction):
            group["direction_hit"] += 1
            errors[direction].append(max(abs(distance - d) for d in row["truth_endpoints_m"]))
        hit = category == f"hit_{direction}"
        group["joint_hit"] += int(hit)
        group["miss"] += int(not hit)
        if active is None or previous != (key[0], key[1] - 1, direction):
            active = []
            events[direction].append(active)
        active.append(hit)
        previous = (*key, direction)
    by_direction = {}
    for direction, values in approach.items():
        groups = events[direction]
        error = errors[direction]
        by_direction[direction] = dict(values, windows=len(groups), windows_hit=sum(any(a) for a in groups),
                                       windows_consecutive_hit=sum(any(b and c for b, c in zip(a, a[1:])) for a in groups),
                                       distance_n=len(error), distance_mae_m=float(np.mean(error)) if error else None)
    return dict(frames=len(output), labeled=len(truth), counts=dict(counts), states=dict(states),
                approaches=by_direction)


def compare():
    import joblib
    receipt = OUT / "frozen.json"
    policy = old.read(receipt)
    fitted = OUT / "left-policy.joblib"
    if old.digest(fitted) != policy["checkpoint_sha256"]:
        raise ValueError("Fixed model changed")
    for path, expected in policy["source_sha256"].items():
        if old.digest(Path(path)) != expected:
            raise ValueError(f"Fixed input changed: {path}")
    observer, settings, _ = load_observer()
    if settings != policy["settings"]:
        raise ValueError("Observer settings changed")
    structured = joblib.load(SELECTED / "confirmers.joblib")["structured"]
    model = joblib.load(fitted)
    result = {}
    for track, name in EVAL:
        directory = CLI / ("vallelunga-retry" if name == "vallelunga" else name)
        input_receipt = directory / "result.json"
        proof = old.read(input_receipt)
        raw, recovered = directory / "sensors.npz", directory / "recovered-sensors.npz"
        baseline, offers = describe(load_sensor(raw, proof["inputs_sha256"]),
                                    load_sensor(recovered, proof["recovered_input_sha256"]),
                                    observer, settings, structured)
        previous = directory / "predictions/forecasts.jsonl"
        if baseline != frames(previous, old.read(previous.with_name("result.json"))["forecasts_sha256"]):
            # Observe returns tuples for candidates, whereas the original JSON uses arrays.
            for a, b in zip(baseline, frames(previous, old.read(previous.with_name("result.json"))["forecasts_sha256"]), strict=True):
                if dict(a, candidate=list(a["candidate"]) if a["candidate"] is not None else None) != b:
                    raise ValueError("Baseline predictions do not reproduce the archived comparator")
        revised, accepted = forecast(baseline, offers, policy, model)
        trace = OUT / f"{track}-forecasts.jsonl"
        old.jsonl(trace, revised)
        truth, labels = reference(track)
        before, after = score(baseline, truth), score(revised, truth)
        delta = {k: after["counts"].get(k, 0) - before["counts"].get(k, 0) for k in RISK}
        truth_index = dict(zip(keys(truth), truth, strict=True))
        changes = [dict(offer, run=baseline[i]["run"], frame=baseline[i]["frame"],
                        truth=truth_index.get((baseline[i]["run"], baseline[i]["frame"])))
                   for i, offer in accepted.items()]
        result[track] = dict(before=before, after=after, added_risk=delta, changes=changes,
                             per_recording_cap_pass=all(delta[k] <= cap for k, cap in PER_RECORDING.items()),
                             source_sha256=digests((raw, recovered, input_receipt, labels, previous,
                                                    previous.with_name("result.json"))),
                             forecasts_sha256=old.digest(trace), in_corner_labeled=track != MONZA)
        print(track, before["approaches"], after["approaches"], delta, flush=True)
    old.write(OUT / "comparison.json", dict(tracks=result, freeze_sha256=old.digest(receipt), termination="archive_end"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "compare"))
    stage = parser.parse_args().stage
    (calibrate if stage == "calibrate" else compare)()
