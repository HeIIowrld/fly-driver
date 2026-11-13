"""Recalibrate left forecasts as a group while preserving the fixed right forecast."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "vision-corner-fixed-left-20261113-01"))
from fixed_left import (
    CLI, DEVELOP, EVAL, METHOD, MONZA, NO_ALERT, PER_RECORDING, RISK, SELECTED, SEED, SensorArrays,
    archive_mapping, build, digests, fit, frames, image, keys, label, load_observer, load_sensor, observe, old,
    phase_scores, read_development, recover_sections, reference, score, score_offers, scores, stream,
)

METHODS = ("structured", "logistic")
CANDIDATES = ("raw", "recovered", "best")
ERRORS = (*RISK, "hit_1", "hit_2")
REJECT = 1.01


def describe(raw, recovered, observer, settings, structured):
    baseline = observe(recovered, observer, settings)[0]
    raw_rows = observe(raw, observer, settings)[0]
    if keys(baseline) != keys(raw_rows):
        raise ValueError("Raw and recovered timelines differ")
    offers = []
    for origin, (data, rows, other) in enumerate(((raw, raw_rows, baseline), (recovered, baseline, raw_rows))):
        built = build(data, settings)
        if keys(rows) != keys(built["rows"]):
            raise ValueError("Candidate and forecast timelines differ")
        group = built["group"]
        phase = phase_scores(group, observer["phase"])
        lookup = {(i, offer): j for j, (i, offer) in enumerate(zip(group["index"], group["offers"], strict=True))}
        selected = []
        for i, row in enumerate(rows):
            offer = row["candidate"]
            if (offer is not None and offer[0] == 1 and baseline[i]["forecasts"][METHOD]["state"] != "right"
                    and baseline[i]["motion_vision_available"] and phase[i, 2] <= settings["phase_cut"]):
                selected.append((i, lookup[i, tuple(offer)]))
        indices = np.asarray([j for _, j in selected], dtype=int)
        if not len(indices):
            continue
        probabilities = scores(group, indices, structured)
        for k, (i, j) in enumerate(selected):
            distance = group["offers"][j][1]
            counterpart = other[i]["candidate"]
            agree = np.exp(-abs(distance-counterpart[1])/3.) if counterpart and counterpart[0] == 1 else 0.
            p, t = probabilities["t4t5"][k], probabilities["temporal"][k]
            vector = np.r_[p, t, phase[i], origin, agree, distance,
                           raw["motion"][i, 1:], group["history"][j], group["edge"][j]]
            offers.append((i, origin, float(distance), float(p), float(agree), vector))
    return baseline, offers


def route(baseline, offers, probabilities, choice):
    incumbent = {}
    alternate = {}
    for row, probability in zip(offers, probabilities, strict=True):
        i, origin, distance, *_ = row
        p = float(probability)
        state = baseline[i]["forecasts"][METHOD]["state"]
        if state == "left":
            if origin == 1:
                incumbent[i] = (p, distance)
            continue
        if state not in NO_ALERT or (choice != "best" and origin != CANDIDATES.index(choice)):
            continue
        if i not in alternate or p > alternate[i][0]:
            alternate[i] = (p, distance)
    original_left = {i for i, r in enumerate(baseline) if r["forecasts"][METHOD]["state"] == "left"}
    if set(incumbent) != original_left:
        raise ValueError("An existing left forecast lacks its original confirmation score")
    return incumbent, alternate


def cumulative(table, truth, cuts, suppress=False):
    """Count all changes at every possible observed score threshold."""
    counts = np.zeros((len(cuts), len(ERRORS)), dtype=int)
    if not table:
        return counts
    rows = sorted((p, label(truth[i], 1, d)) for i, (p, d) in table.items())
    p = np.asarray([a for a, _ in rows])
    y = np.asarray([b for _, b in rows])
    boundary = p.searchsorted(cuts, side="left")
    for col, category in enumerate(ERRORS):
        prefix = np.r_[0, np.cumsum(y == category)]
        counts[:, col] = -prefix[boundary] if suppress else prefix[-1]-prefix[boundary]
    return counts


def select(rows):
    keep_scores = np.unique([0., REJECT, *(p for incumbent, _, _ in rows for p, _ in incumbent.values())])
    add_scores = np.unique([0., REJECT, *(p for _, alternate, _ in rows for p, _ in alternate.values())])
    keep = np.stack([cumulative(a, truth, keep_scores, True) for a, _, truth in rows])
    add = np.stack([cumulative(b, truth, add_scores) for _, b, truth in rows])
    # Choose the same two thresholds for all tracks. Evaluate net changes for each error class.
    risk = np.array(list(RISK.values()))
    per_recording = np.array(list(PER_RECORDING.values()))
    n_risk = len(RISK)
    best = None
    for i, keep_cut in enumerate(keep_scores):
        per = keep[:, i, None, :] + add
        total = per.sum(axis=0)
        permitted = ((per[:, :, n_risk] >= 0).all(axis=0)
                     & (per[:, :, :n_risk] <= per_recording).all(axis=(0, 2))
                     & (total[:, :n_risk] <= risk).all(axis=1))
        for j in np.flatnonzero(permitted):
            key = (int(total[j, n_risk]), int((per[:, j, n_risk] > 0).sum()),
                   -int(total[j, :n_risk].sum()), float(add_scores[j]), -float(keep_cut))
            if best is None or key > best[0]:
                best = key, per[:, j].tolist(), float(keep_cut), float(add_scores[j])
    key, folds, keep_cut, add_cut = best
    return dict(added_hits=key[0], tracks_improved=key[1], net_risk=-key[2], keep=keep_cut, add=add_cut,
                per_track=dict(zip(DEVELOP, (dict(zip(ERRORS, c, strict=True)) for c in folds), strict=True)))


def dependencies():
    observer, settings, paths = load_observer()
    checkpoint = SELECTED / "confirmers.joblib"
    previous = SELECTED / "training.json"
    if old.digest(checkpoint) != old.read(previous)["checkpoint_sha256"]:
        raise ValueError("Confirmation checkpoint hash mismatch")
    paths.update((Path(__file__), previous, checkpoint, SELECTED / "selected_head.py",
                  Path(build.__code__.co_filename), Path(fit.__code__.co_filename)))
    return observer, settings, joblib.load(checkpoint)["structured"], paths


def calibrate():
    frozen = OUT / "frozen.json"
    if frozen.exists():
        raise FileExistsError(frozen)
    observer, settings, structured, paths = dependencies()
    mapping, mapping_path = archive_mapping()
    paths.add(mapping_path)
    dev = {}
    for track in DEVELOP:
        sensor, truth, proof = read_development(track, paths, settings)
        raw = SensorArrays(sensor)
        archive = mapping[track]
        folders = {name: Path(path) for name, path in old.read(archive).items()}
        repaired, _, sources = recover_sections(raw, folders, proof["camera_bundle_sha256_by_run"], stream, image, old)
        paths.update((*sources, archive))
        baseline, offers = describe(raw, SensorArrays(dict(raw, sections=repaired)), observer, settings, structured)
        if keys(baseline) != keys(truth):
            raise ValueError("Development truth and timeline differ")
        target = np.array([label(truth[i], 1, d) for i, _, d, *_ in offers])
        dev[track] = baseline, offers, truth, target
        print(track, len(offers), dict(Counter(target)), flush=True)
    trials = {}
    for method in METHODS:
        held_scores = {}
        for track, (_, offers, _, _) in dev.items():
            training = [(o, y) for t, (_, o, _, y) in dev.items() if t != track]
            model = fit(training, method) if method == "logistic" else None
            held_scores[track] = score_offers(offers, method, model)
        for choice in CANDIDATES:
            rows = [(*route(base, offers, held_scores[track], choice), truth)
                    for track, (base, offers, truth, _) in dev.items()]
            trials[method, choice] = select(rows)
    def priority(trial):
        result = trials[trial]
        return result["added_hits"], result["tracks_improved"], -result["net_risk"]
    method, choice = max(trials, key=priority)
    fitted = OUT / "policy.joblib"
    model = fit([(o, y) for _, o, _, y in dev.values()], method) if method == "logistic" else None
    joblib.dump(model, fitted)
    old.write(frozen, dict(chosen=dict(method=method, source=choice, **trials[method, choice]),
              trials={f"{m}/{c}": r for (m, c), r in trials.items()},
              risk_caps=RISK, per_track_caps=PER_RECORDING, settings=settings,
              development=list(DEVELOP), exclusions=[t for t, _ in EVAL],
              seed=SEED, source_revision=old.revision(), model_sha256=old.digest(fitted),
              source_sha256=digests(paths), termination="frozen_before_comparison"))
    print("Frozen", method, choice, trials[method, choice], flush=True)


def apply_policy(baseline, offers, policy, model):
    chosen = policy["chosen"]
    probability = score_offers(offers, chosen["method"], model)
    incumbent, alternate = route(baseline, offers, probability, chosen["source"])
    output = list(baseline)
    changes = {}
    for i, (p, distance) in incumbent.items():
        if p >= chosen["keep"]:
            continue
        row = baseline[i]
        output[i] = dict(row, forecasts={**row["forecasts"], METHOD: dict(state="abstain", distance_m=None)})
        changes[i] = dict(action="suppress", distance_m=distance, probability=p)
    for i, (p, distance) in alternate.items():
        if p < chosen["add"]:
            continue
        row = baseline[i]
        # The paired edge forecast retains its original candidate in this experiment.
        output[i] = dict(row, forecasts={**row["forecasts"], METHOD: dict(state="left", distance_m=distance)})
        changes[i] = dict(action="add", distance_m=distance, probability=p)
    return output, changes


def compare():
    frozen, fitted = OUT / "frozen.json", OUT / "policy.joblib"
    policy = old.read(frozen)
    if old.digest(fitted) != policy["model_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    if any(old.digest(Path(path)) != digest for path, digest in policy["source_sha256"].items()):
        raise ValueError("Frozen source changed")
    observer, settings, structured, _ = dependencies()
    if settings != policy["settings"]:
        raise ValueError("Observer settings changed")
    model = joblib.load(fitted)
    results = {}
    for track, name in EVAL:
        directory = CLI / ("vallelunga-retry" if name == "vallelunga" else name)
        receipt = directory / "result.json"
        proof = old.read(receipt)
        raw, recovered = directory / "sensors.npz", directory / "recovered-sensors.npz"
        baseline, offers = describe(load_sensor(raw, proof["inputs_sha256"]),
                                    load_sensor(recovered, proof["recovered_input_sha256"]),
                                    observer, settings, structured)
        prior = directory / "predictions/forecasts.jsonl"
        old_rows = frames(prior, old.read(prior.with_name("result.json"))["forecasts_sha256"])
        for a, b in zip(baseline, old_rows, strict=True):
            if dict(a, candidate=list(a["candidate"]) if a["candidate"] is not None else None) != b:
                raise ValueError("Baseline replay changed")
        output, changes = apply_policy(baseline, offers, policy, model)
        if any(a != b for a, b in zip(baseline, output, strict=True) if a["forecasts"][METHOD]["state"] == "right"):
            raise ValueError("Existing right forecast changed")
        truth, label_file = reference(track)
        by_key = dict(zip(keys(truth), truth, strict=True))
        before, after = score(baseline, truth), score(output, truth)
        risk = {k: after["counts"].get(k, 0)-before["counts"].get(k, 0) for k in RISK}
        witness = OUT / f"{track}-forecasts.jsonl"
        old.jsonl(witness, output)
        change_rows = []
        for i, change in sorted(changes.items()):
            key = baseline[i]["run"], baseline[i]["frame"]
            ref = by_key.get(key)
            category = label(ref, 1, change["distance_m"]) if ref is not None else "unlabeled"
            change_rows.append(dict(change, run=key[0], frame=key[1], label=category, truth=ref))
        results[track] = dict(before=before, after=after, net_risk=risk, changes=change_rows,
            per_track_cap_pass=all(risk[k] <= cap for k, cap in PER_RECORDING.items()),
            forecasts_sha256=old.digest(witness),
            source_sha256=digests((raw, recovered, receipt, prior, prior.with_name("result.json"), label_file)),
            in_corner_labeled=track != MONZA)
        print(track, before["counts"], after["counts"], risk, flush=True)
    old.write(OUT / "comparison.json", dict(tracks=results, freeze_sha256=old.digest(frozen),
              termination="archive_end"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "compare"))
    (calibrate if parser.parse_args().stage == "calibrate" else compare)()
