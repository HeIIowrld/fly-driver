"""Learn when consecutive visual alerts refer to the same upcoming corner."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import anchor_distance as anchor
from evaluate_fused_corner import frame_key
import hinge
from profile_replay import CALIBRATION_TRACKS, TRACKS
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import score

OUT = Path(__file__).parent
CUTS = (.40, .50, .60, .70, .75, .80, .85, .90, .95)
MODES = ("distance", "boundary")


def presence(row):
    value = row.get("presence_score")
    return (0., 0.) if value is None else (float(value), 1.)


def visible(proposal, side):
    if proposal is None:
        return (0., 0., 0.)
    gain, direction, distance = proposal
    return float(direction == side), gain, distance / 20.


def pair_features(rows, travel, geometry):
    positions, features = [], []
    for i in range(1, len(rows)):
        a, b = rows[i - 1:i + 1]
        first, second = anchor.raw_observation(a), anchor.raw_observation(b)
        move = travel[i]
        if (first is None or second is None or first[0] != second[0]
                or not adjacent(a, b) or not 0. <= move <= 12.):
            continue
        side, previous = first
        current = second[1]
        projected = previous - move
        innovation = current - projected
        aa, bb = visible(geometry[i - 1], side), visible(geometry[i], side)
        features.append((previous / 20., current / 20., move / 12., innovation / 20.,
                         abs(innovation) / 20., float(projected >= 0.), *presence(a), *presence(b), *aa, *bb,
                         abs((bb[2] - aa[2]) * 20. + move) / 20. if aa[0] and bb[0] else 0.))
        positions.append(i)
    return np.asarray(positions, dtype=int), np.asarray(features, dtype=float).reshape(-1, 17)


def target(rows, index):
    a, b = rows[index - 1:index + 1]
    known = ("approach", "straight", "out_of_range", "in_corner", "band_boundary")
    if a["truth"] not in known or b["truth"] not in known:
        return None
    side = str(anchor.DIRECTIONS[b["state"]])
    da, db = a["truth_distance_m"], b["truth_distance_m"]
    return (a["physical_corner"] is not None and a["physical_corner"] == b["physical_corner"]
            and a["label"] == b["label"] == side and da is not None and db is not None
            and 0. <= db < da <= 30. and (a["truth"] == "approach" or b["truth"] == "approach"))


def feature_mode(x, mode):
    return x[:, :10] if mode == "distance" else x


def train(examples, mode):
    x, y, weights = [], [], []
    for values, labels in examples:
        known = labels >= 0
        selected = labels[known]
        x.append(feature_mode(values[known], mode))
        y.append(selected)
        weights.append(np.where(selected == 1, .5 / max(1, (selected == 1).sum()),
                                .5 / max(1, (selected == 0).sum())))
    x, y, w = map(np.concatenate, (x, y, weights))
    w *= len(w) / w.sum()
    center = np.average(x, axis=0, weights=w)
    spread = np.sqrt(np.average(np.square(x - center), axis=0, weights=w))
    spread[spread < 1e-9] = 1.
    model = LogisticRegression(C=.1).fit((x - center) / spread, y, sample_weight=w)
    return center, spread, model


def predict(model, features, mode):
    center, spread, classifier = model
    return classifier.predict_proba((feature_mode(features, mode) - center) / spread)[:, 1]


def apply(rows, travel, pair_indices, probability, cutoff):
    probabilities = dict(zip(pair_indices, probability, strict=True))
    preceding = None
    for i, (row, move) in enumerate(zip(rows, travel, strict=True)):
        raw = anchor.raw_observation(row)
        if raw is None:
            preceding = None
            yield row
            continue
        side, measurement = raw
        estimate = measurement
        if cutoff is not None and preceding is not None and probabilities.get(i, -1.) >= cutoff:
            predicted = preceding - move
            innovation = measurement - predicted
            if predicted >= 0. and abs(innovation) <= 8.:
                estimate = float(np.clip(predicted + np.clip(innovation, -2., 2.), 0., 20.))
        yield score(row, (side, estimate)) if measurement != estimate else row
        preceding = estimate


def evaluate(group, rows, move, cutoff, predictions):
    idx, p = predictions
    return anchor.evaluate(group, list(apply(rows, move, idx, p, cutoff)))


def choose_cut(groups, calibration):
    baseline = {k:anchor.evaluate(groups[k], groups[k]["rows"]) for k in calibration}
    ranked = []
    for cutoff in (*CUTS, None):
        gain = 0.
        for name, (travel, predictions) in calibration.items():
            group = groups[name]
            old = baseline[name]
            new = evaluate(group, group["rows"], travel, cutoff, predictions)
            a, b = (r["distances"]["candidate"] for r in (old, new))
            if (new["metrics"]["candidate"]["joint"] < old["metrics"]["candidate"]["joint"]
                    or len(new["stable_events"]) < len(old["stable_events"])
                    or b["receding"] > a["receding"]):
                break
            if a["mae_m"] is not None:
                if b["mae_m"] > a["mae_m"] + 1e-9:
                    break
                gain += a["mae_m"] - b["mae_m"]
        else:
            ranked.append(((gain, cutoff is None, cutoff or 0.), cutoff))
    return max(ranked)[1]


def opportunity(group, combined, geometry):
    events = []
    rows = group["rows"]
    for run in dict.fromkeys(r["run"] for r in rows):
        timeline = [r for r in rows if r["run"] == run]
        for event in anchor.summarize(timeline)["events"]:
            relevant = [i for i,r in enumerate(rows) if r["run"] == run and r["truth"] == "approach"
                        and event["first_frame"] <= r["frame"] <= event["last_frame"]
                        and r["physical_corner"] == event["physical_corner"]]
            pairs = [(a,b) for a,b in zip(relevant, relevant[1:]) if adjacent(rows[a], rows[b])]
            eligible = sum(geometry[i] is not None for i in relevant)
            events.append(dict(run=run, **event, opportunities=len(pairs),
                baseline_direction_pairs=sum(rows[a]["direction_hit"] and rows[b]["direction_hit"] for a,b in pairs),
                combined_direction_pairs=sum(combined[a]["direction_hit"] and combined[b]["direction_hit"] for a,b in pairs),
                observed_visual_hinges=eligible,
                timeline=[dict(frame=rows[i]["frame"], actual_m=rows[i]["truth_distance_m"],
                               fixed=rows[i]["state"], combined=combined[i]["state"]) for i in relevant]))
    return events


def run():
    hashes = {Path(__file__), anchor.OUT / "anchor_distance.py", hinge.OUT / "hinge.py",
              Path("scripts/score_corner_captures.py"),
              Path("runs/vision-corner-onset-support-20261111-01/transfer.py")}
    shape = {(r["track"], *frame_key(r)):r for r in anchor.verified_predictions(anchor.SHAPE, hashes)
             if r["mode"] == "aligned"}
    learned = {(r["track"], *frame_key(r)):r["proposal"] for r in anchor.verified_predictions(anchor.LEARNED, hashes)}
    groups, travel, fixed, combined, audit = {}, {}, {}, {}, {}
    for name in TRACKS:
        group = hinge.group(name, hashes)
        groups[name] = group
        rows = group["rows"]
        travel[name] = anchor.read_travel(name, rows, hashes)
        geometry = [p for _, p in group["fits"]["current"]]
        extra = list(anchor.supplementary(rows, name, shape, learned))
        idx, x = pair_features(rows, travel[name], geometry)
        labels = [target(rows, i) for i in idx]
        fixed[name] = idx, x, np.array([-1 if y is None else int(y) for y in labels])
        ext_idx, ext_x = pair_features(extra, travel[name], geometry)
        combined[name] = extra, ext_idx, ext_x
        audit[name] = opportunity(group, extra, geometry)
        print(name, "consecutive same-side alerts", len(idx), "known association positives",
              int((fixed[name][2] == 1).sum()), flush=True)

    @lru_cache(maxsize=None)
    def trained(train_names, mode):
        if "spa" in train_names:
            raise ValueError("Spa excluded from training and selection")
        return train([(fixed[k][1], fixed[k][2]) for k in train_names], mode)

    results = {}
    for name in TRACKS:
        rest = tuple(k for k in CALIBRATION_TRACKS if k != name)
        idx, x, labels = fixed[name]
        result = {}
        for mode in MODES:
            calibration = {}
            for key in rest:
                train_names = tuple(k for k in rest if k != key)
                cal_idx, cal_x, _ = fixed[key]
                cal_p = predict(trained(train_names, mode), cal_x, mode)
                calibration[key] = travel[key], (cal_idx, cal_p)
            cutoff = choose_cut(groups, calibration)
            model = trained(rest, mode)
            probability = predict(model, x, mode)
            known = labels >= 0
            rows, ext_idx, ext_x = combined[name]
            ext_prob = predict(model, ext_x, mode)
            evaluated = {}
            for base, values, preds in (("fixed", groups[name]["rows"], (idx, probability)),
                                         ("combined", rows, (ext_idx, ext_prob))):
                evaluated[base] = dict(
                    chosen=evaluate(groups[name], values, travel[name], cutoff, preds),
                    baseline=evaluate(groups[name], values, travel[name], None, preds),
                    sweep={str(c):evaluate(groups[name], values, travel[name], c, preds) for c in CUTS})
            result[mode] = dict(cutoff=cutoff, association_average_precision=float(average_precision_score(labels[known], probability[known])),
                association_positive_rate=float(labels[known].mean()), evaluated=evaluated)
            print(name, mode, cutoff, result[mode]["association_average_precision"],
                  evaluated["combined"]["chosen"]["metrics"]["candidate"], flush=True)
        results[name] = dict(calibration_circuits=rest, models=result)
    event_path = OUT / "opportunities.json"
    write(event_path, audit)
    write(OUT / "result.json", dict(tracks=results, opportunities_sha256=digest(event_path),
          parameters=dict(cutoffs=CUTS, correction_limit_m=2., reset_innovation_m=8., seed=None),
          source_sha256={str(p):digest(p) for p in sorted(hashes)},
          termination="causal_corner_association_and_opportunity_comparison_complete"))


if __name__ == "__main__":
    run()
