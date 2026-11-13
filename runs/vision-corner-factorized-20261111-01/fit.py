"""Development-only comparison of factorized next-corner readouts."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from infer_frozen_next_corner import contrast, load_features, load_readout
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
runs = out.parent
hashes = {}


def checked(path, receipt=None, key=None):
    hashes[str(path)] = digest(path)
    if receipt is not None:
        hashes[str(receipt)] = digest(receipt)
        assert read(receipt)[key] == hashes[str(path)], path
    return path


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def key(row):
    return row["run"], row["frame"]


def monza_truth(row):
    direction = row["direction"]
    distance = row["distance"]
    approach = direction and distance is not None and 4 <= distance <= 20
    return dict(row, truth="approach" if approach else ("out_of_range" if direction else "straight"),
                label=str(direction), truth_endpoints_m=[] if distance is None else [distance])


def from_score(timeline, a, b, available, index):
    assert len(index) == len(a) == len(b) == len(available)
    positions = {key(row): i for i, row in enumerate(index)}
    assert len(positions) == len(index)
    assert positions.keys() <= {key(row) for row in timeline}
    direct = np.zeros((len(timeline), a.shape[1]), dtype=a.dtype)
    reflected = np.zeros_like(direct)
    valid = np.zeros(len(timeline), dtype=bool)
    for i, row in enumerate(timeline):
        j = positions.get(key(row))
        if j is not None and available[j]:
            direct[i], reflected[i], valid[i] = a[j], b[j], True
    return timeline, direct, reflected, valid, valid


def baseline(cohort, predictor):
    timeline, direct, reflected, valid, mirrored = cohort
    score = np.full((len(timeline), 3), np.nan)
    distance = np.full(len(timeline), np.nan)
    gate = np.full(len(timeline), np.nan)
    for i in np.flatnonzero(valid):
        result = predictor(direct[i:i+1], reflected[i:i+1] if mirrored[i] else None, return_scores=True)
        score[i], distance[i], gate[i] = result[0][0], result[1][0], result[2][0]
    return score, distance, gate


def summarize(timeline, valid, direction, detected, distance):
    approach = np.array([r["truth"] == "approach" for r in timeline])
    truth = np.array([int(r["label"]) if r["label"] in ("1", "2") else 0 for r in timeline])
    hits = approach & detected & (direction == truth)
    error = np.array([max((abs(d-x) for x in r["truth_endpoints_m"]), default=np.inf)
                      for r, d in zip(timeline, distance, strict=True)])
    result = dict(frames=len(timeline), unavailable=int((~valid).sum()),
                  straight_output=int((valid & (direction == 0)).sum()),
                  abstain=int((~valid | (valid & (direction != 0) & ~detected)).sum()),
                  left_total=int((approach & (truth == 1)).sum()),
                  right_total=int((approach & (truth == 2)).sum()),
                  left=int((hits & (truth == 1)).sum()), right=int((hits & (truth == 2)).sum()),
                  joint=int((hits & (error <= 5)).sum()),
                  missed_presence=int((approach & valid & ~detected).sum()),
                  conditional_side_errors=int((approach & detected & (direction != truth)).sum()))
    for name in ("straight", "out_of_range"):
        mask = np.array([r["truth"] == name for r in timeline])
        result[name] = int((mask & detected).sum())
        result[name+"_total"] = int(mask.sum())
    return result


def normalize(features, mode):
    values = contrast(features)
    if mode == "contrast":
        return values
    grid = values.reshape(-1, 3, 4, 8, 16)
    rms = np.sqrt(np.mean(grid**2, axis=(1, 3, 4), keepdims=True)).clip(.025)
    return (grid/rms).reshape(values.shape)


def fit(x, target, groups, alpha):
    counts = Counter(groups)
    classes = Counter(group[0] for group in counts)
    weight = np.array([1/(classes[g[0]]*counts[g]) for g in groups])
    weight *= len(weight)/weight.sum()
    mean = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weight)).clip(.04)
    design = (x-mean)/scale
    offset = np.average(target, weights=weight)
    gram = design.T @ (weight[:, None]*design)
    gram.flat[::len(gram)+1] += alpha
    coef = solve(gram, design.T @ (weight*(target-offset)), assume_a="pos")/scale
    return coef, float(offset-mean@coef)


monza = runs/"vision-monza-two-lap-presence-20261111-01"
receipt = monza/"flyvis-result.json"
all_monza = list(map(monza_truth, rows(checked(monza/"rows.jsonl", receipt, "rows_sha256"))))
with np.load(checked(monza/"features.npz", receipt, "features_sha256"), allow_pickle=False) as z:
    a, b, mirrored = (z[name] for name in ("direct", "reflected", "reflected_available"))
train = np.array([r["split"] == "train" for r in all_monza])
validation = ~train
assert {r["physical_group"] for r in all_monza if r["split"] == "train"}.isdisjoint(
    r["physical_group"] for r in all_monza if r["split"] != "train")
cohorts = {"monza_validation": ([r for r in all_monza if r["split"] != "train"],
                                a[validation], b[validation], np.ones(validation.sum(), bool), mirrored[validation])}

magione = runs/"vision-corner-independent-runner-rehearsal-20261111-01"/"replay"
feature_root = magione/"features"
index = checked(feature_root/"index.jsonl")
mag_index = rows(index)
branches = []
for name in ("original", "reflected"):
    path = checked(feature_root/name/"features.npz")
    checked(path.with_name("flyvis-result.json"))
    branches.append(load_features(path, digest(index), len(mag_index)))
mag_rows = rows(checked(magione/"parent"/"score"/"rows.jsonl"))
cohorts["magione_transfer"] = from_score(mag_rows, branches[0][0], branches[1][0],
                                         branches[0][1] & branches[1][1], mag_index)

imola = runs/"vision-corner-imola-full-feature-replay-20261111-01"
imola_rows = rows(checked(runs/"vision-corner-imola-full-parent-20261111-01"/"rows.jsonl"))
imola_index = rows(checked(imola/"index.jsonl", imola/"result.json", "index_sha256"))
with np.load(checked(imola/"features.npz", imola/"result.json", "features_sha256"), allow_pickle=False) as z:
    imola_a, imola_b = z["features"].transpose(1, 0, 2)
cohorts["imola_negative_transfer"] = from_score(imola_rows, imola_a, imola_b,
                                                np.ones(len(imola_index), bool), imola_index)

summary = {}
references = {}
base_alerts = {}
base_presence = {}
baseline_sides = {}
baseline_alerts = {}
for name, pin_dir in (("parent", "vision-frozen-next-corner-20261110-01"),
                      ("previous_candidate", "vision-corner-blended-candidate-20261111-02")):
    pins = read(checked(runs/pin_dir/"pins.json"))
    predictor, gate_threshold = load_readout(pins)
    for model in pins["models"].values():
        checked(Path(model["path"]))
    summary[name] = {}
    baseline_sides[name], baseline_alerts[name] = {}, {}
    for cohort, data in cohorts.items():
        score, distance, gate = baseline(data, predictor)
        direction = score.argmax(1)
        direction[(score == score.max(1, keepdims=True)).sum(1)>1] = 0
        detected = (direction != 0) & (gate >= gate_threshold) & (distance <= 20)
        summary[name][cohort] = summarize(data[0], data[3], direction, detected, distance)
        baseline_sides[name][cohort] = direction
        baseline_alerts[name][cohort] = detected
        if name == "parent":
            references[cohort] = distance
            base_presence[cohort] = data[3] & (gate >= gate_threshold) & (distance <= 20)
        else:
            base_alerts[cohort] = detected

training_rows = [r for r, keep in zip(all_monza, train, strict=True) if keep]
training_rows += [r for r, keep in zip(all_monza, train & mirrored, strict=True) if keep]
was_mirrored = np.r_[np.zeros(train.sum(), bool), np.ones((train & mirrored).sum(), bool)]
presence_target = np.array([int(r["truth"] == "approach") for r in training_rows])
side_target = np.array([int(r["label"]) for r in training_rows])
side_target[was_mirrored & (side_target != 0)] = 3-side_target[was_mirrored & (side_target != 0)]
positive = presence_target.astype(bool)
presence_groups = [(target, r["physical_group"], r["event"])
                   for target, r in zip(presence_target, training_rows, strict=True)]
side_groups = [(side, r["physical_group"], r["event"]) for side, r, near in
               zip(side_target, training_rows, positive, strict=True) if near]
predictions = []
diagnostics = {}
for mode in ("contrast", "view_rms"):
    fit_features = normalize(np.vstack((a[train], b[train & mirrored])), mode)
    transformed = {name: (normalize(data[1], mode), normalize(data[2], mode))
                   for name, data in cohorts.items()}
    for alpha in (100., 1000., 10000.):
        pc, pi = fit(fit_features, presence_target, presence_groups, alpha)
        sc, si = fit(fit_features[positive], (side_target[positive] == 2)*2.-1., side_groups, alpha)
        model = f"{mode}-{int(alpha)}"
        np.savez_compressed(out/(model+".npz"), presence_coef=pc, presence_intercept=pi,
                            side_coef=sc, side_intercept=si)
        diagnostics[model] = {}
        for cohort, data in cohorts.items():
            timeline, _, _, valid, has_mirror = data
            x, y = transformed[cohort]
            presence_score = x@pc + pi
            signed = x@sc + si
            presence_score[has_mirror] = .5*(presence_score[has_mirror] + y[has_mirror]@pc + pi)
            signed[has_mirror] = .5*(signed[has_mirror] - (y[has_mirror]@sc + si))
            side = np.where(signed > 0, 2, 1)
            distance = references[cohort]
            diagnostics[model][cohort] = {
                str(s): [int((m & (side == s)).sum()), int(m.sum())]
                for s in (1, 2)
                for m in [valid & np.array([r["truth"] == "approach" and r["label"] == str(s) for r in timeline])]
            }
            for threshold in (.35, .5, .65, .8):
                present = valid & (presence_score >= threshold)
                direction = np.where(present, side, 0)
                detected = present & (distance <= 20)
                name = f"{model}-{threshold}"
                summary.setdefault(name, {})[cohort] = summarize(timeline, valid, direction, detected, distance)
            hybrid_alerts = {
                "replace_side": base_alerts[cohort],
                "both_presence": base_alerts[cohort] | (base_presence[cohort] & (presence_score >= .5)),
            }
            for variant, detected in hybrid_alerts.items():
                name = f"{model}-{variant}"
                direction = np.where(detected, side, 0)
                summary.setdefault(name, {})[cohort] = summarize(timeline, valid, direction, detected, distance)
            for i in np.flatnonzero(valid):
                predictions.append(dict(model=model, cohort=cohort, run=timeline[i]["run"],
                                        frame=timeline[i]["frame"], presence_score=float(presence_score[i]),
                                        right_minus_left=float(signed[i]), distance_m=float(distance[i])))
        x = normalize(np.vstack((a[train], b[train & mirrored])), mode)


# Check whether adding observed negative frames rescues detection. Hold whole
# physical corners across Magione recordings and whole Imola recordings out.
mag = cohorts["magione_transfer"]
imo = cohorts["imola_negative_transfer"]
mag_fit = mag[3] & np.array([r["physical_corner"] == 0
                             and r["truth"] in ("approach", "straight", "out_of_range") for r in mag[0]])
imo_fit = imo[3] & np.array([r["run"].endswith(("-01", "-02", "-03"))
                             and r["truth"] in ("straight", "out_of_range") for r in imo[0]])
held = {
    "monza_validation": np.ones(len(cohorts["monza_validation"][0]), bool),
    "magione_corner_2_and_unassigned": np.array([r["physical_corner"] != 0
                                     for r in mag[0]]),
    "imola_holdout_recordings": np.array([not r["run"].endswith(("-01", "-02", "-03")) for r in imo[0]]),
}
cohort_names = dict(monza_validation="monza_validation", magione_corner_2_and_unassigned="magione_transfer",
                    imola_holdout_recordings="imola_negative_transfer")
for name, mask in held.items():
    cohort = cohort_names[name]
    data = cohorts[cohort]
    refs = [r for r, keep in zip(data[0], mask, strict=True) if keep]
    for model in ("parent", "previous_candidate"):
        direction = baseline_sides[model][cohort]
        detected = baseline_alerts[model][cohort]
        summary[model][name] = summarize(refs, data[3][mask], direction[mask], detected[mask],
                                         references[cohort][mask])

for mode in ("contrast", "view_rms"):
    arrays = [a[train], b[train & mirrored]]
    groups = [(label, "monza", corner, event) for label, corner, event in presence_groups]
    targets = list(presence_target)
    for track, data, include in (("magione", mag, mag_fit), ("imola", imo, imo_fit)):
        local = [r for r, keep in zip(data[0], include, strict=True) if keep]
        local_groups = [(int(r["truth"] == "approach"), track, r["physical_corner"], r["truth"])
                        for r in local]
        arrays.extend((data[1][include], data[2][include]))
        groups.extend(local_groups*2)
        targets.extend([g[0] for g in local_groups]*2)
    combined = normalize(np.vstack(arrays), mode)
    with np.load(out/(mode+"-1000.npz"), allow_pickle=False) as readout:
        sc, si = readout["side_coef"], float(readout["side_intercept"])
    for alpha in (1000., 10000.):
        pc, pi = fit(combined, np.array(targets), groups, alpha)
        model = f"mixed-{mode}-{int(alpha)}"
        np.savez_compressed(out/(model+".npz"), presence_coef=pc, presence_intercept=pi,
                            side_coef=sc, side_intercept=si)
        for name, mask in held.items():
            cohort = cohort_names[name]
            timeline, a_test, b_test, valid, mirrored_test = cohorts[cohort]
            refs = [r for r, keep in zip(timeline, mask, strict=True) if keep]
            a_test, b_test = normalize(a_test[mask], mode), normalize(b_test[mask], mode)
            valid, mirrored_test = valid[mask], mirrored_test[mask]
            distance = references[cohort][mask]
            score = a_test@pc + pi
            signed = a_test@sc + si
            score[mirrored_test] = .5*(score[mirrored_test] + b_test[mirrored_test]@pc + pi)
            signed[mirrored_test] = .5*(signed[mirrored_test] - (b_test[mirrored_test]@sc + si))
            side = np.where(signed > 0, 2, 1)
            for threshold in (.35, .5, .65, .8):
                present = valid & (score >= threshold)
                direction = np.where(present, side, 0)
                detected = present & (distance <= 20)
                summary.setdefault(f"{model}-{threshold}", {})[name] = summarize(
                    refs, valid, direction, detected, distance)
            for i in np.flatnonzero(valid):
                predictions.append(dict(model=model, cohort=name, run=refs[i]["run"], frame=refs[i]["frame"],
                                        presence_score=float(score[i]), right_minus_left=float(signed[i]),
                                        distance_m=float(distance[i])))

jsonl(out/"predictions.jsonl", predictions)
write(out/"result.json", dict(summary=summary, conditional_side_hits=diagnostics, hashes=hashes,
    revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda", seed=None,
    fit="Monza existing train for direction and single-track presence; mixed presence adds Magione physical corner 0 and Imola recordings 01-03",
    mixed_fit_counts=dict(magione=int(mag_fit.sum()), imola=int(imo_fit.sum())),
    evaluation="Monza disjoint physical groups; full Magione and Imola development timelines",
    distance="frozen parent ordinal readout; no causal correction in this comparison",
    input="original and reflected three-frame four-view T4/T5 features; original-only fallback",
    prediction_sha256=digest(out/"predictions.jsonl"),
    model_sha256={p.name:digest(p) for p in sorted(out.glob("*.npz"))},
    source_sha256=digest(Path(__file__)), candidate_promoted=False,
    termination="development comparison exported"))
for name, groups in summary.items():
    print(name, {c: tuple(r[k] for k in ("left","right","joint","straight","out_of_range"))
                 for c, r in groups.items()})
