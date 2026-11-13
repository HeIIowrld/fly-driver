"""Fit T4/T5 direction readouts using explicit development recording roles."""
from collections import Counter
from pathlib import Path
import json

import numpy as np
from scipy.linalg import solve

from infer_frozen_next_corner import KEYS, contrast, load_readout
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
root = out.parent
base = root / "vision-frozen-next-corner-20261110-01"
pins = read(base / "pins.json")
predict, threshold = load_readout(pins)
inputs = [base/"pins.json", Path(__file__)]
labels, direct, mirror, train, cohorts, mirror_valid = [], [], [], [], [], []

def load(path):
    return list(map(json.loads, path.read_text().splitlines()))

def checked(path, receipt, key):
    proof = read(receipt)
    assert digest(path) == proof[key], path
    inputs.extend((path, receipt))

monza = root / "vision-monza-two-lap-presence-20261111-01"
checked(monza/"rows.jsonl", monza/"flyvis-result.json", "rows_sha256")
checked(monza/"features.npz", monza/"flyvis-result.json", "features_sha256")
with np.load(monza/"features.npz", allow_pickle=False) as z:
    a, b, valid = (z[k] for k in ("direct","reflected","reflected_available"))
    for i, row in enumerate(load(monza/"rows.jsonl")):
        labels.append(row)
        direct.append(a[i])
        mirror.append(b[i])
        mirror_valid.append(bool(valid[i]))
        train.append(row["split"] == "train")
        cohorts.append("monza_" + row["split"])

archive = base/"archive-parity"
rows = load(archive/"index.jsonl")
actual = root/"vision-magione-edge-midpoint-inventory-20261110-01"
checked(actual/"rows.jsonl", actual/"result.json", "rows_sha256")
truth = {(r["run"],r["frame"]):r for r in load(actual/"rows.jsonl")}
branches = []
for name in ("original","reflected"):
    p = archive/name
    checked(p/"features.npz", p/"flyvis-result.json", "features_sha256")
    assert read(p/"flyvis-result.json")["rows_sha256"] == digest(archive/"index.jsonl")
    with np.load(p/"features.npz", allow_pickle=False) as z:
        assert z["available"].all()
        branches.append(z["features"].copy())
inputs.append(archive/"index.jsonl")
for i, row in enumerate(rows):
    row = truth[row["run"],row["frame"]]
    fit = row["run"] in ("magione-student-parent-02","magione-user-recovered-01")
    labels.append(row)
    direct.append(branches[0][i])
    mirror.append(branches[1][i])
    mirror_valid.append(True)
    train.append(fit)
    cohorts.append("magione_fit" if fit else "magione_recording_diagnostic")

imola = root/"vision-imola-hard-negative-presence-20261110-01"
checked(imola/"imola-rows.jsonl", imola/"result.json", "imola_rows_sha256")
checked(imola/"imola-features.npz", imola/"result.json", "feature_sha256")
with np.load(imola/"imola-features.npz", allow_pickle=False) as z:
    a,b = z["features"].transpose(1,0,2)
    for i,row in enumerate(load(imola/"imola-rows.jsonl")):
        fit = row["run"] == "02"
        labels.append(dict(row,direction=0,distance=None))
        direct.append(a[i])
        mirror.append(b[i])
        mirror_valid.append(True)
        train.append(fit)
        cohorts.append("imola_fit_negative" if fit else "imola_adjacent_negative")

a,b = np.array(direct),np.array(mirror)
mirror_valid = np.array(mirror_valid)
scores, distances, presence = np.empty((len(a),3)), np.empty(len(a)), np.empty(len(a))
for i in range(len(a)):
    result = predict(a[i:i+1], b[i:i+1] if mirror_valid[i] else None, return_scores=True)
    scores[i], distances[i], presence[i] = result[0][0], result[1][0], result[2][0]
gate = (presence >= threshold) & (distances <= 20)
mask = np.array(train)
direction = np.array([r["direction"] for r in labels])
distance = np.array([r["distance"] if r["distance"] is not None else np.inf for r in labels])
target = np.where((direction != 0) & (distance <= 30),direction,0)
x = np.vstack((contrast(a[mask]),contrast(b[mask & mirror_valid])))
y = np.r_[target[mask],np.array([0,2,1])[target[mask & mirror_valid]]]
weight = len(y)/(3*np.bincount(y,minlength=3)[y])
mu = np.average(x,axis=0,weights=weight)
scale = np.sqrt(np.average((x-mu)**2,axis=0,weights=weight)).clip(.04)
offset = np.average(np.eye(3)[y],axis=0,weights=weight)
design = (x-mu)/scale
gram = design.T @ (weight[:,None]*design)
rhs = design.T @ (weight[:,None]*(np.eye(3)[y]-offset))
qa,qb = (contrast(a)-mu)/scale,(contrast(b)-mu)/scale
with np.load(pins["models"]["direction"]["path"],allow_pickle=False) as z:
    bias = float(z["bias"])

def metrics(pred):
    output = {}
    for cohort in sorted(set(cohorts)):
        selected = np.array(cohorts) == cohort
        near = selected & (direction != 0) & (distance >= 4) & (distance <= 20)
        hit = near & (pred == direction) & (pred != 0)
        output[cohort] = dict(frames=int(selected.sum()),
            near_left=int((near & (direction==1)).sum()), near_right=int((near & (direction==2)).sum()),
            hit_left=int((hit & (direction==1)).sum()), hit_right=int((hit & (direction==2)).sum()),
            joint=int((hit & (np.abs(distances-distance)<=5)).sum()),
            outside=int((selected & ~near & (pred!=0)).sum()))
    return output

baseline = scores.argmax(1)
baseline[(scores==scores.max(1,keepdims=True)).sum(1)>1] = 0
baseline[~gate] = 0
summary = {"baseline":metrics(baseline)}
predictions = []
with np.load(pins["models"]["direction"]["path"],allow_pickle=False) as packed:
    old_mean,old_scale,old_offset,old_coefficients = (packed["cls_"+k] for k in KEYS)
old_raw = old_coefficients / old_scale[:,None]
old_intercept = old_offset - old_mean@old_raw
for alpha in (100.,1000.,10000.):
    regularized = gram.copy()
    regularized.flat[::len(regularized)+1] += alpha
    coefficients = solve(regularized,rhs,assume_a="pos")
    learned_scores = qa@coefficients+offset
    learned_scores[mirror_valid] = .5*(learned_scores[mirror_valid]
        + (qb[mirror_valid]@coefficients+offset)[:,[0,2,1]])
    learned_scores += [bias,0.,0.]
    new_raw = coefficients / scale[:,None]
    new_intercept = offset - mu@new_raw
    for gain in (0., .1, .25, .5, .75, 1.):
        name = "ridge-" + str(int(alpha)) + "-gain-" + str(gain)
        candidate_scores = (1-gain)*scores + gain*learned_scores
        pred = candidate_scores.argmax(1)
        pred[(candidate_scores==candidate_scores.max(1,keepdims=True)).sum(1)>1] = 0
        pred[~gate] = 0
        summary[name] = metrics(pred)
        mixed = (np.zeros_like(mu),np.ones_like(scale),
                 (1-gain)*old_intercept+gain*new_intercept,
                 (1-gain)*old_raw+gain*new_raw)
        np.savez_compressed(out/(name+".npz"),
            **{"cls_"+k:v for k,v in zip(KEYS,mixed,strict=True)},bias=bias)
        for r, cohort, side in zip(labels,cohorts,pred,strict=True):
            predictions.append(dict(model=name,run=r["run"],frame=r["frame"],cohort=cohort,
                                    predicted_direction=int(side)))
jsonl(out/"rows.jsonl",predictions)
write(out/"result.json",dict(summary=summary, hashes={str(p):digest(p) for p in inputs},
    rows_sha256=digest(out/"rows.jsonl"), seed=None, revision=pins["revision"],
    roles=dict(monza="existing train versus existing physical-corner validation",
               magione="parent02 and recovered fit; candidate01 and alternative diagnostic; shared physical corners",
               imola="recording02 negative fit; recording03 adjacent diagnostic"),
    scope="previously consumed development data only; complete original-only fallback retained",
    target="direction of upcoming corner within 30m, otherwise background",
    actor_inputs="fixed original and reflected T4/T5 features",
    learned_component="linear direction readout; frozen presence and ordinal distance",
    candidate_promoted=False,termination="development readout comparison exported"))
for name, groups in summary.items():
    print(name, {cohort: (r["hit_left"],r["hit_right"],r["joint"],r["outside"])
                 for cohort,r in groups.items() if cohort in (
                     "magione_recording_diagnostic","imola_adjacent_negative","monza_validation")})
