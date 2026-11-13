"""Fit an onset shared by both road edges and score it on excluded circuits."""
from pathlib import Path

import numpy as np

from evaluate_fused_corner import frame_key, load_frozen, load_score
from profile_replay import CALIBRATION_TRACKS, TRACKS, sources
from train_vision_teacher_line import digest, read, write
from transfer import compare_tracks, score

OUT = Path(__file__).parent
ALERTS = ("left", "right")
CUTS = (.02, .04, .06, .08, .10, .15, .20, .30, .40, .60, .80)
MIN_CURVATURE = .002


def squared_error(x, y, bend=None):
    columns = [np.ones(len(x)), x]
    if bend is not None:
        columns.append(bend)
    design = np.column_stack(columns)
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.square(y - design @ coeff).mean()), coeff


def fit(stations, edges):
    valid = np.isfinite(edges).all(axis=1)
    x, y = stations[valid], edges[valid]
    if len(x) < 6 or x[0] > 6. or np.max(np.diff(x)) > 4.:
        return False, None
    width = y[:, 1] - y[:, 0]
    if np.any((width <= 3.) | (width >= 20.)):
        return False, None
    null = min(squared_error(x, y)[0], squared_error(x, y, x ** 2)[0])
    best = None
    for onset in range(4, 19):
        if (x <= onset).sum() < 2 or (x > onset).sum() < 3:
            continue
        if x[x <= onset][-1] < onset - 2. or x[x > onset][0] > onset + 2.:
            continue
        error, coefficients = squared_error(x, y, np.maximum(0., x - onset) ** 2)
        slopes = coefficients[1]
        curvature = 2. * coefficients[2] / (1. + slopes ** 2) ** 1.5
        if error >= null or min(abs(curvature)) < MIN_CURVATURE or np.prod(curvature) <= 0:
            continue
        gain = (np.sqrt(null) - np.sqrt(error)) / (.20 + np.sqrt(error))
        candidate = float(gain), (1 if curvature[0] > 0 else 2), float(onset)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return True, best


def group(track, hashes):
    final, base, protocol_path = sources(track)
    rules_path = Path(load_frozen(protocol_path)["base_protocol"])
    rules = read(rules_path)["comparison"]
    receipt_path = OUT / f"{track}-sections.json"
    receipt = read(receipt_path)
    path = OUT / f"{track}-sections.npz"
    if digest(path) != receipt["sections_sha256"]:
        raise ValueError(f"Changed edge arrays for {track}")
    final_path, base_path = final / "result.json", base / "result.json"
    proof, base_proof = read(final_path), read(base_path)
    original = load_score(base / "parent/score", base_proof["score_receipt_sha256"]["parent"])
    rows = load_score(final / "score", proof["score_receipt_sha256"])
    hashes.update((receipt_path, path, protocol_path, rules_path, final_path, base_path))
    with np.load(path) as raw:
        if list(zip(raw["run"], raw["frame"], strict=True)) != [frame_key(r) for r in rows]:
            raise ValueError(f"Edge frame order changed for {track}")
        stations, values = raw["stations"], raw["values"]
    baseline = compare_tracks(original, rows, rules)
    assert baseline["metrics"] == proof["comparison"]["metrics"]
    fits = {name:[fit(stations, edge) for edge in values[:, i]]
            for i, name in enumerate(("current", "causal"))}
    return dict(rows=rows, original=original, rules=rules, baseline=baseline, fits=fits)


def recoverable(row):
    return row["state"] == "straight" or (
        row["state"] == "abstain" and row["abstain_reason"] == "presence_rejected")


def apply(row, evidence, threshold, mode):
    if threshold is None:
        return row
    observable, candidate = evidence
    strong = candidate is not None and candidate[0] >= threshold
    state = row["state"]
    if strong and recoverable(row):
        return dict(score(row, candidate[1:]), abstain_reason=None)
    if mode == "confirm" and observable and state in ALERTS:
        if strong and ALERTS[candidate[1] - 1] == state:
            return row
        return dict(row, state="abstain", abstain_reason="boundary_unconfirmed",
                    predicted_direction=None, predicted_distance_m=None, direction_hit=False,
                    joint_hit=False, signed_distance_error_m=None, absolute_distance_error_m=None)
    return row


def evaluate(group, source, mode, cut):
    rows = [apply(row, edge, cut, mode)
            for row, edge in zip(group["rows"], group["fits"][source], strict=True)]
    return compare_tracks(group["original"], rows, group["rules"])


def choose(groups, reports):
    feasible = []
    for cutoff, reviews in reports.items():
        totals = np.zeros(4, dtype=int)
        for key, report in reviews.items():
            base = groups[key]["baseline"]
            metrics = report["metrics"]["candidate"]
            previous = base["metrics"]["candidate"]
            if (metrics["outside"] > max(previous["outside"], base["metrics"]["parent"]["outside"])
                    or any(metrics[n] < previous[n] for n in ("left", "right", "joint"))
                    or len(report["stable_leads_m"]) < len(base["stable_leads_m"])):
                break
            totals += [metrics[n] for n in ("left", "joint", "right", "outside")]
        else:
            left, joint, right, outside = totals
            feasible.append(((left, joint, right, -outside, cutoff is None, cutoff or 0.), cutoff))
    return max(feasible)[1]


def evidence_summary(group, source):
    result = {}
    for name in ("left", "right", "straight", "too_far", "too_near", "in_corner"):
        def category(row):
            if row["truth"] == "approach":
                return ALERTS[int(row["label"]) - 1]
            if row["truth"] == "out_of_range":
                return "too_far" if row["truth_distance_m"] > 20. else "too_near"
            return row["truth"]

        rows = [(row, observable, proposal) for row, (observable, proposal) in
                zip(group["rows"], group["fits"][source], strict=True) if category(row) == name]
        missed = [p for p in rows if recoverable(p[0])]
        for key, population in ((name, rows), (name + "_recoverable", missed)):
            offers = [(r, p) for r, _, p in population if p is not None]
            correct = [(r, p) for r, p in offers if str(p[1]) == r["label"] and r["truth"] == "approach"]
            result[key] = dict(
                frames=len(population), observable=sum(ok for _, ok, _ in population),
                offers=len(offers), correct_direction=len(correct),
                within_five_m=sum(max(abs(p[2] - d) for d in r["truth_endpoints_m"]) <= 5.
                                  for r, p in correct))
    return result


def run():
    hashes = {Path(__file__), Path("scripts/score_corner_captures.py"),
              Path("scripts/evaluate_independent_corner.py"),
              Path("runs/vision-corner-onset-support-20261111-01/transfer.py")}
    groups = {name: group(name, hashes) for name in TRACKS}
    result = {name:dict(baseline=g["baseline"], comparisons={}) for name, g in groups.items()}
    for source in ("current", "causal"):
        for name, g in groups.items():
            result[name][source + "_coverage"] = evidence_summary(g, source)
        for mode in ("append", "confirm"):
            reviews = {cut:{name:evaluate(g, source, mode, cut) for name, g in groups.items()}
                       for cut in (*CUTS, None)}
            for name in TRACKS:
                calibration = tuple(k for k in CALIBRATION_TRACKS if k != name)
                cutoff = choose(groups, {c:{k:fold[k] for k in calibration} for c, fold in reviews.items()})
                result[name]["comparisons"][source + "_" + mode] = dict(
                    calibration_circuits=calibration, cutoff=cutoff, selected=reviews[cutoff][name],
                    sweep={str(c):fold[name] for c, fold in reviews.items() if c is not None})
    write(OUT / "result.json", dict(tracks=result, minimum_curvature_per_m=MIN_CURVATURE, cutoffs=CUTS,
          seed=None, source_sha256={str(p):digest(p) for p in sorted(hashes)},
          termination="boundary_hinge_replays_complete"))
    for name, data in result.items():
        print(name, {key:(v["cutoff"], v["selected"]["metrics"]["candidate"])
                     for key, v in data["comparisons"].items()}, flush=True)


if __name__ == "__main__":
    run()
