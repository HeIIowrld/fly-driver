"""Compare separately fitted road edges before and after relative-motion agreement."""
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_fused_corner import frame_key
import hinge
from hinge import CUTS, MIN_CURVATURE, apply, choose, group, squared_error
from profile_replay import CALIBRATION_TRACKS, TRACKS
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import compare_tracks, score

OUT = Path(__file__).parent
BOUNDARY = Path("runs/vision-corner-boundary-hinge-20261111-01")
EDGE_TOLERANCE_M = 3.
TEMPORAL_TOLERANCE_M = 3.


def edge_onset(stations, positions):
    mask = np.isfinite(positions)
    x, y = stations[mask], positions[mask]
    if len(x) < 6 or x[0] > 6. or max(np.diff(x)) > 4.:
        return None
    y = y[:, None]
    null = min(squared_error(x, y)[0], squared_error(x, y, x * x)[0])
    best = None
    for onset in range(4, 19):
        before = x[x <= onset]
        after = x[x > onset]
        if len(before) < 2 or len(after) < 3 or before[-1] < onset - 2. or after[0] > onset + 2.:
            continue
        error, coeff = squared_error(x, y, np.maximum(0., x - onset) ** 2)
        curvature = float(2. * coeff[2, 0] / (1. + coeff[1, 0] ** 2) ** 1.5)
        if error >= null or abs(curvature) < MIN_CURVATURE:
            continue
        gain = (np.sqrt(null) - np.sqrt(error)) / (.2 + np.sqrt(error))
        candidate = float(gain), 1 if curvature > 0. else 2, float(onset)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def edge_consensus(first, second):
    if first is None or second is None or first[1] != second[1]:
        return None
    if abs(first[2] - second[2]) > EDGE_TOLERANCE_M:
        return None
    return min(first[0], second[0]), first[1], .5 * (first[2] + second[2])


def temporal_consensus(current, prior, travel):
    if current is None or prior is None or current[1] != prior[1] or not 0. <= travel <= 12.:
        return None
    predicted = prior[2] - travel
    if abs(current[2] - predicted) > TEMPORAL_TOLERANCE_M:
        return None
    distance = float(max(4., .5 * (current[2] + predicted)))
    return min(current[0], prior[0]), current[1], distance


def tracks(group, stations, edges, moves):
    previous_row = None
    previous = None
    for row, pair, travel in zip(group["rows"], edges, moves, strict=True):
        individual = tuple(edge_onset(stations, side) for side in pair.T)
        current = edge_consensus(*individual)
        if previous_row is not None and adjacent(previous_row, row):
            confirmed = temporal_consensus(current, previous, travel)
        else:
            confirmed = None
        yield individual, current, confirmed
        previous_row, previous = row, current


def load_inputs(name, hashes):
    source = BOUNDARY / f"{name}-sections.npz"
    proof = source.with_suffix(".json")
    # The earlier paired-edge run is the verified source of current observations.
    if digest(source) != read(proof)["sections_sha256"]:
        raise ValueError(f"Boundary arrays changed: {name}")
    if hinge.OUT.resolve() != BOUNDARY.resolve():
        raise ValueError("Paired-boundary evaluator points to a different receipt")
    g = group(name, hashes)
    movement = OUT / f"{name}-motion.npz"
    motion_proof = movement.with_suffix(".json")
    receipt = read(motion_proof)
    if digest(movement) != receipt["motion_sha256"]:
        raise ValueError(f"Relative motion changed: {name}")
    for filename, sha in receipt["input_sha256"].items():
        path = Path(filename)
        if digest(path) != sha:
            raise ValueError(f"Relative motion input changed: {path}")
        hashes.add(path)
    hashes.update((source, proof, movement, motion_proof))
    with np.load(source) as obs, np.load(movement) as motion:
        if list(zip(motion["run"], motion["frame"], strict=True)) != [frame_key(r) for r in g["rows"]]:
            raise ValueError(f"Relative motion alignment differs: {name}")
        return g, list(tracks(g, obs["stations"], obs["values"][:, 0], motion["motion"]))


def substitute(row, candidate, cut, method):
    if method in ("edge_pair", "travel_pair"):
        return apply(row, (candidate is not None, candidate), cut, "append")
    if cut is None or candidate is None or candidate[0] < cut:
        return row
    side, distance = candidate[1:]
    if row["state"] != ("left" if side == 1 else "right"):
        return row
    return score(row, (side, distance))


def evaluate(g, proposals, cut, method):
    index = 1 if method == "edge_pair" else 2
    rows = [substitute(row, proposal[index], cut, method)
            for row, proposal in zip(g["rows"], proposals, strict=True)]
    return compare_tracks(g["original"], rows, g["rules"])


def choose_distance(groups, reports):
    candidates = []
    for cutoff, by_track in reports.items():
        improvements = []
        for name, review in by_track.items():
            baseline = groups[name]["baseline"]
            old, new = (r["distances"]["candidate"] for r in (baseline, review))
            if (review["metrics"] != baseline["metrics"]
                    or len(review["stable_leads_m"]) < len(baseline["stable_leads_m"])
                    or review["shared_pairs"] != baseline["shared_pairs"]
                    or new["receding"] > old["receding"]):
                break
            if old["mae_m"] is not None:
                if new["mae_m"] > old["mae_m"] + 1e-9:
                    break
                improvements.append(old["mae_m"] - new["mae_m"])
        else:
            candidates.append(((sum(improvements), cutoff is None, cutoff or 0.), cutoff))
    return max(candidates)[1]


def diagnostics(g, proposals):
    totals = {}
    for row, (edges, pair, temporal), (_, shared) in zip(g["rows"], proposals, g["fits"]["current"], strict=True):
        truth = row["truth"]
        if truth == "approach":
            truth = "left" if row["label"] == "1" else "right"
        elif truth == "out_of_range":
            truth = "too_far" if row["truth_distance_m"] > 20. else "too_near"
        metrics = totals.setdefault(truth, Counter())
        metrics["frames"] += 1
        metrics["both_edge_onsets"] += all(e is not None for e in edges)
        for name, prediction in (("shared", shared), ("edge_pair", pair), ("travel_pair", temporal)):
            if prediction is None:
                continue
            metrics[name + "_offers"] += 1
            if truth in ("left", "right") and str(prediction[1]) == row["label"]:
                metrics[name + "_direction"] += 1
                metrics[name + "_joint"] += max(abs(prediction[2] - d) for d in row["truth_endpoints_m"]) <= 5.
    return totals


def run():
    hashes = {Path(__file__), BOUNDARY / "hinge.py", Path("scripts/score_corner_captures.py"),
              Path("scripts/evaluate_independent_corner.py"),
              Path("runs/vision-corner-onset-support-20261111-01/transfer.py")}
    groups, proposals = {}, {}
    for track in TRACKS:
        groups[track], proposals[track] = load_inputs(track, hashes)
        print("loaded", track, len(proposals[track]), flush=True)
    results = {k:dict(baseline=g["baseline"], coverage=diagnostics(g, proposals[k]), methods={})
               for k, g in groups.items()}
    for method in ("edge_pair", "travel_pair", "distance_only"):
        evaluations = {cut:{k:evaluate(g, proposals[k], cut, method) for k, g in groups.items()}
                       for cut in (*CUTS, None)}
        for k in TRACKS:
            train = tuple(t for t in CALIBRATION_TRACKS if t != k)
            select = choose_distance if method == "distance_only" else choose
            threshold = select(groups, {cut:{t:fold[t] for t in train} for cut, fold in evaluations.items()})
            results[k]["methods"][method] = dict(calibration_circuits=train, threshold=threshold,
                                                 selected=evaluations[threshold][k],
                                                 sweep={str(c):evaluations[c][k] for c in CUTS})
    rows = []
    for k, g in groups.items():
        for row, (edges, pair, temporal), (_, shared) in zip(g["rows"], proposals[k], g["fits"]["current"],
                                                           strict=True):
            if row["truth"] == "approach" or any(p is not None for p in (shared, pair, temporal)):
                rows.append(dict(track=k, run=row["run"], frame=row["frame"], truth=row["truth"],
                                 label=row["label"], truth_distance_m=row["truth_distance_m"],
                                 physical_corner=row["physical_corner"], baseline_state=row["state"],
                                 shared=shared, edges=edges, pair=pair, temporal=temporal))
    path = OUT / "rows.jsonl"
    jsonl(path, rows)
    write(OUT / "result.json", dict(tracks=results, rows_sha256=digest(path),
          parameters=dict(edge_tolerance_m=EDGE_TOLERANCE_M, temporal_tolerance_m=TEMPORAL_TOLERANCE_M,
                          cutoff_grid=CUTS, seed=None),
          source_sha256={str(p):digest(p) for p in sorted(hashes)},
          termination="causal_edge_agreement_replays_complete"))
    for k, data in results.items():
        print(k, {m:(r["threshold"], r["selected"]["metrics"]["candidate"])
                  for m, r in data["methods"].items()}, flush=True)


if __name__ == "__main__":
    run()
