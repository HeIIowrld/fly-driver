"""Compare causal onset choices on matched frozen relative-colour boundary replays."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median, mean
from unittest.mock import patch

import numpy as np

import relative_corner_boundaries as relative
from corner_fusion_boundaries import stations, whole_shape_min
from evaluate_fused_corner import frame_key, load_frozen, load_score, read_rows
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write

RUNS = Path("runs")
RECOVERY = RUNS/"vision-corner-recovered-runner-20261111-01"
MUGELLO = RUNS/"vision-mugello-independent-evaluation-20261111-01"
INPUTS = {
    "magione": ("magione-development-protocol.json", "vision-frozen-next-corner-20261110-01/archives.json"),
    "ks_laguna_seca": ("laguna-development-protocol.json", "vision-laguna-independent-capture-20261111-01/archive-map.json"),
    "ks_zandvoort": ("zandvoort-development-protocol.json", "vision-zandvoort-independent-capture-20261111-02/archive-map.json"),
    "mugello": ("mugello-protocol.json", "vision-mugello-independent-capture-20261111-01/archive-map.json"),
}


def weaker_onset(samples, choice, cut):
    """Trace the confirmed direction backward through weaker contiguous curvature."""
    side, distance = choice
    sign = 1 if side == 1 else -1
    signed = dict(samples)
    before = distance - 1
    while before >= 4 and sign*signed.get(before, 0.) >= cut/2:
        distance, before = before, before - 1
    return distance


def supported_start(fused, distance):
    """Earliest section in the visible continuous block that contains the alert."""
    valid = np.isfinite(fused).all(axis=1)
    starts = np.flatnonzero(valid & np.r_[True, ~valid[:-1]])
    ends = np.flatnonzero(valid & np.r_[~valid[1:], True])
    for start, end in zip(starts, ends, strict=True):
        if stations[start] <= distance <= stations[end]:
            return float(stations[start])
    return None


def run(track, output):
    protocol_name, archive_name = INPUTS[track]
    protocol_path = RECOVERY/protocol_name
    protocol = load_frozen(protocol_path)
    settings_path = Path(protocol["boundary_fusion_pins"])
    settings = load_frozen(settings_path)
    cut = settings["curvature_threshold_per_m"]
    capture = RUNS/archive_name
    evaluation = MUGELLO if track == "mugello" else RECOVERY/track
    receipt_path = evaluation/"result.json"
    receipt = read(receipt_path)
    assert receipt["status"] == "evaluated" and receipt["track"] == track
    assert digest(protocol_path) == receipt["protocol_sha256"]
    labels = load_score(evaluation/"score", receipt["score_receipt_sha256"])
    reference = (RUNS/"vision-mugello-readout-stage-audit-20261111-01/rows.jsonl" if track == "mugello"
                 else RUNS/"vision-corner-relative-chroma-gates-20261111-01"/track/"all-gates.jsonl")
    old = read_rows(reference)

    snapshot = None
    def take_snapshot(fused, aligned):
        nonlocal snapshot
        samples = whole_shape_min(fused, aligned)
        snapshot = fused, samples
        return samples

    rows = []
    with patch.object(relative, "whole_shape_min", take_snapshot):
        geometry = relative.read_boundaries(capture, receipt["camera_bundle_sha256_by_run"], track,
                                            Path(protocol["camera_config"]), settings)
        for t, prior, (key, choice, travel) in zip(labels, old, geometry, strict=True):
            assert frame_key(t) == frame_key(prior) == key
            reference_side = prior["boundary_side"] if track == "mugello" else (
                prior["choice"][0] if prior["choice"] else None)
            reference_distance = prior["boundary_distance_m"] if track == "mugello" else (
                prior["choice"][1] if prior["choice"] else None)
            assert (choice if choice else (None, None)) == (reference_side, reference_distance)
            row = {name:t[name] for name in ("run","frame","generation","received_s","truth","label",
                                            "physical_corner","truth_distance_m","truth_endpoints_m")}
            row.update(side=choice[0] if choice else None, travel_m=travel)
            if snapshot is not None:
                fused, samples = snapshot
                row["available_sections_m"] = stations[np.isfinite(fused).all(axis=1)].tolist()
                row["curvature_samples"] = samples
                if choice and choice[0]:
                    row["strong_m"] = choice[1]
                    row["weak_m"] = weaker_onset(samples, choice, cut)
                    row["support_start_m"] = supported_start(fused, choice[1])
                snapshot = None
            rows.append(row)

    metrics = {}
    for label in ("1", "2"):
        positives = [r for r in rows if r["truth"] == "approach" and r["label"] == label]
        selected = [r for r in positives if r["side"] == int(label)]
        pairs = [(a,b) for a,b in zip(selected,selected[1:]) if adjacent(a,b)
                 and a["physical_corner"] == b["physical_corner"]]
        side = metrics[label] = dict(approaches=len(positives), direction_correct=len(selected),
                geometry=Counter(str(r["side"]) for r in positives),
                consecutive_pairs=len(pairs),
                no_sections=sum(not r.get("available_sections_m") for r in positives))
        for field in ("strong_m", "weak_m", "support_start_m"):
            valid = [r for r in selected if r.get(field) is not None]
            errors = [r[field]-r["truth_distance_m"] for r in valid]
            limits = [max(abs(r[field]-v) for v in r["truth_endpoints_m"]) for r in valid]
            deltas = [(b[field]-a[field], b["truth_distance_m"]-a["truth_distance_m"])
                      for a,b in pairs if a.get(field) is not None and b.get(field) is not None]
            side[field] = dict(n=len(valid), within_five_m=sum(v<=5 for v in limits),
                              median_error_m=median(errors) if errors else None,
                              distance_change_mae_m=mean(abs(p-t) for p,t in deltas) if deltas else None,
                              receding=sum(p>0 and t<0 for p,t in deltas), pairs=len(deltas))
    metrics["negative_candidates"] = {kind:sum(r["truth"]==kind and bool(r["side"]) for r in rows)
                                    for kind in ("straight","out_of_range")}
    output.mkdir(exist_ok=False)
    jsonl(output/"rows.jsonl", rows)
    inputs = (protocol_path, settings_path, capture, receipt_path, reference, Path(__file__))
    write(output/"result.json", dict(track=track, metrics=metrics, revision=receipt["revision"], seed=None,
          source_sha256={str(p):digest(p) for p in inputs}, rows_sha256=digest(output/"rows.jsonl"),
          termination="matched_frozen_geometry_replay"))
    print(track, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=INPUTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.track, args.output)
