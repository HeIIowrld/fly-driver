"""Measure whether independent signed visual evidence supports replacing a contradictory alert."""
from collections import Counter
from pathlib import Path

import numpy as np

from audit import OUT, PROTOCOL, RELATIVE, counts, score_rows
from evaluate_fused_corner import frame_key, load_frozen, read_rows
from infer_frozen_next_corner import contrast, load_features
from train_vision_teacher_line import digest, read, write

ROOT = Path("runs")
BASES = {
    "ks_zandvoort": ROOT/"vision-zandvoort-independent-evaluation-20261111-01"/"fusion/base",
    "magione": ROOT/"vision-corner-independent-runner-rehearsal-20261111-01"/"replay",
}


def signed_scores(base, model):
    features = base/"features"
    index = features/"index.jsonl"
    records = read_rows(index)
    sha = digest(index)
    a, av, _ = load_features(features/"original/features.npz", sha, len(records))
    b, bv, _ = load_features(features/"reflected/features.npz", sha, len(records))
    valid = av & bv
    with np.load(model, allow_pickle=False) as packed:
        weight = packed["side_coef"]
        offset = float(packed["side_intercept"])
    signed = .5*((contrast(a[valid])@weight+offset)-(contrast(b[valid])@weight+offset))
    return {frame_key(r):float(value) for r,value in
            zip((r for r, keep in zip(records,valid,strict=True) if keep), signed, strict=True)}


def override(rows, gates, signed, *, require_motion, require_signed, margin):
    for row, gate in zip(rows, gates, strict=True):
        key = frame_key(row)
        assert key == frame_key(gate)
        choice = gate["choice"]
        if (row["alert"] and choice and choice[0] and row["direction"] != choice[0]
                and gate["presence"] and gate["ordinal_range"] and (gate["movement"] or not require_motion)):
            agreement = signed.get(key, 0.) * (1 if choice[0] == 2 else -1)
            if not require_signed or agreement > margin:
                score, distance = gate["visual"]
                yield dict(row, direction=choice[0], distance_m=distance, presence_score=score)
                continue
        yield row


protocol = load_frozen(PROTOCOL)
settings = load_frozen(Path(protocol["development_pins"]))
model = Path(settings["visual_presence_model"])
inputs = [Path(__file__), PROTOCOL, model, Path(__file__).with_name("audit.py")]
results = {}
for track, base in BASES.items():
    result_path = OUT/track/"result.json"
    proof = read(result_path)
    gate_path = OUT/track/"all-gates.jsonl"
    assert digest(gate_path) == proof["all_gates_sha256"]
    gates = read_rows(gate_path)
    reference_path = RELATIVE/track/"forecasts/rows.jsonl"
    labels_path = RELATIVE/track/"score/rows.jsonl"
    assert digest(reference_path) == read(reference_path.with_name("result.json"))["rows_sha256"]
    assert digest(labels_path) == read(RELATIVE/track/"result.json")["score_sha256"]
    reference, labels = read_rows(reference_path), read_rows(labels_path)
    signs = signed_scores(base, model)
    baseline_scores = score_rows(labels, reference)
    stats = {}
    for motion in (True, False):
        for signed, margin in ((False, 0.), (True, 0.), (True, .1), (True, .25)):
            predictions = list(override(reference, gates, signs, require_motion=motion, require_signed=signed, margin=margin))
            changes = Counter(label["truth"] for label,a,b in zip(labels,reference,predictions,strict=True)
                              if a["direction"] != b["direction"])
            scores = score_rows(labels, predictions)
            paired = {}
            for which, report in (("baseline", baseline_scores), ("candidate", scores)):
                paired[which] = {(run,r["first_frame"],r["last_frame"]):r
                                 for run, item in report.items() for r in item["distance_change"]["rows"]}
            common = sorted(paired["baseline"].keys() & paired["candidate"].keys())
            distance_change = {}
            for which, rows in paired.items():
                values = [rows[key] for key in common]
                distance_change[which] = dict(
                    mae_m=sum(abs(r["delta_error_m"]) for r in values)/len(values) if values else None,
                    receding=sum(r["truth_delta_m"] < 0 < r["forecast_delta_m"] for r in values))
            name = f"motion_{motion}_signed_{signed}_margin_{margin}"
            stats[name] = dict(counts(scores), directions_replaced=dict(changes), shared_pairs=len(common), distance_change=distance_change)
    results[track] = dict(baseline=counts(score_rows(labels,reference)), options=stats)
    inputs.extend((result_path, gate_path, reference_path, labels_path, base/"features/index.jsonl",
                   base/"features/original/features.npz", base/"features/reflected/features.npz"))
write(OUT/"override.json", dict(summary=results, candidate_promoted=False,
     scope="development-only counterfactual direction replacement; no independent retest",
     inputs={str(p):digest(p) for p in inputs}))
for track, stats in results.items():
    print(track, stats["options"]["motion_False_signed_False_margin_0.0"])
