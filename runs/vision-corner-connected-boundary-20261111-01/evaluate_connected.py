"""Score connected visual edges on the full capture timelines."""
from pathlib import Path

import numpy as np

from connected import SMOOTHNESS
from evaluate_fused_corner import frame_key, read_rows
import hinge
from hinge import CUTS, choose, evaluate, evidence_summary, fit, group
from profile_replay import CALIBRATION_TRACKS, TRACKS
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
REFERENCE = Path("runs/vision-corner-edge-reference-20261111-01")
MODES = {f"connected_{value:.2f}":i for i, value in enumerate(SMOOTHNESS)}


def evaluate_track(name, hashes):
    g = group(name, hashes)
    path = OUT / f"{name}-edges.npz"
    proof = path.with_suffix(".json")
    receipt = read(proof)
    if digest(path) != receipt["edges_sha256"] or tuple(receipt["smoothness"]) != SMOOTHNESS:
        raise ValueError(f"Changed connected boundary inputs for {name}")
    for filename, sha in receipt["input_sha256"].items():
        source = Path(filename)
        if digest(source) != sha:
            raise ValueError(f"Changed replay source: {source}")
        hashes.add(source)
    hashes.update((path, proof))
    with np.load(path) as data:
        if list(zip(data["run"], data["frame"], strict=True)) != [frame_key(r) for r in g["rows"]]:
            raise ValueError(f"Edge frame alignment differs for {name}")
        x, observations = data["stations"], data["edges"]
    for label, index in MODES.items():
        g["fits"][label] = [fit(x, edges) for edges in observations[:, index]]
    detail = []
    for index, row in enumerate(g["rows"]):
        if row["truth"] not in ("approach", "straight", "out_of_range"):
            continue
        candidates = {label: g["fits"][label][index][1] for label in ("current", *MODES)}
        if row["truth"] == "approach" or any(v is not None for v in candidates.values()):
            detail.append(dict(track=name, run=row["run"], frame=row["frame"], truth=row["truth"],
                               label=row["label"], physical_corner=row["physical_corner"],
                               truth_distance_m=row["truth_distance_m"], proposals=candidates))
    return g, detail, observations


def compare_examples(group, observations, hashes):
    detail = REFERENCE / "rows.jsonl"
    source = REFERENCE / "result.json"
    if digest(detail) != read(source)["rows_sha256"]:
        raise ValueError("Prior reference audit changed")
    hashes.update((detail, source))
    saved = {(r["run"], r["frame"]):r for r in read_rows(detail) if "body_look_y" in r}
    examples = []
    for i, row in enumerate(group["rows"]):
        key = frame_key(row)
        if key not in saved:
            continue
        prior = saved[key]
        ref = np.asarray(prior["reference"], float)
        modes = {"current": np.asarray(prior["visual"], float)}
        modes.update({name: observations[i, j] for name, j in MODES.items()})
        results = {}
        for name, edges in modes.items():
            visible = np.isfinite(edges) & np.isfinite(ref)
            lateral = np.abs(edges - ref)
            results[name] = dict(
                proposal=group["fits"][name][i][1], shared_points=int(visible.sum()),
                lateral_mae_m=float(lateral[visible].mean()) if visible.any() else None,
                edges_m=[[float(v) if np.isfinite(v) else None for v in pair] for pair in edges])
        examples.append(dict(frame=row["frame"], truth_distance_m=row["truth_distance_m"],
                             stations_m=prior["stations_m"], methods=results))
    return examples


def run():
    hashes = {Path(__file__), hinge.OUT / "hinge.py", Path("scripts/score_corner_captures.py"),
              Path("scripts/evaluate_independent_corner.py"),
              Path("runs/vision-corner-onset-support-20261111-01/transfer.py")}
    groups, result, rows = {}, {}, []
    for name in TRACKS:
        g, detail, observations = evaluate_track(name, hashes)
        groups[name] = g
        rows.extend(detail)
        result[name] = dict(baseline=g["baseline"], comparisons={},
                            coverage={m:evidence_summary(g, m) for m in ("current", *MODES)})
        if name == "spa":
            result[name]["examples"] = compare_examples(g, observations, hashes)
    for method in ("current", *MODES):
        for policy in ("append", "confirm"):
            reports = {cut:{name:evaluate(g, method, policy, cut) for name, g in groups.items()}
                       for cut in (*CUTS, None)}
            for name in TRACKS:
                calibration = tuple(k for k in CALIBRATION_TRACKS if k != name)
                threshold = choose(groups, {c:{k:fold[k] for k in calibration} for c, fold in reports.items()})
                result[name]["comparisons"][method + "_" + policy] = dict(
                    calibration_circuits=calibration, cutoff=threshold, selected=reports[threshold][name],
                    sweep={str(c):reports[c][name] for c in CUTS})
    path = OUT / "rows.jsonl"
    jsonl(path, rows)
    write(OUT / "result.json", dict(tracks=result, rows_sha256=digest(path),
          parameters=dict(smoothness=SMOOTHNESS, cutoff_grid=CUTS, seed=None),
          source_sha256={str(p):digest(p) for p in sorted(hashes)},
          termination="connected_boundary_comparison_complete"))
    for name, data in result.items():
        print(name, {k:(v["cutoff"], v["selected"]["metrics"]["candidate"])
                     for k, v in data["comparisons"].items()}, flush=True)


if __name__ == "__main__":
    run()
