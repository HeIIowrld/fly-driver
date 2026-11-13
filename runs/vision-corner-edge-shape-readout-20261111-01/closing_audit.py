"""Measure distance changes on eligible visual approach pairs before warning thresholds."""
from pathlib import Path

import numpy as np

from evaluate_fused_corner import frame_key, read_rows
from hinge import group
from profile_replay import TRACKS
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent
MOTION = Path("runs/vision-corner-edge-agreement-20261111-01")


def run():
    path = OUT / "rows.jsonl"
    source = OUT / "result.json"
    if digest(path) != read(source)["rows_sha256"]:
        raise ValueError("Shape predictions changed")
    keyed = {(r["track"], r["mode"], *frame_key(r)): r for r in read_rows(path)}
    report = {}
    for track in TRACKS:
        g = group(track, set())
        motion_path = MOTION / f"{track}-motion.npz"
        if digest(motion_path) != read(motion_path.with_suffix(".json"))["motion_sha256"]:
            raise ValueError("Motion input changed")
        with np.load(motion_path) as array:
            travel = array["motion"]
        tracks = {}
        for mode in ("current", "aligned"):
            changes, true_changes, propagation_changes = [], [], []
            for a, b, move in zip(g["rows"][:-1], g["rows"][1:], travel[1:], strict=True):
                if (a["truth"] != "approach" or b["truth"] != "approach"
                        or a["physical_corner"] != b["physical_corner"] or a["label"] != b["label"]
                        or not adjacent(a, b) or b["truth_distance_m"] >= a["truth_distance_m"]):
                    continue
                first = keyed[track, mode, *frame_key(a)]
                second = keyed[track, mode, *frame_key(b)]
                if not first["usable"] or not second["usable"] or not 0. <= move <= 12.:
                    continue
                side = int(b["label"]) - 1
                d1, d2 = first["distance_m"][side], second["distance_m"][side]
                changes.append(d2 - d1)
                true_changes.append(b["truth_distance_m"] - a["truth_distance_m"])
                propagation_changes.append(-float(move))
            predicted, truth, propagation = map(np.asarray, (changes, true_changes, propagation_changes))
            tracks[mode] = dict(pairs=len(predicted), distance_change_mae_m=float(abs(predicted - truth).mean()) if len(predicted) else None,
                                receding=int((predicted > 0).sum()),
                                motion_only_change_mae_m=float(abs(propagation - truth).mean()) if len(predicted) else None)
        report[track] = tracks
        print(track, tracks, flush=True)
    write(OUT / "closing-pairs.json", dict(tracks=report, results_sha256=digest(source),
          proposal_rows_sha256=digest(path), audit_code_sha256=digest(Path(__file__)),
          termination="causal_shape_closing_pair_diagnostic_complete"))


if __name__ == "__main__":
    run()
