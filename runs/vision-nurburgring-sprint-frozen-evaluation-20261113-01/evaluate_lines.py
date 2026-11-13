"""Score the already-frozen geometry teacher readout on the Sprint capture."""
from pathlib import Path
import sys

import joblib
import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
TEACHER = RUNS / "vision-optimized-line-teacher-20261113-01"
CAPTURE = RUNS / "vision-nurburgring-sprint-capture-20261113-01"
BASE = RUNS / "vision-racing-line-motion-readout-20261113-01"
sys.path[:0] = [str(TEACHER),str(BASE)]
from evaluate_optimized import evaluate, old, digests, TRACKS, TRACK_DIR
import evaluate_optimized as evaluator
from evaluate_teacher import clearance, nominal_lap, load_edges, borders_from_spline, optimize_line
from train_optimized import FROZEN
from score_archive import run as score_trajectory

TRACK = "ks_nurburgring"
LAYOUT = "layout_sprint_a"


def main():
    frozen = old.read(FROZEN)
    capture = CAPTURE/"result.json"
    if old.read(capture)["inputs"].get(str(FROZEN)) != old.digest(FROZEN):
        raise ValueError("Sprint capture does not bind the frozen line predictor")
    for path,checksum in frozen["sources"].items():
        if old.digest(Path(path)) != checksum:
            raise ValueError("Frozen dependency changed")
    model_file = TEACHER/"optimized-readouts.joblib"
    if old.digest(model_file) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint differs")
    models = joblib.load(model_file)
    if set(models) != set(frozen["models"]) or TRACK in frozen["comparisons"] + frozen["development"]:
        raise ValueError("Model or independent layout inventory differs")
    spline = TRACK_DIR / TRACK / LAYOUT / "ai/fast_lane.ai"
    edges,_ = load_edges(spline)
    optimized,center,fit = optimize_line(edges)
    borders = borders_from_spline(spline)
    path = OUT/f"{TRACK}-teacher.npz"
    np.savez_compressed(path,optimized_xy_m=optimized,center_xy_m=center)
    ref = dict(teacher_sha256=old.digest(path),fit=fit,
               optimized=nominal_lap(optimized),center=nominal_lap(center),
               optimized_clearance=clearance(optimized,borders),center_clearance=clearance(center,borders))
    replay = OUT/"archive-replay"
    archives = CAPTURE/"archive-map.json"
    evaluator.OUT = OUT
    TRACKS[TRACK] = LAYOUT
    result = evaluate(TRACK,replay,archives,frozen,models,{TRACK:ref})
    score_trajectory(archives,TRACK,LAYOUT,replay,OUT/"recorded-path-evaluation")
    old.write(OUT/"optimized-line-result.json",dict(track=TRACK,layout=LAYOUT,selected=frozen["selected_vision"],
        reference=ref,evaluation=result,seed=None,source_revision=old.revision(),
        sources=digests((FROZEN,model_file,capture,Path(__file__),spline,
                        Path(evaluate.__code__.co_filename),Path(optimize_line.__code__.co_filename))),
        termination="archive_end"))


if __name__ == "__main__":
    main()
