"""Check whether same-direction shape and hinge evidence can share a distance estimate."""
from pathlib import Path

from evaluate_fused_corner import frame_key, read_rows
from hinge import recoverable, group
from profile_replay import TRACKS
from train_vision_teacher_line import digest, read, write
from transfer import compare_tracks, score

OUT = Path(__file__).parent
LEARNED = Path("runs/vision-corner-learned-edge-20261111-01")
CUT = .85


def verified_rows(folder):
    path = folder / "rows.jsonl"
    result = folder / "result.json"
    if digest(path) != read(result)["rows_sha256"]:
        raise ValueError("Prediction rows changed")
    return read_rows(path)


def run():
    shape = {(r["track"], *frame_key(r)):r for r in verified_rows(OUT) if r["mode"] == "aligned"}
    hinge = {(r["track"], *frame_key(r)):r["proposal"] for r in verified_rows(LEARNED)}
    results = {}
    for track in TRACKS:
        g = group(track, set())
        candidate, changed = [], []
        for r in g["rows"]:
            key = track, *frame_key(r)
            close, proposal = shape.get(key), hinge.get(key)
            if not recoverable(r) or close is None or not close["usable"] or proposal is None:
                candidate.append(r)
                continue
            side = proposal[1]
            p = close["near_scores"]
            if p[side - 1] >= CUT and p[side - 1] > p[2 - side]:
                item = dict(score(r, (side, proposal[2])), abstain_reason=None)
                candidate.append(item)
                changed.append(dict(frame=r["frame"], truth=r["truth"], label=r["label"],
                                    distance_m=proposal[2], joint_hit=item["joint_hit"]))
            else:
                candidate.append(r)
        result = compare_tracks(g["original"], candidate, g["rules"])
        results[track] = dict(comparison=result, changed=changed)
        print(track, result["metrics"]["candidate"], flush=True)
    inputs = (OUT/"result.json", OUT/"rows.jsonl", LEARNED/"result.json", LEARNED/"rows.jsonl", Path(__file__))
    write(OUT / "joint-evidence-diagnostic.json", dict(tracks=results, cutoff=CUT,
          source_sha256={str(p):digest(p) for p in inputs},
          termination="posthoc_shape_presence_and_hinge_distance_comparison_complete"))


if __name__ == "__main__":
    run()
