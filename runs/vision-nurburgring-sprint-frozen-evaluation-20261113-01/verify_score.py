"""Recompute the independent score and bind the original camera, labels and outputs."""
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
BASE = RUNS / "vision-corner-left-risk-reallocation-20261113-01"
FILTER = RUNS / "vision-corner-added-left-phase-veto-20261113-01"
sys.path[:0] = [str(BASE),str(FILTER)]
from veto_added_left import old, keys, digests, counts, RISK, score

CAPTURE = RUNS / "vision-nurburgring-sprint-capture-20261113-01"
FROZEN = RUNS / "vision-corner-phase-veto-frontier-20261113-01/frozen.json"
TRACK = "ks_nurburgring"



def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def main():
    archive = old.read(CAPTURE/"result.json")
    capture = old.read(OUT/"capture-verification.json")
    result_path = OUT/"result.json"
    result = old.read(result_path)
    frozen = old.read(FROZEN)
    labels = OUT/"labels.jsonl"
    forecasts = OUT/"forecasts.jsonl"
    if result["labels_sha256"] != old.digest(labels) or result["forecasts_sha256"] != old.digest(forecasts):
        raise ValueError("Forecast or label checksum differs")
    if archive["inputs"][str(FROZEN)] != old.digest(FROZEN):
        raise ValueError("Capture did not use the fixed comparison settings")
    if capture["capture_sha256"] != old.digest(CAPTURE/"result.json"):
        raise ValueError("Capture provenance differs")
    predictions, targets = rows(forecasts), rows(labels)
    if keys(predictions) != keys(targets) or len(predictions) != archive["frames"]:
        raise ValueError("Complete timeline differs")
    after = score(predictions,targets)
    delta = counts(result["before"]["counts"],after["counts"],(*RISK,"hit_1","hit_2"))
    assert after == result["after"] and delta == result["delta"]
    assert result["per_recording_risk_pass"] == all(
        delta[k] <= cap for k,cap in frozen["deployment_risk_caps"].items())
    assert {side:entry["frames"] for side,entry in after["approaches"].items()} == capture["approaches_by_direction"]
    assert TRACK not in frozen["selection_recordings"]
    old.write(OUT/"verification.json",dict(
        track=TRACK,frames=len(predictions),approach_frames=sum(capture["approaches_by_direction"].values()),
        right_delta=delta["hit_2"],left_delta=delta["hit_1"],
        per_recording_risk_pass=result["per_recording_risk_pass"],sources=digests(
            (Path(__file__),FROZEN,result_path,labels,forecasts,OUT/"capture-verification.json")),
        seed=None,termination="verified"))
    print("Independent corner score verified",flush=True)


if __name__ == "__main__":
    main()
