"""Check frozen transfer artifacts and account for every labelled frame."""
from collections import Counter
import json
from pathlib import Path

from fixed_left import EVAL, METHOD, keys, old, score
from replay_independent import load_fixed_policy

ROOT = Path(__file__).resolve().parent

def rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]

def main():
    receipt, frozen, *_ = load_fixed_policy()
    frozen_digest = old.digest(receipt)
    comparison = old.read(ROOT / "three-circuit-replay/comparison.json")
    reproduction = old.read(ROOT / "reproduction.json")
    assert comparison["freeze_sha256"] == frozen_digest == reproduction["frozen_sha256"]
    assert reproduction["three_forecast_digests_identical"]
    capture = ROOT / "highlands-capture/capture"
    admission = old.read(ROOT / "highlands-admission.json")
    captured = old.read(capture / "result.json")
    result_dir = ROOT / "highlands-replay"
    result = old.read(result_dir / "result.json")
    assert result["source_sha256"][str(receipt)] == frozen_digest
    for path, expected in result["source_sha256"].items():
        assert old.digest(Path(path)) == expected, path
    assert result["track"] not in set(frozen["development"] + frozen["exclusions"])
    assert result["track"] == captured["track"] == admission["track"] == "ks_highlands"
    assert result["layout"] == captured["layout"] == admission["layout"] == "layout_short"
    assert result["both_directions_evaluated"] and captured["closed"]["worker_alive"] is False
    assert old.digest(capture / "result.json") == admission["capture_sha256"]
    assert old.digest(capture / "archive-map.json") == admission["archives_sha256"]
    for name, expected in (("labels", result["labels_sha256"]), ("forecasts", result["forecasts_sha256"])):
        assert old.digest(result_dir / f"{name}.jsonl") == expected

    baseline = rows(result_dir / "archive-replay/predictions/forecasts.jsonl")
    forecast = rows(result_dir / "forecasts.jsonl")
    labels = rows(result_dir / "labels.jsonl")
    assert len(labels) == captured["frames"] == admission["frames"]
    assert keys(labels) == keys(baseline) == keys(forecast)
    assert all(frame["camera_complete"] for frame in labels)
    before, after = score(baseline, labels), score(forecast, labels)
    assert (before, after) == (result["before"], result["after"])
    assert sum(after["states"].values()) == len(labels)
    assert sum(after["counts"].values()) == len(labels)
    assert all(a == b for a, b in zip(baseline, forecast, strict=True)
               if a["forecasts"][METHOD]["state"] == "right")
    risk = {kind: after["counts"].get(kind, 0) - before["counts"].get(kind, 0)
            for kind in frozen["per_track_caps"]}
    assert risk == result["risk_delta"]
    assert result["per_recording_risk_pass"] == all(
        value <= frozen["per_track_caps"][kind] for kind, value in risk.items())
    assert Counter(row["truth"] for row in labels) == admission["truth_counts"]

    table = {}
    all_tracks = {**comparison["tracks"], result["track"]: result}
    for track, row in all_tracks.items():
        table[track] = {}
        for stage in ("before", "after"):
            data = row[stage]
            approach = data["approaches"]
            table[track][stage] = {
                "left": [approach["1"]["joint_hit"], approach["1"]["frames"]],
                "right": [approach["2"]["joint_hit"], approach["2"]["frames"]],
                "approach_abstentions": sum(side.get("abstain", 0) for side in approach.values()),
                "approach_no_outputs": sum(side.get("no_output", 0) for side in approach.values()),
                "approach_misses": sum(side["miss"] for side in approach.values()),
                "whole_recording_abstentions": data["states"].get("abstain", 0),
                "whole_recording_no_outputs": data["states"].get("no_output", 0),
                **{kind: data["counts"].get(kind, 0) for kind in frozen["per_track_caps"]},
            }
    old.write(ROOT / "verification.json", dict(
        fixed_sha256=frozen_digest, source_revision=frozen["source_revision"],
        compared_tracks=[track for track, _ in EVAL], transfer_track=result["track"],
        transfer_layout=result["layout"], transfer_frames=len(labels),
        transfer_risk_pass=result["per_recording_risk_pass"],
        tracks=table, result_sha256=old.digest(result_dir / "result.json"),
        admission_sha256=old.digest(ROOT / "highlands-admission.json"),
        program_sha256=old.digest(Path(__file__)), seed=None, termination="verification_complete"))
    print(json.dumps(table, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
