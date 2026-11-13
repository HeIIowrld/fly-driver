"""Summarize matched alert changes and verify full-timeline replay evidence."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(row, proposal):
    truth = row["truth"]
    if proposal is None:
        return "quiet"
    if truth != "approach":
        return truth + "_alert"
    side, distance = proposal
    if str(side) != row["label"]:
        return "wrong_direction"
    if max(abs(distance - d) for d in row["truth_endpoints_m"]) <= 5.:
        return "joint_hit"
    return "distance_error"


def count(runs):
    records = list(runs.values())
    return dict(frames=sum(r["frames"] for r in records),
                left=sum(r["by_direction"]["left"]["direction_hits"] for r in records),
                right=sum(r["by_direction"]["right"]["direction_hits"] for r in records),
                joint=sum(s["joint_hits"] for r in records for s in r["by_direction"].values()),
                outside=sum(r["false_alerts"]["outside_band"]["frames"] for r in records),
                stable=sum(len({e["physical_corner"] for e in r["events"]
                                if e["first_stable_frame"] is not None}) for r in records))


def main():
    result = OUT/"result.json"
    receipt = json.loads(result.read_text())
    assert receipt["status"] == "evaluated"
    for source, digest in receipt["source_sha256"].items():
        assert sha(source) == digest, source
    totals = {}
    for track, data in receipt["tracks"].items():
        path, cache = (OUT/f"{track}-{suffix}" for suffix in ("frames.jsonl", "sections.npz"))
        assert sha(path) == data["rows_sha256"]
        assert sha(cache) == data["sections_sha256"]
        for source, digest in data["source_sha256"].items():
            assert sha(source) == digest, source
        rows = [json.loads(line) for line in path.open()]
        with np.load(cache, allow_pickle=False) as saved:
            assert [(r["run"], r["frame"]) for r in rows] == list(zip(saved["run"], saved["frame"], strict=True))
        metrics = {m: count(runs) for m, runs in data["methods"].items()}
        assert all(m["frames"] == len(rows) == 1200 for m in metrics.values())
        transitions = {}
        examples = []
        for variant in ("", "_hold"):
            a, b = ("ranked_local_"+kind+variant for kind in ("boundary", "t4t5"))
            tally = Counter()
            for row in rows:
                left, right = (row["proposals"][mode] for mode in (a,b))
                if left == right:
                    continue
                x, y = category(row, left), category(row, right)
                tally[x,y] += 1
                if row["truth"] == "approach":
                    examples.append(dict(run=row["run"], frame=row["frame"], variant=variant, before=x, after=y,
                                         proposal_boundary=left, proposal_t4t5=right, truth_m=row["truth_endpoints_m"]))
            transitions[variant or "direct"] = [dict(before=x, after=y, frames=n)
                                                for (x,y), n in sorted(tally.items())]
        totals[track] = dict(metrics=metrics, paired_changes=transitions, approaches=examples)
    (OUT/"paired-summary.json").write_text(json.dumps(dict(tracks=totals, result_sha256=sha(result),
        source_sha256=sha(__file__)), indent=2)+"\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
