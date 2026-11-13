"""Separate entry from continuing curvature in the onset supervision."""
from collections import Counter
from pathlib import Path

import train_shape as original
from replay_learned import baseline_data, frame_key

PARENT = original.OUT
OUT = PARENT / "entry-supervised"
physical_targets = original.onset_targets
target_audit = {}


def entry_targets(track, entry, sources):
    directions, distance = physical_targets(track, entry, sources)
    labels, _ = original.labels(track)
    group, _, _, _ = baseline_data(track, sources)
    rows = group["rows"]
    assert [frame_key(row) for row in rows] == list(zip(labels["run"], labels["frame"], strict=True))
    changes = Counter()
    for i, row in enumerate(rows):
        initial = directions[i]
        truth = row["truth"]
        if initial >= 0 and truth in {"straight", "out_of_range", "in_corner"}:
            directions[i] = 0
        elif initial > 0 and truth == "approach" and int(row["label"]) == initial:
            meters = original.MIN_RANGE + (original.MAX_RANGE-original.MIN_RANGE)*distance[i]
            low, high = row["truth_endpoints_m"]
            if not low-5. <= meters <= high+5.:
                directions[i] = -1
        else:
            directions[i] = -1
        changes[truth, int(initial), int(directions[i])] += 1
    target_audit[track] = [dict(truth=t, original=a, corrected=b, frames=n)
                           for (t,a,b),n in sorted(changes.items())]
    return directions, distance


def main():
    OUT.mkdir(exist_ok=True)
    original.OUT = OUT
    original.onset_targets = entry_targets
    original.main()
    path = OUT / "training.json"
    receipt = original.old.read(path)
    receipt["source_sha256"][str(Path(__file__).resolve())] = original.old.digest(Path(__file__))
    receipt["teacher_changes"] = target_audit
    receipt["teacher_policy"] = (
        "Require physical geometry and matching entry direction with distance within 5 m of the offline timing label. "
        "Use no-onset for supported straight, out-of-range and already-in-corner records. "
        "Mask stationary and disputed onset samples."
    )
    original.old.write(path, receipt)


if __name__ == "__main__":
    main()
