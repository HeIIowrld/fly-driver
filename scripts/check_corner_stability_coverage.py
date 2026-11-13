"""Check whether a capture has enough samples to evaluate stable corner warnings."""

import argparse
from pathlib import Path

from evaluate_fused_corner import read_rows
from score_corner_captures import STABLE_FRAMES, adjacent
from train_vision_teacher_line import digest, write


def coverage(rows, minimum_per_side):
    """Count sampled corners and the subset with a scoreable consecutive window."""
    observed = {"1": set(), "2": set()}
    scoreable = {"1": set(), "2": set()}
    previous = None
    streak = 0
    for row in rows:
        if (row["truth"] != "approach" or not row["camera_complete"]
                or row["label"] not in observed or row["physical_corner"] is None):
            previous, streak = None, 0
            continue
        label, corner = row["label"], row["physical_corner"]
        observed[label].add(corner)
        same_approach = (previous is not None and adjacent(previous, row)
                         and previous["physical_corner"] == corner and previous["label"] == label)
        streak = streak + 1 if same_approach else 1
        if streak >= STABLE_FRAMES:
            scoreable[label].add(corner)
        previous = row
    return {
        "stable_frames": STABLE_FRAMES,
        "minimum_per_side": minimum_per_side,
        "by_side": {label: {"sampled": sorted(observed[label]), "scoreable": sorted(scoreable[label]),
                            "insufficient": sorted(observed[label] - scoreable[label])}
                    for label in observed},
        "minimum_coverage_met": all(len(corners) >= minimum_per_side for corners in scoreable.values()),
        "all_sampled_scoreable": observed == scoreable and any(observed.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--minimum-per-side", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_per_side < 1:
        parser.error("--minimum-per-side must be positive")
    if args.output.exists():
        parser.error("--output already exists")
    result = coverage(read_rows(args.rows), args.minimum_per_side)
    result.update(rows_sha256=digest(args.rows), program_sha256=digest(Path(__file__)),
                  termination="coverage_checked")
    write(args.output, result)


if __name__ == "__main__":
    main()
