"""Compare score ordering when both previous replay false-alarm counts are capped."""
from collections import Counter

from aligned_sections import OUT, old
from verify_directional import frames


def under_caps(rows, method, caps):
    bins = {}
    for row in rows:
        forecast = row["scores"][method]
        if forecast["score"] is None or forecast["score"] < 0.:
            continue
        bucket = bins.setdefault(forecast["score"], Counter())
        category = old.category(row, forecast["candidate"])
        if category in ("straight_alert", "out_of_range_alert"):
            bucket["outside"] += 1
        elif category == "in_corner_alert":
            bucket["in_corner"] += 1
        elif category == "joint_hit":
            bucket["joint"] += 1
            bucket["right_joint"] += int(row["label"] == "2")
    count = Counter()
    for score in sorted(bins, reverse=True):
        update = count + bins[score]
        if any(update[k] > caps[k] for k in caps):
            break
        count = update
    return {k: count[k] for k in ("outside", "in_corner", "joint", "right_joint")}


def main():
    source = OUT/"result.json"
    result = old.read(source)
    reports = {}
    for track, report in result["tracks"].items():
        previous = report["variants"]["direct"]["metrics"]["t4t5_confirm_candidate_only"]
        caps = {k: previous[k] for k in ("outside", "in_corner")}
        scores = {}
        for variant, info in report["variants"].items():
            rows = frames(OUT/f"{track}-{variant}.jsonl", info["rows_sha256"])
            scores[variant] = {m: under_caps(rows, m, caps) for m in ("temporal", "t4t5_confirm")}
        reports[track] = dict(caps=caps, scores=scores)
    old.write(OUT/"matched-risk.json", dict(status="heldout_diagnostic_only", tracks=reports,
              result_sha256=old.digest(source), termination="score_ordering_at_previous_false_alarm_caps"))
    print(reports)


if __name__ == "__main__":
    main()
