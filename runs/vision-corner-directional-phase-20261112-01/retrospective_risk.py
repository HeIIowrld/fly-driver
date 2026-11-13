"""Measure score ordering under the prior replay's two false-alarm counts."""
from collections import Counter

from directional_features import OUT, PHASE, old
from verify_directional import frames


def main():
    previous_path, result_path = PHASE/"result.json", OUT/"result.json"
    previous, result = old.read(previous_path), old.read(result_path)
    reports = {}
    for track, report in result["tracks"].items():
        prior = previous["tracks"][track]["metrics"]["t4t5_confirm_veto_candidate_only"]
        rows = frames(OUT/f"{track}-veto-frames.jsonl", report["veto_rows_sha256"])
        track_report = {}
        for method in ("temporal", "t4t5_confirm"):
            by_score = {}
            for row in rows:
                pick = row["scores"][method]
                if pick["score"] is None or pick["score"] < 0.:
                    continue
                bucket = by_score.setdefault(pick["score"], Counter())
                outcome = old.category(row, pick["candidate"])
                if outcome in ("straight_alert", "out_of_range_alert"):
                    bucket["outside"] += 1
                elif outcome == "in_corner_alert":
                    bucket["in_corner"] += 1
                elif outcome == "joint_hit":
                    bucket["joint"] += 1
                    bucket["right_joint"] += int(row["label"] == "2")
            count = Counter()
            for score in sorted(by_score, reverse=True):
                candidate = count + by_score[score]
                if candidate["outside"] > prior["outside"] or candidate["in_corner"] > prior["in_corner"]:
                    break
                count = candidate
            track_report[method] = {k: count[k] for k in ("joint", "right_joint", "outside", "in_corner")}
        reports[track] = dict(caps={k: prior[k] for k in ("outside", "in_corner")}, methods=track_report)
    old.write(OUT/"retrospective-risk.json", dict(status="heldout_diagnostic_only", tracks=reports,
              prior_sha256=old.digest(previous_path), result_sha256=old.digest(result_path),
              termination="score_ordering_compared_at_prior_false_counts"))
    print(reports)


if __name__ == "__main__":
    main()
