"""Audit whether missed true onsets are supported by recorded paired boundaries."""
import json
from collections import Counter

import numpy as np

from listwise import OUT, PREVIOUS, SILVER, old
from directional_features import PARENT, hinge, signed_curve
from verify_directional import frames


def measure(stations, bounds, distance):
    x = stations - distance
    width = bounds[:, 1] - bounds[:, 0]
    support = np.isfinite(bounds).all(1) & (width > 3.) & (width < 20.)
    before = support & (x >= -12.) & (x <= 0.)
    after = support & (x >= 0.) & (x <= 12.)
    window = support & (abs(x) <= 12.)
    curve_before = signed_curve(stations, bounds, before)
    curve_after = signed_curve(stations, bounds, after)
    step = hinge(x[window], bounds[window])
    continuous = bool(window.sum() > 1 and np.max(np.diff(x[window])) <= 4.)
    supported = bool(step[0] and continuous and curve_before[0] >= 3 and curve_after[0] >= 3)
    delta = curve_after[1] - curve_before[1]
    return dict(before_samples=int(before.sum()), after_samples=int(after.sum()), fit_supported=supported,
                before_curve_m_inv=curve_before[1], after_curve_m_inv=curve_after[1],
                curve_change_m_inv=delta, hinge_change_m_inv=step[1], hinge_rms_gain_m=step[-1])


def main():
    previous = old.read(PREVIOUS/"result.json")
    replay = PREVIOUS/f"{SILVER}-frames.jsonl"
    records = frames(replay, previous["tracks"][SILVER]["rows_sha256"])
    observations = PARENT/f"{SILVER}-observations.npz"
    assert old.digest(observations) == old.read(observations.with_suffix(".json"))["arrays_sha256"]
    result = []
    with np.load(observations, allow_pickle=False) as data:
        assert list(zip(data["run"], data["frame"], strict=True)) == [(r["run"], r["frame"]) for r in records]
        for row, bounds in zip(records, data["sections"], strict=True):
            if row["truth"] != "approach" or row["label"] != "2":
                continue
            distance = float(np.mean(row["truth_endpoints_m"]))
            features = measure(old.STATIONS, bounds, distance)
            present = any(old.category(row, offer) == "joint_hit" for offer in row["pool"])
            result.append(dict(frame=row["frame"], truth_distance_m=distance, has_joint_candidate=present,
                               pool=row["pool"], **features))
    counts = Counter()
    for row in result:
        if row["has_joint_candidate"]:
            continue
        if not row["fit_supported"]:
            counts["insufficient_continuous_samples"] += 1
        elif row["hinge_change_m_inv"] < 0. and row["curve_change_m_inv"] < 0.:
            counts["both_estimators_bend_right"] += 1
        else:
            counts["estimators_do_not_both_bend_right"] += 1
    old.write(OUT/"missing-support.json", dict(status="label_position_diagnostic_only", counts=dict(counts),
              frames=result, source_sha256={str(p): old.digest(p) for p in (observations, replay)},
              termination="paired_boundary_support_at_actual_onsets_audited"))
    print(dict(counts))


if __name__ == "__main__":
    main()
