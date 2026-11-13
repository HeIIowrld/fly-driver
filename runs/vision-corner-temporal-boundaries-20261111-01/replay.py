"""Compare causal boundary fusion with single-frame curvature on saved captures."""
from collections import defaultdict, deque
import json
from pathlib import Path
from statistics import mean

import numpy as np

from score_corner_captures import summarize
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
runs = out.parent
stations = np.arange(2., 25., 2.)
cuts = (.002, .004, .008)
ages = (.24, .48, .85)
radius = 6.
sides = {1: "left", 2: "right"}
extraction = read(out/"extraction.json")


def read_rows(path, sha=None):
    if sha is not None:
        assert digest(path) == sha, path
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row):
    return row["run"], row["frame"]


def advance(points, rotation, move):
    return (points-move) @ rotation


def transform(motion, old_motion):
    t, _, _ = motion
    before, speed, yaw = old_motion
    dt = t-before
    if not 0.045 < dt <= .18:
        return None
    theta = yaw*dt
    c, s = np.cos(theta), np.sin(theta)
    move = speed*dt*np.array([np.cos(theta/2), np.sin(theta/2)])
    return np.array([[c, -s], [s, c]]), move


def section_values(points):
    """Only interpolate between observed points separated by at most two sections."""
    result = np.full((len(stations), 2), np.nan)
    for side in range(2):
        curve = points[:, side]
        if len(curve) == 0:
            continue
        order = np.argsort(curve[:, 0])
        x, y = curve[order].T
        stop = np.searchsorted(x, stations)
        for i, j in enumerate(stop):
            if j < len(x) and abs(x[j]-stations[i]) < 1e-8:
                result[i, side] = y[j]
            elif 0 < j < len(x) and x[j]-x[j-1] <= 4.1:
                fraction = (stations[i]-x[j-1])/(x[j]-x[j-1])
                result[i, side] = y[j-1]+fraction*(y[j]-y[j-1])
    return result


def fuse(values, times, now, age):
    stack = values[(now-times) <= age]
    available = np.isfinite(stack).all(axis=-1)
    fused = np.full((len(stations), 2), np.nan)
    for i, valid in enumerate(available.T):
        # A fused section needs two separately captured views of the same location.
        if valid.sum() >= 2:
            fused[i] = np.median(stack[valid, i], axis=0)
    return fused


def curvature(sections):
    samples = []
    usable = np.isfinite(sections).all(axis=1)
    width = sections[:, 1]-sections[:, 0]
    usable &= (width > 3.) & (width < 20.)
    for center in np.arange(4., 22.):
        offset = stations-center
        mask = usable & (abs(offset) <= radius)
        x = offset[mask]
        if len(x) < 4 or x[0] >= 0 or x[-1] <= 0 or max(np.diff(x)) > 4.:
            continue
        basis = np.column_stack((np.ones(len(x)), x, x*x))
        coefficients = np.linalg.lstsq(basis, sections[mask], rcond=None)[0]
        k = 2.*coefficients[2]/(1.+coefficients[1]**2)**1.5
        signed = float(np.copysign(min(abs(k)), k[0])) if k[0]*k[1] > 0 else 0.
        samples.append((float(center), signed))
    return samples


def decide(samples, cut):
    if len(samples) < 2 or not any(b[0]-a[0] == 1. for a, b in zip(samples, samples[1:])):
        return None
    for (x, a), (y, b) in zip(samples, samples[1:]):
        if y-x == 1. and a*b > 0 and min(abs(a), abs(b)) >= cut:
            return (1 if a > 0 else 2, x)
    return 0, None


def predictions(observations):
    previous = None
    history = deque()
    for row in observations:
        motion = row["motion"]
        points = row["sections"]
        step = None
        if (previous and motion is not None and previous["motion"] is not None and
                row["run"] == previous["run"] and row["generation"] == previous["generation"] and
                row["frame"] == previous["frame"]+1):
            step = transform(motion, previous["motion"])
        if step is None or points is None:
            history.clear()
        else:
            rotation, move = step
            while history and motion[0]-history[0][0] > max(ages):
                history.popleft()
            history = deque((stamp, advance(obs, rotation, move)) for stamp, obs in history)
        results = {}
        if points is not None:
            sections = np.array(points, dtype=float).reshape(-1, 3)
            direct = np.full((len(stations), 2), np.nan)
            direct[np.searchsorted(stations, sections[:, 0])] = sections[:, 1:]
            results["current"] = curvature(direct)
            if motion is not None:
                coords = np.stack((np.repeat(sections[:, :1], 2, axis=1), sections[:, 1:]), axis=-1)
                history.append((motion[0], coords))
                values = np.stack([section_values(obs) for _, obs in history])
                times = np.array([stamp for stamp, _ in history])
                for age in ages:
                    fused = fuse(values, times, motion[0], age)
                    results[f"history_{age}"] = curvature(fused)
        yield dict(run=row["run"], frame=row["frame"], curvature=results,
                   motion_contiguous=step is not None,
                   travel_m=float(step[1][0]) if step is not None else None)
        previous = row if points is not None else None


def confirmed_predictions(rows, cut, required, tolerance):
    """Require the predicted position to follow measured travel before alerting."""
    confirmations = {}
    previous = None
    count = 0
    for row in rows:
        choice = decide(row["curvature"].get("history_0.24", []), cut)
        travel = row["travel_m"]
        if (travel is not None and travel >= .10 and choice and choice[0] and previous and
                previous[0] == choice[0] and abs(choice[1] - (previous[1]-travel)) <= tolerance):
            count += 1
        else:
            count = 1 if choice and choice[0] else 0
        confirmations[key(row)] = choice if count >= required else None
        previous = choice
    return confirmations


def rescore(row, decision):
    if decision is None:
        return dict(row, state="abstain", abstain_reason="boundary_support_unavailable",
                    predicted_direction=0, predicted_distance_m=None, presence_score=None,
                    direction_hit=False, joint_hit=False,
                    signed_distance_error_m=None, absolute_distance_error_m=None)
    side, distance = decision
    hit = row["truth"] == "approach" and str(side) == row["label"]
    errors = [distance-truth for truth in row["truth_endpoints_m"]] if hit else []
    return dict(row, state=sides.get(side, "straight"), abstain_reason=None,
                predicted_direction=side, predicted_distance_m=distance, presence_score=None,
                direction_hit=hit, joint_hit=bool(errors) and max(map(abs, errors)) <= 5.,
                signed_distance_error_m=mean(errors) if errors else None,
                absolute_distance_error_m=mean(map(abs, errors)) if errors else None)


def reports(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run"]].append(row)
    return {run: summarize(selected) for run, selected in grouped.items()}


def compact(summary):
    changes = [p for r in summary.values() for p in r["distance_change"]["rows"]]
    events = [e for r in summary.values() for e in r["events"]]
    return dict(frames=sum(r["frames"] for r in summary.values()),
                abstain=sum(r["outputs"]["abstain"] for r in summary.values()),
                left=sum(r["by_direction"]["left"]["direction_hits"] for r in summary.values()),
                right=sum(r["by_direction"]["right"]["direction_hits"] for r in summary.values()),
                joint=sum(s["joint_hits"] for r in summary.values() for s in r["by_direction"].values()),
                outside=sum(r["false_alerts"]["outside_band"]["frames"] for r in summary.values()),
                stable_leads_m=[e["first_stable_lead_m"] for e in events if e["first_stable_lead_m"] is not None],
                pairs=len(changes), delta_mae_m=mean(abs(p["delta_error_m"]) for p in changes) if changes else None,
                receding=sum(p["truth_delta_m"] < 0 and p["forecast_delta_m"] > 0 for p in changes))


def matched_changes(a, b):
    def pairs(report):
        return {(run, p["first_frame"], p["last_frame"]): p for run, data in report.items()
                for p in data["distance_change"]["rows"]}
    aa, bb = pairs(a), pairs(b)
    keys = sorted(aa.keys() & bb.keys())
    stats = {}
    for name, data in (("baseline", aa), ("candidate", bb)):
        stats[name] = dict(receding=sum(data[k]["truth_delta_m"] < 0 and data[k]["forecast_delta_m"] > 0 for k in keys),
                           delta_mae_m=mean(abs(data[k]["delta_error_m"]) for k in keys) if keys else None)
    return dict(pairs=len(keys), **stats)


def run():
    observation_file = out/"observations.jsonl"
    all_predictions = list(predictions(read_rows(observation_file, extraction["observations_sha256"])))
    by_key = {key(r): r for r in all_predictions}
    assert len(by_key) == len(all_predictions)
    jsonl(out/"geometry.jsonl", all_predictions)
    scores = {
        "magione": runs/"vision-corner-independent-runner-rehearsal-20261111-01"/"replay"/"candidate"/"score"/"rows.jsonl",
        "imola": runs/"vision-corner-imola-full-blended-20261111-01"/"rows.jsonl",
    }
    result = {}
    scored_examples = []
    for track, path in scores.items():
        reference = read_rows(path)
        assert {key(r) for r in reference} == {k for k in by_key if ("imola" in k[0]) == (track == "imola")}
        reference_report = reports(reference)
        track_result = {"existing_blend": compact(reference_report)}
        for mode in ("current", *(f"history_{age}" for age in ages)):
            for cut in cuts:
                for supplement in (False, True):
                    rows = []
                    for row in reference:
                        candidate = decide(by_key[key(row)]["curvature"].get(mode, []), cut)
                        keep = supplement and (row["state"] in sides.values() or not candidate or candidate[0] == 0)
                        rows.append(row if keep else rescore(row, candidate))
                    name = f"{'add' if supplement else 'only'}_{mode}_{cut}"
                    report = reports(rows)
                    track_result[name] = dict(compact(report), paired_with_blend=matched_changes(reference_report, report))
        for cut in cuts:
            for required in (2, 3):
                for tolerance in (1., 2.):
                    confirmed = confirmed_predictions(all_predictions, cut, required, tolerance)
                    rows = [rescore(row, confirmed[key(row)]) if confirmed[key(row)]
                            and row["state"] not in sides.values() else row for row in reference]
                    report = reports(rows)
                    name = f"confirm_{cut}_{required}_{tolerance}"
                    track_result[name] = dict(compact(report), paired_with_blend=matched_changes(reference_report, report))
        for cut in cuts:
            direct = reports([rescore(row, decide(by_key[key(row)]["curvature"].get("current", []), cut))
                              for row in reference])
            fused = reports([rescore(row, decide(by_key[key(row)]["curvature"].get("history_0.24", []), cut))
                             for row in reference])
            track_result[f"only_history_0.24_{cut}"]["paired_with_current"] = matched_changes(direct, fused)
        # Retain the reference alerts and replace only their distances when the
        # short history agrees on direction and the two measurements are close.
        for cut in cuts:
            for tolerance in (2., 5.):
                for gain in (.25, .5):
                    updated = 0
                    rows = []
                    for row in reference:
                        distance = row["predicted_distance_m"]
                        if row["state"] not in sides.values():
                            rows.append(row)
                            continue
                        side = row["predicted_direction"]
                        candidate = decide(by_key[key(row)]["curvature"].get("history_0.24", []), cut)
                        if (candidate and candidate[0] == side and
                                abs(candidate[1]-distance) <= tolerance):
                            distance += gain*(candidate[1]-distance)
                            changed = rescore(row, (side, distance))
                            updated += int(distance != row["predicted_distance_m"])
                            scored_examples.append(dict(track=track, cut=cut, tolerance=tolerance,
                                gain=gain, before=row, after=changed))
                            rows.append(changed)
                        else:
                            rows.append(row)
                    name = f"distance_{cut}_{tolerance}_{gain}"
                    report = reports(rows)
                    track_result[name] = dict(compact(report), updated=updated,
                                              paired_with_blend=matched_changes(reference_report, report))
        result[track] = track_result
    jsonl(out/"distance-rows.jsonl", scored_examples)
    write(out/"result.json", dict(summary=result, scoring_inputs={str(p): digest(p) for p in scores.values()},
        extraction_sha256=digest(out/"extraction.json"), geometry_sha256=digest(out/"geometry.jsonl"),
        distance_rows_sha256=digest(out/"distance-rows.jsonl"),
        source_sha256=digest(Path(__file__)), scoring_source_sha256=digest(Path("scripts/score_corner_captures.py")),
        revision=extraction["revision"], seed=None, candidate_promoted=False, termination="archive_end"))
    for track, values in result.items():
        for name, r in values.items():
            print(track, name, *(r[k] for k in ("left", "right", "joint", "outside", "pairs", "receding")),
                  r.get("delta_mae_m"), r.get("updated"))


if __name__ == "__main__":
    run()
