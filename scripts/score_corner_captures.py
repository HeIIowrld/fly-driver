"""Score frozen next-corner decisions on a complete capture timeline."""
from collections import Counter
from math import isfinite
from statistics import mean

MIN_SPEED_MPS = 1 / 3.6
DISTANCE_TOLERANCE_M = 5
MAX_GAP_S = .18
STABLE_FRAMES = 2
STATES = ("no_output", "abstain", "straight", "left", "right")


def forecast_state(row):
    if row is None:
        return "no_output"
    if not row["available"]:
        return "abstain"
    direction = row["direction"]
    if row["alert"]:
        distance = row["distance_m"]
        if direction not in (1, 2) or distance is None or not isfinite(distance) or not 0 <= distance <= 20:
            raise ValueError("Alert lacks a valid near-range forecast")
        return "left" if direction == 1 else "right"
    return "straight" if direction == 0 else "abstain"


def capture_row(run, capture, prediction, endpoints, speeds):
    early, late = endpoints
    label = early[0] if early[0] == late[0] and early[2] == late[2] else "ambiguous"
    distances = [d for _, d, _ in endpoints if d is not None]
    moving = all(s is not None and isfinite(s) and s >= MIN_SPEED_MPS for s in speeds)
    if not moving:
        category = "motion_unverified"
        if all(s is not None and isfinite(s) and 0 <= s < MIN_SPEED_MPS for s in speeds):
            category = "stationary"
    elif label == "0":
        category = "straight"
    elif label in ("1", "2") and len(distances) == 2:
        if min(distances) >= 4 and max(distances) <= 20:
            category = "approach"
        elif max(distances) < 4 or min(distances) > 20:
            category = "out_of_range"
        else:
            category = "band_boundary"
    else:
        category = label

    state = forecast_state(prediction)
    forecast = prediction if prediction is not None else {}
    distance = forecast.get("distance_m")
    reason = None
    if state == "abstain":
        if not forecast["available"]:
            reason = forecast.get("path_status") if forecast.get("body_valid") is False else "readout_unavailable"
        else:
            reason = "distance_out_of_range" if distance is not None and distance > 20 else "presence_rejected"
    hit = category == "approach" and state == {"1": "left", "2": "right"}[label]
    errors = [distance - truth for truth in distances] if hit else []
    return dict(run=run, frame=capture["capture_id"], received_s=capture["received"],
                generation=capture.get("generation"), state=state, abstain_reason=reason, truth=category, label=label,
                physical_corner=early[2] if early[2] == late[2] else None,
                truth_endpoints_m=distances, truth_distance_m=mean(distances) if distances else None,
                predicted_direction=forecast.get("direction"), presence_score=forecast.get("presence_score"),
                predicted_distance_m=distance, direction_hit=hit,
                joint_hit=bool(errors) and max(map(abs, errors)) <= DISTANCE_TOLERANCE_M,
                signed_distance_error_m=mean(errors) if errors else None,
                absolute_distance_error_m=mean(map(abs, errors)) if errors else None)


def adjacent(previous, row):
    return (row["run"] == previous["run"] and row["generation"] == previous["generation"]
            and row["frame"] == previous["frame"] + 1
            and 0 < row["received_s"] - previous["received_s"] <= MAX_GAP_S)


def streaks(rows):
    chains = []
    for row in rows:
        if not chains or not adjacent(chains[-1][-1], row):
            chains.append([])
        chains[-1].append(row)
    return chains


def summarize(rows):
    matrix = {}
    for row in rows:
        counts = matrix.setdefault(row["truth"], dict.fromkeys(STATES, 0))
        counts[row["state"]] += 1
    near = [r for r in rows if r["truth"] == "approach"]
    sides = {}
    for code, direction in (("1", "left"), ("2", "right")):
        selected = [r for r in near if r["label"] == code]
        sides[direction] = dict(frames=len(selected), direction_hits=sum(r["direction_hit"] for r in selected),
                                joint_hits=sum(r["joint_hit"] for r in selected),
                                outputs=dict(Counter(r["state"] for r in selected)))
    # A camera gap breaks stability, but does not start a new physical approach.
    approaches = []
    previous = None
    for row in rows:
        if (previous is not None and (row["generation"] != previous["generation"]
                or (row["physical_corner"] is not None and row["physical_corner"] != previous["physical_corner"])
                or row["truth"] == "in_corner")):
            previous = None
        if row["truth"] != "approach":
            continue
        if previous is None:
            approaches.append([])
        approaches[-1].append(row)
        previous = row

    events, pairs = [], []
    for approach in approaches:
        hits = [r for r in approach if r["joint_hit"]]
        stable = next((chain[STABLE_FRAMES - 1] for chain in streaks(hits)
                       if len(chain) >= STABLE_FRAMES), None)
        first = approach[0]
        events.append(dict(physical_corner=first["physical_corner"], first_frame=first["frame"],
                           first_hit_lead_m=hits[0]["truth_distance_m"] if hits else None,
                           last_frame=approach[-1]["frame"], label=first["label"], frames=len(approach),
                           joint_hits=len(hits), first_stable_frame=stable["frame"] if stable else None,
                           first_stable_lead_m=stable["truth_distance_m"] if stable else None))
        for a, b in zip(approach, approach[1:]):
            if adjacent(a, b) and a["direction_hit"] and b["direction_hit"]:
                truth_delta = b["truth_distance_m"] - a["truth_distance_m"]
                forecast_delta = b["predicted_distance_m"] - a["predicted_distance_m"]
                pairs.append(dict(first_frame=a["frame"], last_frame=b["frame"],
                                  truth_delta_m=truth_delta, forecast_delta_m=forecast_delta,
                                  delta_error_m=forecast_delta - truth_delta))

    negatives = {}
    for name, categories in (("straight", ("straight",)), ("out_of_range", ("out_of_range",)),
                             ("outside_band", ("straight", "out_of_range"))):
        false_alarms = [r for r in rows if r["truth"] in categories and r["state"] in ("left", "right")]
        chains = streaks(false_alarms)
        negatives[name] = dict(frames=len(false_alarms), episodes=len(chains),
                               episodes_2plus=sum(len(c) >= 2 for c in chains),
                               longest_frames=max(map(len, chains), default=0),
                               streaks=[dict(first_frame=c[0]["frame"], last_frame=c[-1]["frame"],
                                             frames=len(c), duration_s=c[-1]["received_s"] - c[0]["received_s"])
                                        for c in chains])
    hits = [r for r in near if r["direction_hit"]]
    return dict(frames=len(rows), outputs=dict.fromkeys(STATES, 0) | dict(Counter(r["state"] for r in rows)),
                truth_by_output=matrix, approaches=len(approaches), approach_frames=len(near),
                by_direction=sides, events=events, false_alerts=negatives,
                distance_bias_m=mean(r["signed_distance_error_m"] for r in hits) if hits else None,
                distance_mae_m=mean(r["absolute_distance_error_m"] for r in hits) if hits else None,
                distance_change=dict(pairs=len(pairs),
                                     mae_m=mean(abs(p["delta_error_m"]) for p in pairs) if pairs else None,
                                     receding_when_truth_closing=sum(p["truth_delta_m"] < 0
                                         and p["forecast_delta_m"] > 0 for p in pairs),
                                     rows=pairs))
