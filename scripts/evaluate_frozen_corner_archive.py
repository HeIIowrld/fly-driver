"""Evaluate frozen forecasts using a separately preserved corner-label index."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path

import numpy as np

from train_vision_teacher_line import digest, read, write


def indexed(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    result = {(row["run"],row["frame"]):row for row in rows}
    if not result or len(result)!=len(rows):
        raise ValueError("Expected a nonempty index with unique run/frame pairs")
    return result


def evaluate(forecasts, labels, output, pins=None, expected_track=None):
    prediction_proof = read(forecasts.with_name("result.json"))
    label_proof = read(labels.with_name("result.json"))
    if prediction_proof["rows_sha256"]!=digest(forecasts) or label_proof["rows_sha256"]!=digest(labels):
        raise ValueError("Forecast or label digest differs")
    predictions, truth = indexed(forecasts), indexed(labels)
    if predictions.keys()!=truth.keys():
        raise ValueError("Forecasts and labels must cover the same frames")
    track = prediction_proof.get("verified_track_id")
    label_track = label_proof.get("verified_track_id")
    bundle_key = "camera_bundle_sha256_by_run"
    if (expected_track is not None or track is not None or label_track is not None
            or bundle_key in prediction_proof or bundle_key in label_proof):
        if (not isinstance(track, str) or re.fullmatch(r"[a-z0-9_-]+", track) is None
                or track != label_track or (expected_track is not None and track != expected_track)):
            raise ValueError("Verified forecast and label tracks must match")
        bundles = prediction_proof.get(bundle_key)
        if (not isinstance(bundles, dict) or bundles != label_proof.get(bundle_key)
                or bundles.keys() != {run for run, _ in predictions}
                or any(not isinstance(h, str) or re.fullmatch(r"[a-f0-9]{64}", h) is None
                       for h in bundles.values())):
            raise ValueError("Verified forecast and label camera bundles must match")
        provenance = dict(verified_track_id=track, camera_bundle_sha256_by_run=bundles)
    else:
        provenance = {}
    diagnostics = None
    diagnostic_inputs = ()
    if pins is not None:
        from infer_frozen_next_corner import load_readout

        pin_hash = digest(pins)
        if pin_hash not in prediction_proof["inputs_sha256"].values():
            raise ValueError("Pins do not match the forecast receipt")
        _, threshold = load_readout(read(pins))
        diagnostics = {
            str(side): {band: Counter() for band in ("4-10m", "over-10-20m")}
            for side in (1, 2)
        }
        diagnostic_inputs = (pins,)

    counts = Counter()
    by_side = {side:Counter() for side in (1,2)}
    events = {}
    errors = []
    for key,actual in truth.items():
        forecast = predictions[key]
        direction,distance = actual["direction"],actual["distance"]
        expected = direction in by_side and distance is not None and 4<=distance<=20
        alert = forecast["alert"]
        available = forecast["available"]
        estimate = forecast["distance_m"]
        if alert and (not available or forecast["direction"] not in by_side or estimate is None
                      or not np.isfinite(estimate) or not 0<=estimate<=20):
            raise ValueError("Alert lacks a valid near-range forecast")
        counts["frames"]+=1
        counts["unavailable"]+=not available
        counts["outside_alert"]+=alert and not expected
        if not expected:
            continue
        counts["near"]+=1
        counts["near_unavailable"]+=not available
        side = by_side[direction]
        side["near"]+=1
        side["unavailable"]+=not available
        hit = alert and forecast["direction"]==direction
        if diagnostics is not None:
            eligible = forecast["direction"] == direction
            passes_presence = available and forecast["presence_score"] >= threshold
            if not available:
                reason = "unavailable"
            elif not eligible:
                reason = "no_direction" if not forecast["direction"] else "opposite_direction"
            elif not passes_presence:
                reason = "presence_rejected"
            elif estimate > 20:
                reason = "estimated_beyond_range"
            else:
                reason = "hit"
            if (reason == "hit") != hit:
                raise ValueError("Forecast alert disagrees with pinned readout rules")
            band = "4-10m" if distance <= 10 else "over-10-20m"
            diagnostics[str(direction)][band][reason] += 1
        side["direction_and_range"]+=hit
        counts["wrong_direction_alert"]+=alert and not hit
        event = events.setdefault((key[0],actual["event"],direction),
                                  dict(near=0,hit=0,within_5m=0,max_correct_warning_distance_m=None))
        event["near"]+=1
        if hit:
            error = estimate-distance
            errors.append(error)
            counts["direction_and_range"]+=1
            counts["within_5m"]+=abs(error)<=5
            side["within_5m"]+=abs(error)<=5
            event["hit"]+=1
            event["within_5m"]+=abs(error)<=5
            last = event["max_correct_warning_distance_m"]
            event["max_correct_warning_distance_m"] = distance if last is None else max(last,distance)
    event_summary=dict(n=len(events),any_hit=sum(e["hit"]>0 for e in events.values()),
                       any_within_5m=sum(e["within_5m"]>0 for e in events.values()))
    output.mkdir(parents=True,exist_ok=False)
    write(output/"result.json",dict(revision=prediction_proof["revision"],seed=None,
        **provenance, counts=dict(counts),by_direction={str(k):dict(v) for k,v in by_side.items()},
        event_summary=event_summary,
        gate_diagnostics=diagnostics,
        events=[dict(run=run,event=event,direction=side,**data) for (run,event,side),data in events.items()],
        distance_mae_m=float(np.mean(np.abs(errors))) if errors else None,
        distance_bias_m=float(np.mean(errors)) if errors else None,
        inputs_sha256={str(p):digest(p) for p in (forecasts,labels,forecasts.with_name("result.json"),
                       labels.with_name("result.json"), Path(__file__), *diagnostic_inputs)},
        termination="frozen forecasts evaluated against separately loaded labels"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("forecasts","labels","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--expected-track", help="Require a verified shared forecast and label track")
    evaluate(**vars(parser.parse_args()))


if __name__=="__main__":
    main()
