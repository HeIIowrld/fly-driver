"""Run a frozen parent/candidate comparison on admitted independent camera bundles."""
import argparse
import json
from collections import Counter, deque
from pathlib import Path
from statistics import mean

from evaluate_live_corner_geometry import common_rows, evaluate, load_track_truth
from flybrain_drive.ac_camera_archive import CameraArchive
from infer_frozen_next_corner import infer
from score_corner_captures import adjacent, forecast_state, summarize
from train_vision_teacher_line import digest, jsonl, read, write


def admission(rows, requirements):
    corners = {label: {row["physical_corner"] for row in rows
                      if row["truth"] == "approach" and row["label"] == label
                      and row["camera_complete"]}
               for label in ("1", "2")}
    truth = Counter(row["truth"] for row in rows if row["camera_complete"])
    missing = [label for label, found in corners.items()
               if len(found) < requirements["minimum_distinct_physical_corners_per_side"]]
    if requirements["requires_straight_and_out_of_range"]:
        missing.extend(category for category in ("straight", "out_of_range") if not truth[category])
    return dict(frames=len(rows), truth=truth, corners={label: sorted(found)
                for label, found in corners.items()}, missing=missing, passed=bool(rows) and not missing)


def compare(timelines, rules):
    summaries = {name: {run: summarize(rows) for run, rows in records.items()}
                 for name, records in timelines.items()}
    metrics, pairs = {}, {}
    for name, reports in summaries.items():
        metrics[name] = dict(
            left=sum(r["by_direction"]["left"]["direction_hits"] for r in reports.values()),
            right=sum(r["by_direction"]["right"]["direction_hits"] for r in reports.values()),
            joint=sum(side["joint_hits"] for r in reports.values() for side in r["by_direction"].values()),
            outside=sum(r["false_alerts"]["outside_band"]["frames"] for r in reports.values()),
            missing=sum(r["outputs"]["no_output"] for r in reports.values()))
        pairs[name] = {(run, p["first_frame"], p["last_frame"]): p
                       for run, report in reports.items() for p in report["distance_change"]["rows"]}
    shared = sorted(pairs["parent"].keys() & pairs["candidate"].keys())
    closing = [key for key in shared if pairs["parent"][key]["truth_delta_m"] < 0]
    distances = {}
    for name, lookup in pairs.items():
        distances[name] = dict(
            mae_m=mean(abs(lookup[key]["delta_error_m"]) for key in shared) if shared else None,
            receding=sum(lookup[key]["forecast_delta_m"] > 0 for key in closing))
    events = [e for r in summaries["candidate"].values() for e in r["events"]]
    stable = Counter(e["physical_corner"] for e in events if e["first_stable_frame"] is not None)
    parent, candidate = metrics["parent"], metrics["candidate"]
    gates = dict(
        left=candidate["left"]-parent["left"] >= rules["minimum_left_direction_hit_increase"],
        right=parent["right"]-candidate["right"] <= rules["maximum_right_direction_hit_loss"],
        joint=parent["joint"]-candidate["joint"] <= rules["maximum_joint_hit_loss"],
        outside=candidate["outside"]-parent["outside"] <= rules["maximum_outside_alert_increase"],
        missing=candidate["missing"]-parent["missing"] <= rules["maximum_new_missing_decisions"],
        stable=bool(events) and all(stable[e["physical_corner"]] >= rules["minimum_stable_warning_per_physical_corner"]
                                    for e in events),
        distance_change=bool(shared) and distances["candidate"]["mae_m"] < distances["parent"]["mae_m"],
        receding=bool(closing) and distances["candidate"]["receding"] <= distances["parent"]["receding"])
    return dict(metrics=metrics, shared_pairs=len(shared), shared_closing_pairs=len(closing),
                distances=distances, gates=gates, passed=all(gates.values()))


def adjust_visual_distance(rows, captures, cap):
    """Apply the frozen correction using previous visual distances and capture clocks."""
    history = deque(maxlen=3)
    previous = None
    previous_raw = previous_filtered = None
    for row in rows:
        key = row["run"], row["frame"]
        frame = captures[key]
        state = forecast_state(row)
        output = dict(row)
        if state not in ("left", "right"):
            previous = None
            history.clear()
        else:
            raw = row["distance_m"]
            current = dict(frame, state=state)
            consecutive = previous is not None and adjacent(previous, current) and previous["state"] == state
            if not consecutive:
                history.clear()
            else:
                if raw > previous_filtered and history:
                    drift = mean(history)
                    if drift > 0:
                        output["distance_m"] = max(raw-cap, previous_filtered-drift, 0.)
                history.append(max(0., previous_raw-raw))
            previous, previous_raw, previous_filtered = current, raw, output["distance_m"]
        yield output


def adjust_forecasts(forecasts, output, admission, protocol, cap):
    original = forecasts / "rows.jsonl"
    receipt = forecasts / "result.json"
    proof = read(receipt)
    if digest(original) != proof["rows_sha256"]:
        raise ValueError("Raw candidate forecast digest differs")
    # Expose only timing and identity to the distance rule.
    captures = {(r["run"], r["frame"]): {key: r[key] for key in ("run", "frame", "generation", "received_s")}
                for r in admission}
    rows = map(json.loads, original.read_text(encoding="utf8").splitlines())
    output.mkdir(exist_ok=False)
    jsonl(output/"rows.jsonl", adjust_visual_distance(rows, captures, cap))
    proof.update(rows_sha256=digest(output/"rows.jsonl"), raw_forecast_rows_sha256=digest(original),
                 distance_adjustment=dict(cap_m=cap, history_intervals=3),
                 raw_forecast_receipt_sha256=digest(receipt), protocol_sha256=digest(protocol))
    write(output/"result.json", proof)


def run(args):
    protocol = read(args.protocol)
    track = args.track_dir.name
    if track != protocol["proposed_track"] or track in protocol["excluded_development_tracks"]:
        raise ValueError("Track is outside the frozen independent protocol")
    adjustment = protocol["comparison"].get("candidate_visual_distance")
    if adjustment is not None and adjustment != {"cap_m": .05, "history_intervals": 3}:
        raise ValueError("Unrecognized frozen visual distance adjustment")
    for path, expected in protocol["hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen protocol input differs: {path}")
    geometry = args.track_dir / "ai/fast_lane.ai"
    if protocol["hashes"].get(str(geometry.resolve())) != digest(geometry):
        raise ValueError("Installed geometry differs from the frozen track reference")
    mapping = read(args.archives)
    truth, _, _ = load_track_truth(geometry)
    admitted = []
    bundles = {}
    for tag, directory in mapping.items():
        with CameraArchive(directory) as archive:
            if not archive.bundle_sha256 or archive.bundle_sha256 in bundles.values():
                raise ValueError("Expected unique hashed camera bundles")
            bundles[tag] = archive.bundle_sha256
            admitted.extend(common_rows(tag, archive, {}, track, truth))

    args.output.mkdir(parents=True, exist_ok=False)
    check = admission(admitted, protocol["data_admission"])
    receipt = dict(track=track, revision=protocol["revision"], seed=None, protocol_stage=protocol["stage"],
                   protocol_sha256=digest(args.protocol), archives_sha256=digest(args.archives),
                   program_sha256=digest(Path(__file__)), camera_bundle_sha256_by_run=bundles, admission=check)
    jsonl(args.output/"admission-rows.jsonl", admitted)
    receipt["admission_rows_sha256"] = digest(args.output/"admission-rows.jsonl")
    if not check["passed"]:
        write(args.output/"result.json", dict(receipt, status="not_admitted", termination="metadata_admission"))
        return

    # Import GPU dependencies only after metadata admission.
    from extract_frozen_corner_archive import run as extract

    features = args.output / "features"
    extract(args.archives, None, args.cache, features, batch_pair=True, expected_track=track)
    rules = protocol["comparison"]
    timelines = {}
    results = {}
    admission_keys = [(r["run"], r["frame"], r["truth"], r["truth_endpoints_m"]) for r in admitted]
    for name in ("parent", "candidate"):
        pins = Path(rules[f"{name}_pins"])
        forecasts = args.output / name / "forecasts"
        infer(pins, features/"index.jsonl", features/"original/features.npz",
              features/"reflected/features.npz", forecasts, Path(rules["joint_distance_pins"]))
        if name == "candidate" and adjustment is not None:
            adjusted = args.output / name / "adjusted-forecasts"
            adjust_forecasts(forecasts, adjusted, admitted, args.protocol, adjustment["cap_m"])
            forecasts = adjusted
        scored = args.output / name / "score"
        evaluate(argparse.Namespace(output=scored, track_dir=args.track_dir, track=track, common=True,
                 pins=pins, forecasts=forecasts/"rows.jsonl", archive_map=args.archives, recordings=[]))
        proof = read(scored/"result.json")
        rows = list(map(json.loads, (scored/"rows.jsonl").read_text(encoding="utf8").splitlines()))
        if (proof["camera_bundle_sha256_by_run"] != bundles or proof["parameters"] != rules["score_parameters"]
                or [(r["run"], r["frame"], r["truth"], r["truth_endpoints_m"]) for r in rows] != admission_keys):
            raise ValueError("Scored input differs from the admitted timeline or frozen rules")
        timelines[name] = {tag: [r for r in rows if r["run"] == tag] for tag in mapping}
        results[name] = digest(scored/"result.json")
    comparison = compare(timelines, rules)
    write(args.output/"result.json", dict(receipt, status="evaluated", comparison=comparison,
          score_receipt_sha256=results, termination="frozen_comparison_end"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "archives", "track-dir", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
