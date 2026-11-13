"""Evaluate the frozen fusion with causal termination of near-corner warnings."""
import argparse
import json
from pathlib import Path

from corner_fusion_boundaries import read_motion, transform
from evaluate_fused_corner import frame_key, load_frozen, load_score, read_rows, run as run_fusion
from evaluate_independent_corner import compare
from evaluate_live_corner_geometry import evaluate
from flybrain_drive.ac_camera_archive import CameraArchive
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write


def capture_motion(archives, track, bundles):
    """Read capture clocks and relative motion from the same camera bundles."""
    for run, folder in read(archives).items():
        previous = None
        old_motion = None
        with CameraArchive(folder) as archive:
            if archive.bundle_sha256 != bundles[run]:
                raise ValueError(f"Warning motion bundle differs: {run}")
            for capture in map(json.loads, archive.read("stream.jsonl").splitlines()):
                frame = dict(run=run, frame=capture["capture_id"], generation=capture["generation"],
                             received_s=capture["received"])
                meta = archive.json(Path(capture["file"]).name)
                if (meta["track_id"], meta["index"], meta["generation"]) != (
                        track, frame["frame"], frame["generation"]):
                    raise ValueError(f"Warning motion capture differs: {run}/{frame['frame']}")
                views = capture["views"]
                usable = (len(views) == 4 and {view["view_id"] for view in views} == {"front", "rear", "left", "right"}
                          and not any(r.get(f) for r in (capture, meta) for f in ("ideal_line_shown", "aid_probe_only")))
                motion = read_motion(meta) if usable else None
                step = None
                if motion is not None and old_motion is not None and adjacent(previous, frame):
                    step = transform(motion, old_motion)
                yield frame_key(frame), float(step[1][0]) if step is not None else None
                previous, old_motion = frame, motion


def end_warnings(forecasts, motion, settings):
    previous_side = None
    first_distance = traveled = 0.
    for forecast, (key, step) in zip(forecasts, motion, strict=True):
        if frame_key(forecast) != key:
            raise ValueError("Warning motion and forecasts have different capture order")
        if not forecast["alert"]:
            previous_side = None
            yield forecast
            continue
        side = forecast["direction"]
        distance = forecast["distance_m"]
        if side != previous_side or step is None:
            first_distance, traveled = distance, 0.
        else:
            traveled += step
        previous_side = side
        if (distance <= settings["maximum_current_distance_m"]
                and first_distance-traveled+settings["allowance_m"] < settings["minimum_remaining_m"]):
            # Keep the estimated direction and distance for diagnostics. The
            # common scorer interprets a nonzero direction without alert as abstain.
            yield dict(forecast, alert=False, warning_ended=True)
        else:
            yield forecast


def run(args):
    protocol = load_frozen(args.protocol)
    fusion_protocol_path = Path(protocol["fusion_protocol"])
    fusion_protocol = load_frozen(fusion_protocol_path)
    base_protocol = read(Path(fusion_protocol["base_protocol"]))
    rules = base_protocol["comparison"]
    track = args.track_dir.name
    if track != base_protocol["proposed_track"]:
        raise ValueError("Track differs from the frozen warning-end protocol")
    if args.fusion_replay and not args.base_replay:
        raise ValueError("Cached fusion requires its original base replay")

    args.output.mkdir(parents=True, exist_ok=False)
    fusion = args.fusion_replay or args.output/"fusion"
    if args.fusion_replay is None:
        run_fusion(argparse.Namespace(protocol=fusion_protocol_path, archives=args.archives, track_dir=args.track_dir,
                                      cache=args.cache, output=fusion, base_replay=args.base_replay))
    base = args.base_replay or fusion/"base"
    receipt_path = fusion/"result.json"
    fused = read(receipt_path)
    base_receipt = base/"result.json"
    base_proof = read(base_receipt)
    if (fused["track"] != track or fused["protocol_sha256"] != digest(fusion_protocol_path)
            or fused["base_receipt_sha256"] != digest(base_receipt)
            or base_proof["archives_sha256"] != digest(args.archives)
            or base_proof["admission_rows_sha256"] != digest(base/"admission-rows.jsonl")):
        raise ValueError("Warning-end replay differs from the verified inputs")
    result = dict(track=track, revision=fused["revision"], seed=None,
                  development_rehearsal=fused["development_rehearsal"], admission=fused["admission"],
                  protocol_sha256=digest(args.protocol), fusion_receipt_sha256=digest(receipt_path))
    if fused["status"] == "not_admitted":
        write(args.output/"result.json", dict(result, status="not_admitted", termination="metadata_admission"))
        return
    if fused["status"] != "evaluated":
        raise ValueError("Fusion must complete before warning termination")

    pins = Path(rules["candidate_pins"])
    bundles = fused["camera_bundle_sha256_by_run"]
    original = fusion/"forecasts/rows.jsonl"
    proof = read(original.with_name("result.json"))
    if (fused["forecasts_sha256"] != digest(original) or proof["rows_sha256"] != fused["forecasts_sha256"]
            or proof["verified_track_id"] != track or proof["camera_bundle_sha256_by_run"] != bundles):
        raise ValueError("Unverified input forecasts for warning termination")

    output = args.output/"forecasts"
    output.mkdir()
    rows = list(end_warnings(read_rows(original), capture_motion(args.archives, track, bundles),
                            protocol["warning_end"]))
    jsonl(output/"rows.jsonl", rows)
    ended = [dict(run=row["run"], frame=row["frame"], distance_m=row["distance_m"])
             for row in rows if row.get("warning_ended")]
    jsonl(args.output/"ended-warnings.jsonl", ended)
    proof["inputs_sha256"].update({str(path):digest(path) for path in (original, args.protocol, Path(__file__))})
    proof.update(rows_sha256=digest(output/"rows.jsonl"), termination="frozen warning-end inference completed")
    write(output/"result.json", proof)

    scored = args.output/"score"
    evaluate(argparse.Namespace(output=scored, track_dir=args.track_dir, track=track, common=True,
             pins=pins, forecasts=output/"rows.jsonl", archive_map=args.archives, recordings=[]))
    score_proof = read(scored/"result.json")
    candidate = read_rows(scored/"rows.jsonl")
    parent = load_score(base/"parent/score", base_proof["score_receipt_sha256"]["parent"])
    identity = lambda rows: [(frame_key(row), row["truth"], row["truth_endpoints_m"]) for row in rows]
    admission = identity(read_rows(base/"admission-rows.jsonl"))
    if (score_proof["parameters"] != rules["score_parameters"]
            or score_proof["camera_bundle_sha256_by_run"] != bundles
            or identity(candidate) != admission or identity(parent) != admission):
        raise ValueError("Warning-end scoring differs from the admitted timeline or frozen parameters")
    timelines = {name: {run: [row for row in rows if row["run"] == run] for run in bundles}
                 for name, rows in (("parent", parent), ("candidate", candidate))}
    write(args.output/"result.json", dict(result, status="evaluated", comparison=compare(timelines, rules),
          camera_bundle_sha256_by_run=bundles, score_receipt_sha256=digest(scored/"result.json"),
          forecasts_sha256=proof["rows_sha256"], ended_warnings=len(ended),
          ended_warnings_sha256=digest(args.output/"ended-warnings.jsonl"), termination="frozen_comparison_end"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "archives", "track-dir", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--base-replay", type=Path)
    parser.add_argument("--fusion-replay", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
