"""Evaluate the frozen boundary and FlyVis fusion on a camera archive."""
import argparse
import json
from pathlib import Path

import numpy as np

from corner_fusion_boundaries import read_boundaries
from evaluate_independent_corner import compare, run as run_base
from evaluate_live_corner_geometry import evaluate
from infer_frozen_next_corner import contrast, load_features, load_readout
from train_vision_teacher_line import digest, jsonl, read, write


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def frame_key(row):
    return row["run"], row["frame"]


def load_frozen(path, archived_entrypoint=None):
    settings = read(path)
    for name, expected in settings["hashes"].items():
        source = Path(name)
        if archived_entrypoint is not None and source == Path("scripts/evaluate_independent_corner.py"):
            source = archived_entrypoint
        if digest(source) != expected:
            raise ValueError(f"Frozen fusion input differs: {name}")
    return settings


def visual_evidence(features, base_receipt, pins, model):
    index = features/"index.jsonl"
    keys = [frame_key(row) for row in read_rows(index)]
    original = features/"original/features.npz"
    reflected = features/"reflected/features.npz"
    checksum = digest(index)
    direct, a, dp = load_features(original, checksum, len(keys))
    mirror, b, rp = load_features(reflected, checksum, len(keys))
    for branch, proof in ((original, dp), (reflected, rp)):
        for path in (branch, branch.with_name("flyvis-result.json"), index):
            if base_receipt["inputs_sha256"].get(str(path)) != digest(path):
                raise ValueError(f"Feature input differs from base readout: {path}")
        if proof["verified_track_id"] != base_receipt["verified_track_id"]:
            raise ValueError("Feature track differs from base readout")
    valid = a & b
    with np.load(model, allow_pickle=False) as weights:
        coefficient = weights["presence_coef"]
        intercept = float(weights["presence_intercept"])
    x, y = direct[valid], mirror[valid]
    presence = .5 * ((contrast(x) @ coefficient + intercept) + (contrast(y) @ coefficient + intercept))
    predict, _ = load_readout(read(pins))
    distance = predict(x, y)[1] if len(x) else np.empty(0)
    return {key: (float(p), float(d)) for key, p, d in
            zip((key for key, available in zip(keys, valid, strict=True) if available),
                presence, distance, strict=True)}


def merge_forecasts(rows, boundaries, evidence, pins):
    travel_total = 0.
    previous_geometry = None
    previous_alert = None
    for base, (key, choice, travel) in zip(rows, boundaries, strict=True):
        if frame_key(base) != key:
            raise ValueError("Boundary stream differs from the base capture order")
        if (travel is not None and travel >= .10 and choice and choice[0] and previous_geometry
                and previous_geometry[0] == choice[0]
                and abs(choice[1]-(previous_geometry[1]-travel)) <= pins["maximum_position_innovation_m"]):
            travel_total += travel
        else:
            travel_total = 0.
        previous_geometry = choice
        row = dict(base)
        visual = evidence.get(key)
        # The development threshold is an observed float64 score. Ignore only
        # rounding differences from the number of rows in the BLAS batch.
        if (not row["alert"] and travel_total >= pins["minimum_local_travel_m"] and visual
                and visual[0] >= pins["visual_presence_threshold"] - 1e-12 and 0 <= visual[1] <= 20):
            row.update(available=True, direction=choice[0], distance_m=visual[1], presence_score=visual[0], alert=True)
        if row["alert"]:
            raw = row["distance_m"]
            if (previous_alert is not None and travel is not None and travel > 0
                    and previous_alert["direction"] == row["direction"]):
                projected = max(0., previous_alert["distance_m"]-travel)
                if raw > projected:
                    row["distance_m"] = max(raw-pins["maximum_distance_reduction_m"], projected)
            previous_alert = row
        else:
            previous_alert = None
        yield row


def load_score(directory, checksum):
    receipt = directory/"result.json"
    proof = read(receipt)
    rows = directory/"rows.jsonl"
    if digest(receipt) != checksum or digest(rows) != proof["rows_sha256"]:
        raise ValueError(f"Base score provenance differs: {directory}")
    return read_rows(rows)


def run(args):
    protocol = load_frozen(args.protocol)
    development = load_frozen(Path(protocol["development_pins"]))
    base_protocol_path = Path(protocol["base_protocol"])
    historical = protocol.get("archived_base_entrypoint")
    if historical and (not protocol["development_rehearsal"] or not args.base_replay):
        raise ValueError("Archived base entrypoint is allowed only with its cached development rehearsal")
    archived_entrypoint = Path(historical) if historical else None
    base_protocol = load_frozen(base_protocol_path, archived_entrypoint)
    base_entrypoint = archived_entrypoint or Path("scripts/evaluate_independent_corner.py")
    rules = base_protocol["comparison"]
    track = args.track_dir.name
    if track != base_protocol["proposed_track"]:
        raise ValueError("Track differs from the pinned fusion protocol")
    if not protocol["development_rehearsal"]:
        excluded = development["development_tracks"] + [development["prior_independent_result_track"]]
        if track in excluded or base_protocol["data_admission"] != development["independent_data_admission"]:
            raise ValueError("Expected fresh track and the unchanged independent admission criteria")
    if any(rules[name] != value for name, value in development["inherited_independent_gates"].items()):
        raise ValueError("Fusion changes the inherited independent comparison criteria")
    if development["geometry"] != "whole_shape_min":
        raise ValueError("Unsupported frozen geometry mode")
    for name in ("parent_pins", "candidate_pins", "joint_distance_pins"):
        path = Path(rules[name])
        if development["hashes"].get(str(path)) != digest(path):
            raise ValueError(f"Fusion base differs from development pins: {name}")

    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_replay if args.base_replay else args.output/"base"
    if args.base_replay is None:
        run_base(argparse.Namespace(protocol=base_protocol_path, archives=args.archives,
                                    track_dir=args.track_dir, cache=args.cache, output=base))
    base_receipt_path = base/"result.json"
    base_proof = read(base_receipt_path)
    if (base_proof["program_sha256"] != digest(base_entrypoint) or base_proof["track"] != track or base_proof["protocol_sha256"] != digest(base_protocol_path)
            or base_proof["archives_sha256"] != digest(args.archives)
            or base_proof["admission_rows_sha256"] != digest(base/"admission-rows.jsonl")):
        raise ValueError("Base admission differs from this protocol and archive map")
    receipt = dict(track=track, development_rehearsal=protocol["development_rehearsal"], seed=None,
                   revision=development["revision"], protocol_sha256=digest(args.protocol),
                   base_receipt_sha256=digest(base_receipt_path), admission=base_proof["admission"])
    if base_proof["status"] == "not_admitted":
        write(args.output/"result.json", dict(receipt, status="not_admitted", termination="metadata_admission"))
        return
    if base_proof["status"] != "evaluated":
        raise ValueError("Base comparison must complete before fusion")

    candidate_pins = Path(rules["candidate_pins"])
    base_forecasts = base/"candidate/forecasts"
    proof = read(base_forecasts/"result.json")
    rows_path = base_forecasts/"rows.jsonl"
    bundles = base_proof["camera_bundle_sha256_by_run"]
    if (proof["rows_sha256"] != digest(rows_path) or proof["verified_track_id"] != track
            or proof["camera_bundle_sha256_by_run"] != bundles
            or proof["inputs_sha256"].get(str(candidate_pins)) != digest(candidate_pins)):
        raise ValueError("Base forecasts differ from the verified fusion inputs")
    evidence = visual_evidence(base/"features", proof, candidate_pins, Path(development["visual_presence_model"]))
    boundaries = read_boundaries(args.archives, bundles, track, Path(protocol["camera_config"]), development)
    forecasts = args.output/"forecasts"
    forecasts.mkdir()
    outputs = list(merge_forecasts(read_rows(rows_path), boundaries, evidence, development))
    jsonl(forecasts/"rows.jsonl", outputs)
    proof["inputs_sha256"].update({str(path):digest(path) for path in
                                  (rows_path, args.protocol, Path(protocol["development_pins"]), Path(__file__))})
    proof.update(rows_sha256=digest(forecasts/"rows.jsonl"), termination="frozen boundary fusion completed")
    write(forecasts/"result.json", proof)

    scored = args.output/"score"
    evaluate(argparse.Namespace(output=scored, track_dir=args.track_dir, track=track, common=True,
             pins=candidate_pins, forecasts=forecasts/"rows.jsonl", archive_map=args.archives, recordings=[]))
    score_proof = read(scored/"result.json")
    candidate = read_rows(scored/"rows.jsonl")
    parent = load_score(base/"parent/score", base_proof["score_receipt_sha256"]["parent"])
    admission = read_rows(base/"admission-rows.jsonl")
    identity = lambda rows: [(frame_key(row), row["truth"], row["truth_endpoints_m"]) for row in rows]
    if (score_proof["parameters"] != rules["score_parameters"]
            or score_proof["camera_bundle_sha256_by_run"] != bundles
            or identity(candidate) != identity(admission) or identity(parent) != identity(admission)):
        raise ValueError("Fusion scoring differs from the admitted timeline or frozen parameters")
    timelines = {name: {run: [row for row in rows if row["run"] == run] for run in bundles}
                 for name, rows in (("parent", parent), ("candidate", candidate))}
    write(args.output/"result.json", dict(receipt, status="evaluated", comparison=compare(timelines, rules),
          camera_bundle_sha256_by_run=bundles, score_receipt_sha256=digest(scored/"result.json"),
          forecasts_sha256=proof["rows_sha256"], termination="frozen_fusion_comparison_end"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "archives", "track-dir", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--base-replay", type=Path, help="Reuse a matching completed base comparison")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
