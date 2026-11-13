"""Run the frozen alert-recovery candidate on admitted four-view captures."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_fused_corner import frame_key, load_frozen, load_score, merge_forecasts, read_rows, visual_evidence
from evaluate_fused_corner_warning_end import capture_motion, end_warnings
from evaluate_independent_corner import compare, run as run_base
from evaluate_live_corner_geometry import evaluate
from infer_frozen_next_corner import contrast, load_features
from relative_corner_boundaries import read_boundaries
from train_vision_teacher_line import digest, jsonl, read, write


def normalized_moments(features):
    grid = features.astype(float).reshape(-1, 3, 4, 8, 4, 4)
    average = grid.mean(axis=(-1, -2))
    centered = grid-average[..., None, None]
    rms = np.sqrt(np.mean(grid**2, axis=(1, 3, 4, 5), keepdims=True)).clip(.025)
    average = (grid/rms).mean(axis=(-1, -2))
    centered /= rms
    spread = np.sqrt(np.mean(centered**2, axis=(-1, -2)))
    quadrants = centered.reshape(-1, 3, 4, 8, 2, 2, 2, 2).mean(axis=(5, 7))
    return np.concatenate([item.reshape(len(features), -1) for item in (average, spread, quadrants)], axis=1)


def estimate_components(feature_dir, direction_path, distance_path, limits):
    index = feature_dir/"index.jsonl"
    rows = read_rows(index)
    sha = digest(index)
    a, av, _ = load_features(feature_dir/"original/features.npz", sha, len(rows))
    b, bv, _ = load_features(feature_dir/"reflected/features.npz", sha, len(rows))
    available = av & bv
    if not available.any():
        return {}
    x, y = a[available], b[available]
    direction = .5*(normalized_moments(x)-normalized_moments(y))
    distance = .5*(contrast(x)+contrast(y))
    with np.load(direction_path, allow_pickle=False) as weights:
        signed = direction/weights["scale"] @ weights["coefficients"]
    with np.load(distance_path, allow_pickle=False) as weights:
        metric = ((distance-weights["mean"])/weights["scale"] @ weights["coefficients"] + weights["offset"])
    metric = np.clip(metric, *limits)
    if not np.isfinite(signed).all() or not np.isfinite(metric).all():
        raise ValueError("Nonfinite recovery readout")
    keys = (frame_key(r) for r, ok in zip(rows, available, strict=True) if ok)
    return {key:(float(s), float(d)) for key, s, d in zip(keys, signed, metric, strict=True)}


def recover(parent, candidate, boundaries, visual, components, threshold, margin, gain):
    for original, current, (key, choice, _) in zip(parent, candidate, boundaries, strict=True):
        if key != frame_key(original) or key != frame_key(current):
            raise ValueError("Recovery forecast inputs differ in capture order")
        component = components.get(key)
        if component is None:
            yield current
            continue
        signed, estimate = component
        if current["alert"]:
            side, distance = current["direction"], current["distance_m"]
            observed = visual.get(key)
            if (choice and choice[0] and choice[0] != side and observed is not None
                    and observed[0] >= threshold-1e-12 and 0 <= observed[1] <= 20):
                side, distance = choice[0], observed[1]
            yield dict(current, direction=side, distance_m=distance+gain*(estimate-distance))
        elif original["alert"] and signed*(1 if original["direction"] == 1 else -1) > margin:
            yield dict(original, restored_prior=True)
        else:
            yield current


def frozen_components(path):
    receipt = read(path)
    for name, value in receipt["source_hashes"].items():
        if digest(Path(name)) != value:
            raise ValueError(f"Component source differs: {name}")
    return receipt


def run(args):
    protocol = load_frozen(args.protocol)
    component_path = Path(protocol["component_pins"])
    components = frozen_components(component_path)
    distance_pins = load_frozen(Path(components["active_alerts"]["distance_pins"]))
    fusion_pins = load_frozen(Path(protocol["boundary_fusion_pins"]))
    base_path = Path(protocol["base_protocol"])
    historical = protocol.get("archived_base_entrypoint")
    if historical and (not protocol["development_rehearsal"] or not args.base_replay):
        raise ValueError("Archived entrypoint requires a cached development replay")
    entrypoint = Path(historical) if historical else Path("scripts/evaluate_independent_corner.py")
    base_config = load_frozen(base_path, entrypoint if historical else None)
    rules = base_config["comparison"]
    track = args.track_dir.name
    if track != base_config["proposed_track"]:
        raise ValueError("Track differs from the recovery protocol")
    if not protocol["development_rehearsal"]:
        excluded = set(base_config["excluded_development_tracks"]) | set(components["training_tracks"])
        if track in excluded or base_config["data_admission"] != components["data_admission"]:
            raise ValueError("Recovery requires an unseen track with frozen admission criteria")
    if any(rules[key] != value for key, value in components["inherited_gates"].items()):
        raise ValueError("Recovery comparison criteria changed")

    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_replay or args.output/"base"
    if not args.base_replay:
        run_base(argparse.Namespace(protocol=base_path, archives=args.archives, track_dir=args.track_dir,
                                    cache=args.cache, output=base))
    base_receipt = base/"result.json"
    proof = read(base_receipt)
    if (proof["track"] != track or proof["program_sha256"] != digest(entrypoint)
            or proof["protocol_sha256"] != digest(base_path) or proof["archives_sha256"] != digest(args.archives)
            or proof["admission_rows_sha256"] != digest(base/"admission-rows.jsonl")):
        raise ValueError("Base recovery provenance differs")
    receipt = dict(track=track, seed=None, revision=fusion_pins["revision"], admission=proof["admission"],
                   development_rehearsal=protocol["development_rehearsal"],
                   protocol_sha256=digest(args.protocol), base_receipt_sha256=digest(base_receipt))
    if proof["status"] == "not_admitted":
        write(args.output/"result.json", dict(receipt, status="not_admitted", termination="metadata_admission"))
        return
    if proof["status"] != "evaluated":
        raise ValueError("Base comparison did not complete")

    bundles = proof["camera_bundle_sha256_by_run"]
    sources = {}
    forecast_proof = None
    for name in ("parent", "candidate"):
        folder = base/name/"forecasts"
        forecast_proof = read(folder/"result.json")
        source = folder/"rows.jsonl"
        pins = Path(rules[f"{name}_pins"])
        if (forecast_proof["rows_sha256"] != digest(source)
                or forecast_proof["verified_track_id"] != track
                or forecast_proof["camera_bundle_sha256_by_run"] != bundles
                or forecast_proof["inputs_sha256"].get(str(pins)) != digest(pins)):
            raise ValueError(f"Unverified base forecast: {name}")
        sources[name] = read_rows(source)
    features = base/"features"
    visual = visual_evidence(features, forecast_proof, Path(rules["candidate_pins"]),
                             Path(fusion_pins["visual_presence_model"]))
    values = estimate_components(features, Path(components["direction_model"]), Path(distance_pins["model"]),
                                 distance_pins["prediction_clip_m"])
    boundaries = list(read_boundaries(args.archives, bundles, track, Path(protocol["camera_config"]), fusion_pins))
    merged = merge_forecasts(sources["candidate"], boundaries, visual, fusion_pins)
    warnings = end_warnings(merged, capture_motion(args.archives, track, bundles), protocol["warning_end"])
    output_rows = list(recover(sources["parent"], warnings, boundaries, visual, values,
                     fusion_pins["visual_presence_threshold"], components["direction_margin"],
                     distance_pins["publish_gain"]))
    output = args.output/"forecasts"
    output.mkdir()
    jsonl(output/"rows.jsonl", output_rows)
    forecast_proof["inputs_sha256"].update({str(p):digest(p) for p in (
        args.protocol, component_path, Path(components["direction_model"]), Path(distance_pins["model"]),
        base/"parent/forecasts/rows.jsonl", base/"candidate/forecasts/rows.jsonl", Path(__file__))})
    forecast_proof.update(rows_sha256=digest(output/"rows.jsonl"), termination="frozen alert recovery completed")
    write(output/"result.json", forecast_proof)

    scored = args.output/"score"
    evaluate(argparse.Namespace(output=scored, track_dir=args.track_dir, track=track, common=True,
             pins=Path(rules["candidate_pins"]), forecasts=output/"rows.jsonl", archive_map=args.archives, recordings=[]))
    score_proof = read(scored/"result.json")
    parent = load_score(base/"parent/score", proof["score_receipt_sha256"]["parent"])
    candidate = read_rows(scored/"rows.jsonl")
    admission = read_rows(base/"admission-rows.jsonl")
    identity = lambda rows: [(frame_key(r), r["truth"], r["truth_endpoints_m"]) for r in rows]
    if (score_proof["parameters"] != rules["score_parameters"]
            or score_proof["camera_bundle_sha256_by_run"] != bundles
            or identity(parent) != identity(admission) or identity(candidate) != identity(admission)):
        raise ValueError("Recovery scoring differs from admitted capture timeline")
    timelines = {name:{tag:[r for r in rows if r["run"] == tag] for tag in bundles}
                 for name, rows in (("parent", parent), ("candidate", candidate))}
    write(args.output/"result.json", dict(receipt, status="evaluated", comparison=compare(timelines, rules),
        restored=sum(bool(r.get("restored_prior")) for r in output_rows),
        camera_bundle_sha256_by_run=bundles, forecasts_sha256=forecast_proof["rows_sha256"],
        score_receipt_sha256=digest(scored/"result.json"), termination="frozen_comparison_end"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "archives", "track-dir", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--base-replay", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
