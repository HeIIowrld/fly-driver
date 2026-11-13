"""Compare a contrast detector without its absolute pavement chroma ceiling."""
import argparse
import ast
from pathlib import Path

from corner_fusion_boundaries import read_boundaries
from evaluate_fused_corner import load_frozen, load_score, merge_forecasts, read_rows, visual_evidence
from evaluate_fused_corner_warning_end import capture_motion, end_warnings
from evaluate_independent_corner import compare
from evaluate_live_corner_geometry import evaluate
from train_vision_teacher_line import digest, jsonl, read, write


def alternative_boundaries():
    path = Path("scripts/corner_fusion_boundaries.py")
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    surface = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SurfaceContrast")
    body = ast.get_source_segment(source, surface)
    gate = "support & (inner_chroma <= 13) & (outer_chroma-inner_chroma >= 6)"
    assert body.count(gate) == 1
    namespace = read_boundaries.__globals__.copy()
    exec(compile(body.replace(gate, "support & (outer_chroma-inner_chroma >= 6)"), str(path), "exec"), namespace)
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_boundaries")
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["read_boundaries"]


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "archives", "track-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    track = args.track_dir.name
    root = Path("runs")
    old = root/"vision-corner-fusion-runner-20261111-01"/"zandvoort-protocol.json"
    fusion = load_frozen(old)
    settings = load_frozen(Path(fusion["development_pins"]))
    warning = load_frozen(root/"vision-corner-warning-end-runner-20261111-01"/"zandvoort-protocol.json")
    base = read(args.base/"result.json")
    assert base["status"] == "evaluated" and base["track"] == track
    assert digest(args.archives) == base["archives_sha256"]
    rules = read(Path(fusion["base_protocol"]))["comparison"]
    candidate_pins = Path(rules["candidate_pins"])
    forecast_path = args.base/"candidate/forecasts/rows.jsonl"
    proof = read(forecast_path.with_name("result.json"))
    assert proof["rows_sha256"] == digest(forecast_path)
    scores = visual_evidence(args.base/"features", proof, candidate_pins, Path(settings["visual_presence_model"]))
    bundles = base["camera_bundle_sha256_by_run"]
    altered = list(alternative_boundaries()(args.archives, bundles, track, Path(fusion["camera_config"]), settings))
    original = read_rows(forecast_path)
    merged = list(merge_forecasts(original, altered, scores, settings))
    rows = list(end_warnings(merged, capture_motion(args.archives, track, bundles), warning["warning_end"]))
    forecasts = args.output/"forecasts"
    forecasts.mkdir(parents=True, exist_ok=False)
    jsonl(forecasts/"rows.jsonl", rows)
    proof.update(rows_sha256=digest(forecasts/"rows.jsonl"), termination="development_relative_chroma_replay")
    write(forecasts/"result.json", proof)
    scored = args.output/"score"
    evaluate(argparse.Namespace(output=scored, track_dir=args.track_dir, track=track, common=True,
              pins=candidate_pins, forecasts=forecasts/"rows.jsonl", archive_map=args.archives, recordings=[]))
    parent = load_score(args.base/"parent/score", base["score_receipt_sha256"]["parent"])
    candidate = read_rows(scored/"rows.jsonl")
    timelines = {name:{tag:[row for row in rows if row["run"] == tag] for tag in bundles}
                 for name, rows in (("parent", parent), ("candidate", candidate))}
    reference = { (r["run"], r["frame"]):r for r in candidate}
    approach = [(reference[key], choice) for key, choice, _ in altered if reference[key]["truth"] == "approach"]
    changed = sum(not before["alert"] and after["alert"] for before, after in zip(original, merged, strict=True))
    result = dict(track=track, stage="development_replay", geometry_on_approach=sum(c is not None for _,c in approach),
        geometry_correct_side=sum(c is not None and str(c[0]) == r["label"] for r,c in approach),
        added_alerts=changed, approach_frames=len(approach), candidate_promoted=False,
        comparison=compare(timelines,rules), score_sha256=digest(scored/"rows.jsonl"),
        inputs={str(p):digest(p) for p in (Path(__file__), old, forecast_path, args.base/"result.json", args.archives,
                Path("scripts/corner_fusion_boundaries.py"))})
    write(args.output/"result.json", result)
    print(result["comparison"], result["geometry_on_approach"], result["geometry_correct_side"], changed)


if __name__ == "__main__":
    run()
