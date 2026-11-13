"""Attribute relative-chroma recovery and test one-gate changes on development captures."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path

from evaluate_fused_corner import frame_key, load_frozen, merge_forecasts, read_rows, visual_evidence
from evaluate_fused_corner_warning_end import capture_motion, end_warnings
from score_corner_captures import forecast_state, summarize
from train_vision_teacher_line import digest, jsonl, read, write

ROOT = Path("runs")
RELATIVE = ROOT/"vision-corner-surface-contrast-audit-20261111-01"
MODULE = RELATIVE/"replay_relative.py"
SPEC = importlib.util.spec_from_file_location("relative_chroma_replay", MODULE)
RELATIVE_REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELATIVE_REPLAY)
OUT = Path(__file__).parent
PROTOCOL = ROOT/"vision-corner-fusion-runner-20261111-01"/"zandvoort-protocol.json"
WARNING = ROOT/"vision-corner-warning-end-runner-20261111-01"/"zandvoort-protocol.json"


def gate_rows(geometry, visual, settings):
    previous = None
    accumulated = 0.
    for key, choice, travel in geometry:
        move = travel is not None and travel >= .10
        same = bool(choice and choice[0] and previous and choice[0] == previous[0])
        error = abs(choice[1]-(previous[1]-travel)) if move and same else None
        aligned = error is not None and error <= settings["maximum_position_innovation_m"]
        accumulated = accumulated+travel if aligned else 0.
        score = visual.get(key)
        yield dict(run=key[0], frame=key[1], choice=choice, travel=travel, same_side_before=same,
                   alignment_error_m=error, accumulated_travel_m=accumulated, visual=score,
                   movement=accumulated >= settings["minimum_local_travel_m"],
                   presence=bool(score and score[0] >= settings["visual_presence_threshold"]-1e-12),
                   ordinal_range=bool(score and 0 <= score[1] <= 20))
        previous = choice


def with_extra_alerts(rows, gates, mode):
    """Diagnostic only: change one condition after the frozen replay."""
    for row, gate in zip(rows, gates, strict=True):
        assert frame_key(row) == frame_key(gate)
        choice = gate["choice"]
        if not row["alert"] and not row.get("warning_ended") and choice and choice[0]:
            movement, presence, in_range = (gate[name] for name in ("movement", "presence", "ordinal_range"))
            allowed = ((mode == "skip_movement" or movement) and (mode == "skip_presence" or presence)
                       and in_range)
            if allowed:
                score, distance = gate["visual"]
                yield dict(row, available=True, alert=True, direction=choice[0], distance_m=distance,
                           presence_score=score)
                continue
        yield row


def score_rows(labels, predictions):
    results = {}
    for original, prediction in zip(labels, predictions, strict=True):
        assert frame_key(original) == frame_key(prediction)
        state = forecast_state(prediction)
        distance = prediction["distance_m"]
        hit = original["truth"] == "approach" and state == {"1":"left", "2":"right"}[original["label"]]
        errors = [distance-actual for actual in original["truth_endpoints_m"]] if hit else []
        row = dict(original, state=state, direction_hit=hit, predicted_distance_m=distance,
                   joint_hit=bool(errors) and max(map(abs, errors)) <= 5.,
                   signed_distance_error_m=sum(errors)/len(errors) if errors else None,
                   absolute_distance_error_m=sum(map(abs, errors))/len(errors) if errors else None)
        results.setdefault(row["run"], []).append(row)
    return {run:summarize(rows) for run, rows in results.items()}


def counts(scores):
    return dict(left=sum(r["by_direction"]["left"]["direction_hits"] for r in scores.values()),
                right=sum(r["by_direction"]["right"]["direction_hits"] for r in scores.values()),
                joint=sum(side["joint_hits"] for r in scores.values() for side in r["by_direction"].values()),
                outside=sum(r["false_alerts"]["outside_band"]["frames"] for r in scores.values()),
                stable_leads_m=[e["first_stable_lead_m"] for r in scores.values() for e in r["events"]
                                if e["first_stable_frame"] is not None])


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "archives", "track"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    base, archives, track = Path(args.base), Path(args.archives), args.track
    proof = read(base/"result.json")
    assert proof["status"] == "evaluated" and proof["track"] == track and proof["archives_sha256"] == digest(archives)
    config = load_frozen(PROTOCOL)
    settings = load_frozen(Path(config["development_pins"]))
    warning = load_frozen(WARNING)["warning_end"]
    candidate_pins = Path(read(Path(config["base_protocol"]))["comparison"]["candidate_pins"])
    original_path = base/"candidate/forecasts/rows.jsonl"
    original_proof = read(original_path.with_name("result.json"))
    assert digest(original_path) == original_proof["rows_sha256"]
    original = read_rows(original_path)
    visual = visual_evidence(base/"features", original_proof, candidate_pins, Path(settings["visual_presence_model"]))
    geometry = list(RELATIVE_REPLAY.alternative_boundaries()(archives, proof["camera_bundle_sha256_by_run"],
                       track, Path(config["camera_config"]), settings))
    gates = list(gate_rows(geometry, visual, settings))
    motion = list(capture_motion(archives, track, proof["camera_bundle_sha256_by_run"]))
    reference_path = RELATIVE/track/"forecasts/rows.jsonl"
    reference_proof = read(reference_path.with_name("result.json"))
    assert digest(reference_path) == reference_proof["rows_sha256"]
    reference = read_rows(reference_path)
    reproduced = list(end_warnings(merge_forecasts(original, geometry, visual, settings), motion, warning))
    assert reproduced == reference
    label_path = RELATIVE/track/"score/rows.jsonl"
    labels = read_rows(label_path)
    assert digest(label_path) == read(RELATIVE/track/"result.json")["score_sha256"]
    stages = {"1": Counter(), "2": Counter()}
    details = []
    for label, raw, forecast, gate in zip(labels, original, reference, gates, strict=True):
        assert frame_key(label) == frame_key(raw) == frame_key(forecast) == frame_key(gate)
        choice = gate["choice"]
        if label["truth"] != "approach" or not choice or str(choice[0]) != label["label"]:
            continue
        report = stages[label["label"]]
        report["correct_geometry"] += 1
        report["original_already_correct"] += bool(raw["alert"] and str(raw["direction"]) == label["label"])
        for name in ("same_side_before", "movement", "presence", "ordinal_range"):
            report[name] += int(gate[name])
        report["first_rejection_movement" if not gate["movement"] else
               "first_rejection_presence" if not gate["presence"] else
               "first_rejection_distance" if not gate["ordinal_range"] else "all_conditions"] += 1
        details.append(dict(**gate, label=label["label"], truth_distance_m=label["truth_distance_m"],
                            original_alert=raw["alert"], original_direction=raw["direction"],
                            final_hit=label["direction_hit"]))
    summaries = {"frozen_relative": counts(score_rows(labels, reference))}
    for mode in ("skip_movement", "skip_presence"):
        summaries[mode] = counts(score_rows(labels, with_extra_alerts(reference, gates, mode)))
    for limit in (2., 4.):
        new = dict(settings, maximum_position_innovation_m=limit)
        alternative = end_warnings(merge_forecasts(original, geometry, visual, new), motion, warning)
        summaries[f"innovation_{limit}"] = counts(score_rows(labels, alternative))
    output = OUT/track
    output.mkdir(exist_ok=False)
    jsonl(output/"approach-stages.jsonl", details)
    jsonl(output/"all-gates.jsonl", gates)
    write(output/"result.json", dict(track=track, stages=stages, summary=summaries,
        candidate_promoted=False, scope="development comparison after independent evaluation",
        stage_rows_sha256=digest(output/"approach-stages.jsonl"),
        all_gates_sha256=digest(output/"all-gates.jsonl"),
        inputs={str(p):digest(p) for p in (Path(__file__), MODULE, PROTOCOL, WARNING, archives, original_path,
                                          reference_path, label_path)}))
    print(track, stages, summaries)


if __name__ == "__main__":
    run()
