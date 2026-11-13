"""Attribute the unchanged warning-end cascade on the frozen Zandvoort evaluation."""
import argparse
from collections import Counter
from pathlib import Path

from corner_fusion_boundaries import read_boundaries
from evaluate_fused_corner import frame_key, load_frozen, read_rows, visual_evidence
from train_vision_teacher_line import digest, jsonl, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("replay", "base", "archives"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    run, base = args.replay, args.base
    end_receipt = read(run/"result.json")
    assert not end_receipt["development_rehearsal"] and end_receipt["status"] == "evaluated"
    outer = Path("runs/vision-corner-warning-end-runner-20261111-01/zandvoort-protocol.json")
    assert digest(outer) == end_receipt["protocol_sha256"]
    fusion_path = Path(load_frozen(outer)["fusion_protocol"])
    fusion_protocol = load_frozen(fusion_path)
    settings = load_frozen(Path(fusion_protocol["development_pins"]))
    base_protocol = load_frozen(Path(fusion_protocol["base_protocol"]))
    receipt = read(base/"result.json")
    track, bundles = receipt["track"], receipt["camera_bundle_sha256_by_run"]
    assert track == end_receipt["track"] == "ks_zandvoort"
    pins = Path(base_protocol["comparison"]["candidate_pins"])
    forecast_path = base/"candidate/forecasts/rows.jsonl"
    forecast_proof = read(forecast_path.with_name("result.json"))
    assert digest(forecast_path) == forecast_proof["rows_sha256"]
    assert digest(args.archives) == receipt["archives_sha256"]
    scores = visual_evidence(base/"features", forecast_proof, pins, Path(settings["visual_presence_model"]))
    observations = read_boundaries(args.archives, bundles, track, Path(fusion_protocol["camera_config"]), settings)
    truth_path = run/"score/rows.jsonl"
    assert digest(truth_path) == read(truth_path.with_name("result.json"))["rows_sha256"]
    final_path = run/"forecasts/rows.jsonl"
    assert digest(final_path) == end_receipt["forecasts_sha256"]
    fused_path = run/"fusion/forecasts/rows.jsonl"
    assert digest(fused_path) == read(run/"fusion/result.json")["forecasts_sha256"]
    history = None
    travel_total = 0.
    stages = {"1": Counter(), "2": Counter()}
    records = []
    for base_row, (key, choice, travel), fused, final, truth in zip(
            read_rows(forecast_path), observations, read_rows(fused_path), read_rows(final_path),
            read_rows(truth_path), strict=True):
        assert all(frame_key(r) == key for r in (base_row, fused, final, truth))
        if (travel is not None and travel >= .10 and choice and choice[0] and history and
                history[0] == choice[0] and abs(choice[1]-(history[1]-travel)) <=
                settings["maximum_position_innovation_m"]):
            travel_total += travel
        else:
            travel_total = 0.
        history = choice
        confirmed = travel_total >= settings["minimum_local_travel_m"]
        visual = scores.get(key)
        presence_pass = bool(visual and visual[0] >= settings["visual_presence_threshold"] - 1e-12)
        range_pass = bool(visual and 0 <= visual[1] <= 20)
        added = not base_row["alert"] and confirmed and presence_pass and range_pass
        assert fused["alert"] == (base_row["alert"] or added)
        if truth["truth"] != "approach":
            continue
        label = truth["label"]
        count = stages[label]
        count["approaches"] += 1
        count["base_correct"] += bool(base_row["alert"] and str(base_row["direction"]) == label)
        count["correct_current_geometry"] += bool(choice and str(choice[0]) == label)
        count["wrong_current_geometry"] += bool(choice and choice[0] and str(choice[0]) != label)
        count["geometry_unavailable"] += choice is None
        count["geometry_straight"] += bool(choice and choice[0] == 0)
        count["correct_confirmed_geometry"] += bool(confirmed and str(choice[0]) == label)
        count["correct_confirmed_and_visual_presence"] += bool(confirmed and str(choice[0]) == label and presence_pass)
        count["geometry_correct_on_missed"] += bool(choice and str(choice[0]) == label and
                                                   not (base_row["alert"] and str(base_row["direction"]) == label))
        count["fused_additions"] += added
        count["warning_ended"] += bool(final.get("warning_ended"))
        count["final_direction_hit"] += bool(truth["direction_hit"])
        count["final_joint_hit"] += bool(truth["joint_hit"])
        records.append(dict(run=key[0], frame=key[1], label=label, truth_distance_m=truth["truth_distance_m"],
                            geometry=choice, travel_total_m=travel_total, visual=visual, base=base_row,
                            added=added, final=final, direction_hit=truth["direction_hit"], joint_hit=truth["joint_hit"]))
    assert len(records) == 58
    path = run/"failure-stages.jsonl"
    jsonl(path, records)
    write(run/"failure-stages.json", dict(stages=stages, scope="diagnostic on frozen independent result; no parameter selection",
        rows_sha256=digest(path), inputs={str(p):digest(p) for p in (
            Path(__file__), forecast_path, fused_path, final_path, truth_path, fusion_path,
            run/"result.json", base/"result.json", args.archives)}))
    print(stages)


if __name__ == "__main__":
    main()
