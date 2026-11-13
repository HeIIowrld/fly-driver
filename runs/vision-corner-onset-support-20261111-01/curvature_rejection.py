"""Diagnose which relative-colour curvature check removes corner evidence."""
from collections import Counter
from pathlib import Path
import argparse
from unittest.mock import patch

import numpy as np

from audit import INPUTS, RECOVERY, RUNS, MUGELLO
from corner_fusion_boundaries import curvature, decide, whole_shape_min
from evaluate_fused_corner import frame_key, load_frozen, read_rows
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write
import relative_corner_boundaries as relative

OUT = Path(__file__).parent

def history_vote(fused, aligned, cut):
    curves = [dict(curvature(shape)) for shape in aligned]
    accepted = []
    for x,k in fused:
        agree = sum(x in curve and np.sign(k)*curve[x] >= cut for curve in curves)
        accepted.append((x,k if agree >= 2 else 0.))
    return accepted

def run(track):
    config_name, archive_name = INPUTS[track]
    config_path = RECOVERY/config_name
    config = load_frozen(config_path)
    settings_path = Path(config["boundary_fusion_pins"])
    settings = load_frozen(settings_path)
    cut = settings["curvature_threshold_per_m"]
    capture_path = RUNS/archive_name
    result_path = (MUGELLO if track == "mugello" else RECOVERY/track)/"result.json"
    proof = read(result_path)
    assert digest(config_path) == proof["protocol_sha256"]
    label_path = OUT/track/"rows.jsonl"
    assert digest(label_path) == read(label_path.with_name("result.json"))["rows_sha256"]
    rows = read_rows(label_path)

    snapshot = {}
    def observe(fused, aligned):
        raw = curvature(fused)
        strict = whole_shape_min(fused, aligned)
        snapshot["two_history_votes"] = decide(history_vote(raw, aligned, cut), cut)
        snapshot["fused_only"] = decide(raw, cut)
        return strict

    previous = {}
    records = []
    modes = ("fused_only","two_history_votes","strict")
    metrics = {name:Counter() for name in modes}
    with patch.object(relative,"whole_shape_min",observe):
        values = relative.read_boundaries(capture_path, proof["camera_bundle_sha256_by_run"], track,
                                          Path(config["camera_config"]), settings)
        for row,(key,strict,travel) in zip(rows,values,strict=True):
            assert key == frame_key(row)
            assert (strict[0] if strict else None) == row["side"]
            assert (strict[1] if strict else None) == row.get("strong_m")
            record = {name:row[name] for name in ("run","frame","truth","label","physical_corner")}
            snapshot["strict"] = strict
            for name in modes:
                choice = snapshot.get(name)
                old = previous.get(name)
                moving = bool(choice and choice[0] and old and old[1] and choice[0]==old[1][0]
                         and adjacent(old[0],row) and travel is not None
                         and travel >= settings["minimum_local_travel_m"]
                         and abs(choice[1]-(old[1][1]-travel)) <= settings["maximum_position_innovation_m"])
                record[name] = {"choice":choice,"movement":moving}
                previous[name] = row,choice
                if not choice or not choice[0]:
                    continue
                if row["truth"] in ("straight","out_of_range"):
                    field = row["truth"]
                elif row["truth"] == "approach" and str(choice[0]) == row["label"]:
                    field = "correct_"+row["label"]
                else:
                    continue
                metrics[name][field] += 1
                if moving:
                    metrics[name][field+"_with_movement"] += 1
            snapshot.clear()
            records.append(record)
    p = OUT/track/"curvature-rejection-rows.jsonl"
    jsonl(p,records)
    write(OUT/track/"curvature-rejection.json",dict(track=track, metrics=metrics, rows_sha256=digest(p),
          seed=None,revision=proof["revision"],source_sha256={str(f):digest(f) for f in (
          label_path,config_path,settings_path,capture_path,result_path,Path(__file__))},
          termination="matched_strict_geometry_replayed"))
    print(track,metrics)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",choices=INPUTS,required=True)
    run(parser.parse_args().track)
