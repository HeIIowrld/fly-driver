"""Replay short-baseline edge curvature in current and motion-aligned previous images."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

import relative_corner_boundaries as relative
from audit import INPUTS, RECOVERY, MUGELLO, RUNS
from corner_fusion_boundaries import stations, whole_shape_min
from evaluate_fused_corner import frame_key, load_frozen, load_score, read_rows
from train_vision_teacher_line import digest, jsonl, read, write
from transfer import BASES, compare_tracks, score
from score_corner_captures import adjacent

OUT = Path(__file__).parent

def edge_curvatures(shape):
    """Estimate signed curvature over three adjacent sections for each road edge."""
    width = shape[:,1]-shape[:,0]
    legal = np.isfinite(shape).all(axis=1) & (width>3) & (width<20)
    heading = np.arctan2(np.diff(shape,axis=0), np.diff(stations)[:,None])
    ds = (stations[2:]-stations[:-2])/2
    curvature = np.diff(heading,axis=0)/ds[:,None]
    curvature[~(legal[:-2] & legal[1:-1] & legal[2:])] = np.nan
    return curvature

def estimate(aligned, cut):
    """Keep all pooled bends until the next frame can disambiguate position."""
    if len(aligned) < 2:
        return {"pooled_all": []}
    curves = np.stack([edge_curvatures(s) for s in aligned])
    pooled = (curves[:,1:]+curves[:,:-1])/2
    options = []
    for side, sign in ((1,1.),(2,-1.)):
        agreement = np.all(sign*pooled>=cut,axis=-1)
        hits = agreement[-1] & agreement[:-1].any(axis=0)
        options.extend((side,float(stations[i+1])) for i in np.flatnonzero(hits) if stations[i+1]<=20)
    return {"pooled_all": options}




def run(track):
    cfg_name, archive_name = INPUTS[track]
    protocol_path = RECOVERY/cfg_name
    configuration = load_frozen(protocol_path)
    settings_path = Path(configuration["boundary_fusion_pins"])
    settings = load_frozen(settings_path)
    cut = settings["curvature_threshold_per_m"]
    rules_path = Path(configuration["base_protocol"])
    rules = read(rules_path)["comparison"]
    archive_path = RUNS/archive_name
    evaluation = MUGELLO if track=="mugello" else RECOVERY/track
    eval_path = evaluation/"result.json"
    result = read(eval_path)
    assert digest(protocol_path)==result["protocol_sha256"]
    base = BASES[track]
    base_path = base/"result.json"
    assert digest(base_path)==result["base_receipt_sha256"]
    parent = load_score(base/"parent/score", read(base_path)["score_receipt_sha256"]["parent"])
    current = load_score(evaluation/"score", result["score_receipt_sha256"])
    geometry_path = RUNS/"vision-corner-onset-support-20261111-01"/track/"rows.jsonl"
    assert digest(geometry_path)==read(geometry_path.with_name("result.json"))["rows_sha256"]
    saved = read_rows(geometry_path)
    snapshot = {}
    def inspect(fused, aligned):
        snapshot.update(estimate(aligned, cut))
        return whole_shape_min(fused,aligned)
    rows = []
    keys = ("pooled_all","associated","associated_visual")
    with patch.object(relative, "whole_shape_min", inspect):
        raw = relative.read_boundaries(archive_path,result["camera_bundle_sha256_by_run"],track,
                                       Path(configuration["camera_config"]),settings)
        for reference, source,(key,choice,move) in zip(saved,current,raw,strict=True):
            assert key==frame_key(reference)==frame_key(source)
            assert (choice[0] if choice else None)==reference["side"]
            assert (choice[1] if choice else None)==reference.get("strong_m")
            rows.append(dict(run=key[0],frame=key[1],received_s=source["received_s"],
                             generation=source["generation"],travel_m=move,
                             **{name:snapshot.get(name) for name in keys}))
            snapshot.clear()
    stage_root = (RUNS/"vision-mugello-readout-stage-audit-20261111-01" if track=="mugello"
                  else RUNS/"vision-corner-relative-chroma-gates-20261111-01"/track)
    visual_path = stage_root/("rows.jsonl" if track=="mugello" else "all-gates.jsonl")
    visual = read_rows(visual_path)
    previous = None
    for obs, evidence in zip(rows,visual,strict=True):
        assert frame_key(obs)==frame_key(evidence)
        move = obs["travel_m"]
        allowed = (previous is not None and adjacent(previous,obs)
                   and move is not None and move>=settings["minimum_local_travel_m"])
        proposals = []
        if allowed:
            for side,x in obs["pooled_all"] or ():
                residuals = [abs(x-(old_x-move)) for old_side,old_x in previous["pooled_all"] or () if old_side==side]
                if residuals and min(residuals)<=settings["maximum_position_innovation_m"]:
                    proposals.append((min(residuals),x,side))
        choice = None
        if proposals:
            best = min((residual,x) for residual,x,_ in proposals)
            sides = {side for residual,x,side in proposals if (residual,x)==best}
            if len(sides)==1:
                choice = sides.pop(),best[1]
        obs["associated"] = choice
        accept = evidence["fusion_visual_pass"] if track=="mugello" else evidence["presence"] and evidence["ordinal_range"]
        obs["associated_visual"] = choice if accept else None
        previous = obs
    metrics = {}
    before = compare_tracks(parent,current,rules)
    assert before["metrics"]==result["comparison"]["metrics"]
    for name in ("associated","associated_visual"):
        selection = {}
        for require_motion in (False,True):
            previous = None
            emitted = []
            extra = []
            for obs, source in zip(rows,current,strict=True):
                choice = obs[name]
                prior = previous[name] if previous is not None else None
                move = obs["travel_m"]
                confirm = (choice and prior and choice[0]==prior[0] and adjacent(previous,obs)
                           and move is not None and move>=settings["minimum_local_travel_m"]
                           and abs(choice[1]-(prior[1]-move))<=settings["maximum_position_innovation_m"])
                allow = choice and (not require_motion or confirm) and source["state"]=="straight"
                row = score(source, choice) if allow else source
                if allow:
                    extra.append((row["truth"],row["direction_hit"],row["joint_hit"]))
                emitted.append(row)
                previous = obs
            comparison = compare_tracks(parent,emitted,rules)
            comparison["added_frames"] = len(extra)
            selection["motion" if require_motion else "direct"] = comparison
        metrics[name] = selection
    folder = OUT/track
    folder.mkdir()
    rows_path = folder/"rows.jsonl"
    jsonl(rows_path,rows)
    write(folder/"result.json",dict(track=track,revision=result["revision"],seed=None,before=before,metrics=metrics,
        rows_sha256=digest(rows_path),source_sha256={str(p):digest(p) for p in (
        Path(__file__),protocol_path,rules_path,settings_path,archive_path,eval_path,base_path,geometry_path,visual_path)},
        termination="development_comparison_completed"))
    print(track, {k:{m:v["metrics"]["candidate"] for m,v in selection.items()}
                  for k,selection in metrics.items()})

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",choices=INPUTS,required=True)
    run(parser.parse_args().track)
