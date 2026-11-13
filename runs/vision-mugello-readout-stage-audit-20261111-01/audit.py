from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_fused_corner import frame_key, load_frozen, load_score, merge_forecasts, read_rows, visual_evidence
from evaluate_fused_corner_warning_end import capture_motion, end_warnings
from evaluate_recovered_corner import estimate_components, frozen_components, recover
from infer_frozen_next_corner import load_features, load_readout
from relative_corner_boundaries import read_boundaries
from score_corner_captures import adjacent
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
root = Path("runs/vision-mugello-independent-evaluation-20261111-01")
base = root/"base"
protocol_path = Path("runs/vision-corner-recovered-runner-20261111-01/mugello-protocol.json")
archive_map = Path("runs/vision-mugello-independent-capture-20261111-01/archive-map.json")
protocol = load_frozen(protocol_path)
result = read(root/"result.json")
assert result["protocol_sha256"] == digest(protocol_path)
base_proof = read(base/"result.json")
assert result["base_receipt_sha256"] == digest(base/"result.json")
assert base_proof["archives_sha256"] == digest(archive_map)
rules = load_frozen(Path(protocol["base_protocol"]))["comparison"]
fusion_path = Path(protocol["boundary_fusion_pins"])
fusion = load_frozen(fusion_path)
component_path = Path(protocol["component_pins"])
component = frozen_components(component_path)
distance_path = Path(component["active_alerts"]["distance_pins"])
distance = load_frozen(distance_path)
parent_dir = base/"parent/forecasts"
mixed_dir = base/"candidate/forecasts"
final_dir = root/"forecasts"
proof = read(mixed_dir/"result.json")
for directory in (parent_dir, mixed_dir, final_dir):
    assert digest(directory/"rows.jsonl") == read(directory/"result.json")["rows_sha256"]
assert digest(final_dir/"rows.jsonl") == result["forecasts_sha256"]
assert digest(root/"score/result.json") == result["score_receipt_sha256"]
truth = load_score(root/"score", result["score_receipt_sha256"])
parent, mixed, final = [read_rows(p/"rows.jsonl") for p in (parent_dir, mixed_dir, final_dir)]

features = base/"features"
index = features/"index.jsonl"
keys = [frame_key(r) for r in read_rows(index)]
x, a, _ = load_features(features/"original/features.npz", digest(index), len(keys))
y, b, _ = load_features(features/"reflected/features.npz", digest(index), len(keys))
available = a & b
valid_keys = [key for key, ok in zip(keys, available, strict=True) if ok]
def raw_readout(pins):
    predict, threshold = load_readout(read(pins))
    scores, distances, presence = predict(x[available], y[available], return_scores=True)
    return {k: (score, float(d), float(p)) for k, score, d, p in
            zip(valid_keys, scores, distances, presence, strict=True)}, threshold

mixed_pins = Path(rules["candidate_pins"])
raw, threshold = raw_readout(mixed_pins)
raw_parent, parent_threshold = raw_readout(Path(rules["parent_pins"]))
visual = visual_evidence(features, proof, mixed_pins, Path(fusion["visual_presence_model"]))
components = estimate_components(features, Path(component["direction_model"]), Path(distance["model"]),
                                 distance["prediction_clip_m"])
bundles = result["camera_bundle_sha256_by_run"]
track = result["track"]
boundaries = list(read_boundaries(archive_map, bundles, track, Path(protocol["camera_config"]), fusion))
merged = list(merge_forecasts(mixed, boundaries, visual, fusion))
warnings = list(end_warnings(merged, capture_motion(archive_map, track, bundles), protocol["warning_end"]))
replayed = list(recover(parent, warnings, boundaries, visual, components, fusion["visual_presence_threshold"],
                        component["direction_margin"], distance["publish_gain"]))
assert replayed == final

def measurements(raw_scores, threshold):
    if raw_scores is None:
        return {}
    scores, distance_m, presence = raw_scores
    signed_side = int(np.argmax(scores[1:])) + 1
    return dict(scores=scores.tolist(), side_without_background=signed_side, winner=int(np.argmax(scores)),
                distance_m=distance_m, presence_pass=presence >= threshold, distance_pass=distance_m <= 20.)

rows = []
for t, p, m, g, warn, f, (key, choice, travel) in zip(truth, parent, mixed, merged, warnings, final, boundaries, strict=True):
    assert all(frame_key(r) == key for r in (t, p, m, g, warn, f))
    raw_boundary = int(choice[0]) if choice is not None else None
    v = visual.get(key)
    c = components.get(key)
    rows.append(dict(run=key[0], frame=key[1], truth=t["truth"], state=t["state"], label=t["label"],
                     physical_corner=t["physical_corner"], direction_hit=t["direction_hit"],
                     parent=measurements(raw_parent.get(key), parent_threshold), mixed=measurements(raw.get(key), threshold),
                     boundary_side=raw_boundary, boundary_distance_m=float(choice[1]) if raw_boundary else None,
                     travel_m=travel, fusion_visual_pass=bool(v and v[0]>=fusion["visual_presence_threshold"]-1e-12 and 0<=v[1]<=20),
                     recovery_signed=c[0] if c else None, metric_distance_m=c[1] if c else None,
                     stage_alerts=[r["alert"] for r in (p,m,g,warn,f)]))
summary = {}
for label in ("1","2"):
    missed = [r for r in rows if r["truth"]=="approach" and r["label"]==label and not r["direction_hit"]]
    side = int(label)
    summary[label] = dict(total=len(missed), available=sum(bool(r["mixed"]) for r in missed),
        mixed_winner=dict(Counter(r["mixed"].get("winner") for r in missed)),
        parent_winner=dict(Counter(r["parent"].get("winner") for r in missed)),
        side_correct=sum(r["mixed"].get("side_without_background")==side for r in missed),
        side_correct_with_gates=sum(r["mixed"].get("side_without_background")==side and r["mixed"]["presence_pass"] and r["mixed"]["distance_pass"] for r in missed),
        presence=sum(r["mixed"].get("presence_pass",False) for r in missed),
        distance=sum(r["mixed"].get("distance_pass",False) for r in missed),
        boundary=dict(Counter(r["boundary_side"] for r in missed)),
        boundary_correct_with_visual=sum(r["boundary_side"]==side and r["fusion_visual_pass"] for r in missed),
        signed_correct=sum(r["recovery_signed"] is not None and r["recovery_signed"]*(1 if side==1 else -1)>component["direction_margin"] for r in missed),
        alerts_at_each_stage=np.sum([r["stage_alerts"] for r in missed],axis=0).tolist())
for group in ("approach","straight","out_of_range"):
    subset = [r for r in rows if r["truth"]==group]
    summary[group] = dict(frames=len(subset),
        presence_and_distance=sum(r["mixed"].get("presence_pass",False) and r["mixed"].get("distance_pass",False) for r in subset),
        boundary_visual=sum(bool(r["boundary_side"]) and r["fusion_visual_pass"] for r in subset))
jsonl(out/"rows.jsonl", rows)
write(out/"result.json", dict(summary=summary, revision=result["revision"], seed=None, thresholds={"mixed":threshold,"parent":parent_threshold},
      track=track, termination="replayed_frozen_candidate_matches", rows_sha256=digest(out/"rows.jsonl"),
      source_sha256={str(p):digest(p) for p in (protocol_path,root/"result.json",base/"result.json",archive_map,index,Path(__file__))}))
print(summary)
