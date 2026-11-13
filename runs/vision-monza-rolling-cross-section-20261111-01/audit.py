"""Score spatial agreement across complete archived Monza corner windows."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import train_vision_teacher_line as teacher
from train_next_corner import BASE, CURVATURE_THRESHOLD, MONZA
from train_vision_teacher_line import CameraArchive, at, digest, jsonl, read, write

out = Path(__file__).parent
parent = out.parent / "vision-monza-extension-width-guard-20261111-01"
source = parent / "rows.jsonl"
proof = read(parent / "result.json")
assert digest(source) == proof["rows_sha256"]
track = Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
ref = out.parent / "vision-corner-label-geometry-20261110-01/probe.py"
assert all(digest(path) in proof["hashes"].values() for path in (track, ref))
teacher.CORNER = CURVATURE_THRESHOLD
code = ref.read_text(encoding="utf-8-sig")
exec(code[code.index("def periodic_line("):code.index("def near(")])
data = np.genfromtxt(track, names=True, delimiter=",")
center = np.column_stack([(data[f"left_border_{axis}"] + data[f"right_border_{axis}"])/2
                          for axis in ("x", "y")])
length, spans, stations, tree = periodic_line(center)

groups = defaultdict(list)
with source.open(encoding="utf8") as stream:
    for row in map(json.loads, stream):
        if row["policy"] in ("short", "alert_guarded"):
            groups[row["run"], row["event"], row["policy"]].append(row)

phases = {"approach", "early_turn", "mid_turn", "late_turn", "exit"}
complete = {key: rows for key, rows in groups.items() if {r["phase"] for r in rows} == phases}
poses = {}
inputs = [source, track, ref, Path(__file__)]
for run, qualification in MONZA:
    folder = BASE / run
    receipt = folder / qualification / "result.json"
    assert digest(receipt) in proof["hashes"].values()
    inputs.append(receipt)
    frames = {r["frame"] for key, rows in complete.items() if key[0] == run for r in rows}
    with CameraArchive(folder / "demonstration/camera/vision") as archive:
        assert archive.bundle_sha256 == read(receipt)["camera_bundle_sha256"]
        for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
            frame = item["capture_id"]
            if frame not in frames:
                continue
            meta = archive.json(Path(item["file"]).name)
            assert meta["index"] == frame
            pose = max(meta["views"], key=lambda v:v["render_call_finished_clock"])["source_body_after_render"]
            origin = np.asarray(pose["renderer_only_position"])[[2, 0]]
            forward = np.asarray(pose["renderer_only_look"])[[2, 0]]
            forward /= np.linalg.norm(forward)
            poses[run, frame] = origin, forward, item["received"], item["generation"]

reports = []
for (run, event, policy), rows in complete.items():
    chains = []
    failures = Counter()
    prior = None
    for row in sorted(rows, key=lambda r:r["frame"]):
        frame = row["frame"]
        origin, forward, time, generation = poses[run, frame]
        assert time == row["received"] and generation == row["generation"]
        if row["map_margin_m"] is None or row["map_margin_m"] < 0:
            failures["no_bounded_line"] += 1
            prior = None
            continue
        line = row["line"]
        left = np.array([-forward[1], forward[0]])
        world = origin + np.asarray(line["x_m"])[:, None]*forward + np.asarray(line["y_m"])[:, None]*left
        distance, nearest = tree.query(world)
        start, end, _ = spans[event]
        relative = (stations[nearest]-start+length/2)%length-length/2
        if np.any(np.diff(relative) < 0) or max(distance) > 10:
            failures["unusable_projection"] += 1
            prior = None
            continue
        points = {}
        for station in range(int(min(relative)), int(max(relative))+1):
            i = (start+station) % len(tree.data)
            axis = tree.data[(i+1)%len(tree.data)]-tree.data[i-1]
            axis /= np.linalg.norm(axis)
            offset = world-tree.data[i]
            signed = offset@axis
            crossings = np.flatnonzero((signed[:-1] < 0) & (signed[1:] >= 0))
            if len(crossings) == 1:
                j = crossings[0]
                weight = -signed[j]/(signed[j+1]-signed[j])
                point = offset[j] + weight*(offset[j+1]-offset[j])
                points[station] = point@np.array([-axis[1], axis[0]])
        if not points:
            failures["no_cross_section"] += 1
            prior = None
            continue
        reason = None
        if prior is None:
            reason = "first_available"
        else:
            old_frame, old_time, old_generation, old_points = prior
            common = points.keys() & old_points.keys()
            if frame != old_frame+1 or old_generation != generation or not 0 < time-old_time <= .18:
                reason = "capture_gap"
            elif len(common) < 5:
                reason = "overlap_under_5_stations"
            elif max(abs(points[k]-old_points[k]) for k in common) > 1:
                reason = "lateral_disagreement_over_1m"
        if reason:
            failures[reason] += 1
            chains.append(dict(phases=set(), frames=0, furthest_preview_m=-float("inf")))
        chain = chains[-1]
        chain["phases"].add(row["phase"])
        chain["frames"] += 1
        chain["furthest_preview_m"] = max(chain["furthest_preview_m"], max(points))
        prior = frame, time, generation, points
    entry = [c for c in chains if "approach" in c["phases"]]
    reports.append(dict(run=run, event=event, policy=policy, frames=len(rows),
                        span_m=spans[event][1]-spans[event][0],
                        whole_corner=any(c["phases"] == phases for c in chains),
                        entry_chain_furthest_m=max((c["furthest_preview_m"] for c in entry),default=None),
                        max_chain_frames=max((c["frames"] for c in chains),default=0),
                        failures=dict(failures)))

summary = {}
for policy in ("short", "alert_guarded"):
    subset = [r for r in reports if r["policy"] == policy]
    summary[policy] = dict(events=len(subset), whole_corner=sum(r["whole_corner"] for r in subset),
                           reaches_exit_from_approach=sum(r["entry_chain_furthest_m"] is not None and
                               r["entry_chain_furthest_m"] >= r["span_m"] for r in subset),
                           failures=dict(sum((Counter(r["failures"]) for r in subset), Counter())))
jsonl(out / "rows.jsonl", reports)
write(out / "result.json", dict(summary=summary, revision=proof["revision"], seed=None,
                               termination="archive_end", map_for_evaluation_only=True,
                               cross_section_interpolation=True, min_overlap_stations=5, max_overlap_separation_m=1,
                               hashes={str(p):digest(p) for p in inputs},
                               rows_sha256=digest(out / "rows.jsonl")))
print(summary)
