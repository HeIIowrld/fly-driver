"""Evaluate saved frozen readouts against installed AI-spline geometry."""
import argparse
import re
import struct
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

import train_vision_teacher_line as teacher
from flybrain_drive.ac_camera_archive import CameraArchive
from train_next_corner import CURVATURE_THRESHOLD, HORIZON, assign
from audit_road_edge_projection import road_edges
from train_vision_teacher_line import VIEWS, digest, jsonl, read, write
from score_corner_captures import (
    DISTANCE_TOLERANCE_M, MAX_GAP_S, MIN_SPEED_MPS, STABLE_FRAMES, capture_row, summarize,
)


def load_edges(path):
    # AssettoServer FastLane.FromFileV7 documents field layout.
    raw = path.read_bytes()
    version, n, _, _ = struct.unpack_from("<4i", raw)
    assert version == 7 and 3 <= n <= 100000
    details = np.frombuffer(raw, "<f4", count=5*n, offset=16).reshape(n, 5)
    extra_offset = 16 + 20*n
    assert struct.unpack_from("<i", raw, extra_offset)[0] == n
    extra = np.frombuffer(raw, "<f4", count=18*n, offset=extra_offset+4).reshape(n, 18)
    assert np.isfinite(details[:, :3]).all() and np.isfinite(extra[:, 5:7]).all()
    assert ((extra[:, 5:7] >= 0) & (extra[:, 5:7] < 100)).all()
    return road_edges([dict(native_xyz=point.tolist(), left_m=float(sides[0]), right_m=float(sides[1]))
                       for point, sides in zip(details[:, :3], extra[:, 5:7], strict=True)]), n


def index_frames(rows, field):
    indexed = {}
    for row in rows:
        key = row[field]
        if type(key) is not int or key < 0 or key in indexed:
            raise ValueError(f"Invalid or duplicate {field}: {key}")
        indexed[key] = row
    return indexed


def common_rows(tag, archive, predictions, track, truth):
    captures = index_frames(map(json.loads, archive.read("stream.jsonl").splitlines()), "capture_id")
    if not captures or predictions.keys() - captures.keys():
        raise ValueError(f"Missing capture provenance in {tag}")
    for frame, capture in captures.items():
        filename = Path(capture["file"]).name
        if archive.by_name[filename]["sha256"] != capture["file_sha256"]:
            raise ValueError(f"Frame receipt differs for {tag}/{frame}")
        meta = archive.json(filename)
        if (meta["track_id"] != track or meta["index"] != frame
                or not capture.get("generation") or meta.get("generation") != capture["generation"]):
            raise ValueError(f"Track, generation or capture ID differs for {tag}/{frame}")
        views = meta["views"]
        early = min(views, key=lambda v: v["render_started_clock"], default={}).get("source_body_before_render")
        late = max(views, key=lambda v: v["render_call_finished_clock"], default={}).get("source_body_after_render")
        speeds = [pose.get("speed_m_s") if pose and pose.get("physics_available") else None
                  for pose in (early, late)]
        row = capture_row(tag, capture, predictions.get(frame), (truth(early), truth(late)), speeds)
        row["camera_complete"] = (len(views) == len(VIEWS)
                                  and {view["view_id"] for view in views} == set(VIEWS))
        if any(item.get(flag) for item in (capture, meta) for flag in ("ideal_line_shown", "aid_probe_only")):
            row.update(truth="aid_probe", direction_hit=False, joint_hit=False,
                       signed_distance_error_m=None, absolute_distance_error_m=None)
        yield row


def offline_sources(args, inputs):
    forecasts = args.forecasts
    receipt = forecasts.with_name("result.json")
    proof = read(receipt)
    mapping = read(args.archive_map)
    bundles = proof.get("camera_bundle_sha256_by_run", {})
    if (proof.get("verified_track_id") != args.track or not mapping or mapping.keys() != bundles.keys()
            or any(not isinstance(h, str) or re.fullmatch(r"[a-f0-9]{64}", h) is None for h in bundles.values())
            or proof["rows_sha256"] != digest(forecasts)
            or digest(args.pins) not in proof["inputs_sha256"].values()):
        raise ValueError("Replay requires matching track, bundle hashes, forecast digest and pins")
    inputs.extend((forecasts, receipt, args.archive_map))
    predictions = {tag: [] for tag in mapping}
    for row in map(json.loads, forecasts.read_text(encoding="utf8").splitlines()):
        predictions[row["run"]].append(row)
    for tag, directory in mapping.items():
        yield tag, Path(directory), index_frames(predictions[tag], "frame"), bundles[tag]


def load_track_truth(track):
    edges, n = load_edges(track)
    center = ((edges[0]+edges[1])*.5)[:, [2,0]]
    start, end = center, np.roll(center,-1,axis=0)
    diff = end-start
    lengths = np.linalg.norm(diff,axis=1)
    assert (lengths > 0).all()
    arc = np.r_[0.,np.cumsum(lengths)]
    # Periodic padding keeps the sampled smoothing kernel away from the lap seam.
    pad = 50
    repeated = np.r_[center[-pad:],center,center[:pad+1]]
    padded_arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(repeated,axis=0),axis=1))]
    shift = padded_arc[pad]
    teacher.CORNER = CURVATURE_THRESHOLD
    spans = [(a-shift,b-shift,sgn) for a,b,sgn in teacher.corners(padded_arc,repeated)
             if shift <= a < shift+arc[-1]]
    assert spans

    def truth(pose):
        if not pose or any(key not in pose for key in ("renderer_only_position", "renderer_only_look")):
            return "unregistered", None, None
        point = np.array(pose["renderer_only_position"])[[2,0]]
        look = np.array(pose["renderer_only_look"])[[2,0]]
        t = np.clip(np.sum((point-start)*diff,axis=1)/lengths**2,0,1)
        delta = point-(start+t[:,None]*diff)
        j = int(np.argmin(np.sum(delta**2,axis=1)))
        if (not np.isfinite(point).all() or not np.isfinite(look).all()
                or np.linalg.norm(delta[j]) > 8 or look @ (diff[j]/lengths[j]) < .7):
            return "unregistered", None, None
        station = (arc[j]+t[j]*lengths[j])%arc[-1]
        info = assign(station,spans,arc[-1],periodic=True)
        if info is None:
            return "in_corner", None, None
        corner, direction, distance = info
        return str(direction), float(distance), corner

    return truth, n, spans


def evaluate(args):
    out = args.output
    assert args.track_dir.name == args.track
    track = args.track_dir/"ai/fast_lane.ai"
    truth, n, spans = load_track_truth(track)
    report, combined = {}, []
    inputs = [track,Path(__file__),Path("scripts/train_vision_teacher_line.py"),Path("scripts/train_next_corner.py")]
    common = getattr(args, "common", False)
    bundle_hashes = {}
    if common:
        inputs.extend((Path("scripts/score_corner_captures.py"), Path("scripts/audit_road_edge_projection.py"),
                       Path("src/flybrain_drive/ac_camera_archive.py")))
    if getattr(args, "forecasts", None) is not None:
        for tag, directory, predictions, bundle in offline_sources(args, inputs):
            with CameraArchive(directory) as archive:
                if archive.bundle_sha256 != bundle:
                    raise ValueError(f"Camera bundle differs for {tag}")
                bundle_hashes[tag] = bundle
                rows = list(common_rows(tag, archive, predictions, args.track, truth))
            report[tag] = summarize(rows)
            combined.extend(rows)
    for source in args.recordings:
        tag = source.name
        if tag in report:
            raise ValueError("Recording names must be unique")
        trace = source/"live-frozen-readout.jsonl"
        receipt = source/"result.json"
        proof = read(receipt)
        manifest = source/"manifest.json"
        captured_pins = read(manifest)["config"]["arguments"]["live_frozen_pins"]
        if proof["live_input_hashes"][captured_pins] != digest(args.pins):
            raise ValueError(f"Readout pins differ for {tag}")
        inputs.append(manifest)
        if digest(trace) != proof["live_frozen_readout"]["rows_sha256"]:
            raise ValueError(f"Readout trace differs for {tag}")
        inputs.extend((trace,receipt))
        output = []
        with CameraArchive(source/"vision") as archive:
            if archive.bundle_sha256 != proof["camera_archive"]["bundle_sha256"]:
                raise ValueError(f"Camera bundle differs for {tag}")
            if common:
                bundle_hashes[tag] = archive.bundle_sha256
                predictions = index_frames(map(json.loads, trace.read_text(encoding="utf8").splitlines()), "frame")
                rows = list(common_rows(tag, archive, predictions, args.track, truth))
                report[tag] = summarize(rows)
                combined.extend(rows)
                continue
            frames = {r["capture_id"]:r for r in map(json.loads,archive.read("stream.jsonl").splitlines())}
            for row in map(json.loads,trace.read_text(encoding="utf8").splitlines()):
                if not row["body_valid"] or row["speed_mps"] < 1/3.6 or not row["available"]:
                    continue
                frame = row["frame"]
                meta = archive.json(Path(frames[frame]["file"]).name)
                assert meta["track_id"] == args.track and meta["index"] == frame
                early = min(meta["views"],key=lambda v:v["render_started_clock"])["source_body_before_render"]
                late = max(meta["views"],key=lambda v:v["render_call_finished_clock"])["source_body_after_render"]
                lo,hi = truth(early),truth(late)
                label = lo[0] if lo[0] == hi[0] else "ambiguous"
                ds = [x[1] for x in (lo,hi) if x[1] is not None]
                band = label in ("1","2") and len(ds)==2 and min(ds)>=4 and max(ds)<=20
                outside = (label == "0" or label in ("1","2") and len(ds)==2
                           and (max(ds)<4 or min(ds)>20))
                scored = band or outside
                alert = bool(row["alert"])
                distance = row["distance_m"]
                error = (float(np.mean([abs(distance-d) for d in ds]))
                         if alert and band and str(row["direction"])==label else None)
                output.append(dict(run=tag,frame=frame,label=label,truth_endpoints_m=ds,
                    in_4_to_20_m=band,scored=scored,outside=outside,alert=alert,predicted_direction=row["direction"],
                    predicted_distance_m=distance,absolute_distance_error_m=error))
        report[tag] = dict(forecasts=len(output), labels=Counter(r["label"] for r in output),
            alerts=sum(r["alert"] for r in output), approach_frames=sum(r["in_4_to_20_m"] for r in output),
            direction_hits=sum(r["absolute_distance_error_m"] is not None for r in output),
            alerts_outside_band=sum(r["alert"] and r["outside"] for r in output),
            unscored=sum(not r["scored"] for r in output),
            distance_errors_m=[r["absolute_distance_error_m"] for r in output if r["absolute_distance_error_m"] is not None])
        combined.extend(output)
    metadata = {}
    if common:
        revision = subprocess.check_output(
            ["git", "-c", "safe.directory=%(prefix)/" + Path.cwd().as_posix(), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        corners = {}
        for tag, summary in report.items():
            for event in summary["events"]:
                corners.setdefault(event["physical_corner"], []).append(dict(run=tag, **event))
        metadata = dict(mode="common", revision=revision, camera_bundle_sha256_by_run=bundle_hashes,
                        capture_frames=len(combined), physical_corners=[
                            dict(id=i, onset_station_m=spans[i][0], end_station_m=spans[i][1],
                                 side="left" if spans[i][2] > 0 else "right", approaches=events)
                            for i, events in sorted(corners.items())
                        ],
                        parameters=dict(approach_band_m=[4, 20], tolerance_m=DISTANCE_TOLERANCE_M,
                                        minimum_speed_mps=MIN_SPEED_MPS, stable_frames=STABLE_FRAMES,
                                        max_capture_gap_s=MAX_GAP_S, straight_horizon_m=HORIZON))
    out.mkdir(parents=True,exist_ok=False)
    inputs.append(args.pins)
    jsonl(out/"rows.jsonl",combined)
    write(out/"result.json",dict(status="completed_live_corner_geometry_evaluation", track=args.track, hashes={str(p):digest(p) for p in inputs},
         rows_sha256=digest(out/"rows.jsonl"), installed_spline_points=n, corner_spans=len(spans),
         curvature_threshold_inv_m=CURVATURE_THRESHOLD, report=report, actor_input=False,
         seed=None, termination="archive_end", **metadata))
    if common:
        print({tag: dict(frames=r["frames"], approach_frames=r["approach_frames"], outputs=r["outputs"],
                         outside_alerts=r["false_alerts"]["outside_band"]["frames"]) for tag, r in report.items()})
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",required=True)
    for name in ("track-dir", "pins", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--common", action="store_true", help="Score all saved camera captures")
    parser.add_argument("--archive-map", type=Path, help="JSON mapping run IDs to camera archive directories")
    parser.add_argument("--forecasts", type=Path, help="Track-verified offline forecasts JSONL")
    parser.add_argument("recordings",type=Path,nargs="*")
    args = parser.parse_args()
    if re.fullmatch(r"[a-z0-9_-]+",args.track) is None or args.track_dir.name != args.track:
        parser.error("Exact installed AC track directory and matching track ID required")
    offline = args.forecasts is not None or args.archive_map is not None
    if offline and (not args.common or args.forecasts is None or args.archive_map is None or args.recordings):
        parser.error("Use --common with both --archive-map and --forecasts for offline forecasts")
    if not offline and not args.recordings:
        parser.error("Provide at least one live recording")
    evaluate(args)


if __name__ == "__main__":
    main()
