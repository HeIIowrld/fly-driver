"""Generate body-frame lines and preview speed envelopes from recorded four-view images."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.local_line_extension import extend_line
from flybrain_drive.local_racing_line import maneuver_from_line, plan_local_line, speed_bound_with_corner_preview
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from flybrain_drive.road_boundary_memory import RoadBoundaryMemory
from flybrain_drive.surface_road_boundary import SurfaceContrastRoadBoundary
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix
from train_vision_teacher_line import digest, jsonl, read, write


LIMITS = dict(clearance_m=1.1, max_curvature=.15, lateral_accel_mps2=4.,
              braking_mps2=4., speed_ceiling_mps=30/3.6)


BODY_CONTRACT = Path("configs/ac-virtual-navigation-body-v2.json")
MAX_BODY_AGE_S = .05


class ArchivedMotion:
    """Same-episode received speed/yaw samples and relative boundary state."""

    def __init__(self, folder, episode, run):
        if not folder.resolve().is_relative_to(episode.resolve()):
            raise ValueError("Camera archive must be under its native episode directory")
        path = episode/"owner-native-copy.json"
        receipt = episode/"result.json"
        manifest = episode/"manifest.json"
        if digest(path) != read(receipt)["native_sha256"]:
            raise ValueError("Native archive digest differs")
        self.inputs = path, receipt, manifest, BODY_CONTRACT
        identity = read(manifest)["config"]["identity"]
        keys = ("car_model", "ac_version", "shared_memory_version")
        self.identity = {k: identity[k] for k in keys}
        encoder = NavigationBodyEncoder(BODY_CONTRACT, native_identity=self.identity)
        self.native_records = []
        seen, samples = {}, []
        for row in read(path):
            authority = row["authority"]
            payload = base64.b64decode(authority["prefix_base64"], validate=True)
            prefix = payload[:PHYSICS_PREFIX.size]
            if (not authority["same_native_copy"]
                    or hashlib.sha256(payload).hexdigest() != authority["prefix_sha256"]
                    or hashlib.sha256(prefix).hexdigest() != authority["body_prefix_sha256"]):
                raise ValueError("Invalid original native packet")
            packet = authority["packet_id"]
            if packet in seen:
                if seen[packet] != prefix:
                    raise ValueError("Native packet changed between copies")
                continue
            seen[packet] = prefix
            t = authority["received_monotonic_seconds"]
            body = encoder.encode(decode_navigation_body_prefix(prefix,t), episode_id=run,
                                  decision_monotonic_seconds=t)
            if body.packet_id != packet:
                raise ValueError("Native packet identity differs")
            self.native_records.append((body, prefix))
            valid = body.valid[[16,4]].all()
            samples.append((t,float(body.values[16]) if valid else np.nan,
                              float(body.values[4]) if valid else np.nan))
        self.samples = np.asarray(samples)
        if self.samples.ndim != 2 or not (np.diff(self.samples[:,0]) > 0).all():
            raise ValueError("Expected increasing native receipt times")
        self.memory = RoadBoundaryMemory(.60)
        self.reset()

    def reset(self):
        self.memory.reset()
        self.last = None

    def update(self, item, sections):
        t, frame, generation = item["received"], item["capture_id"], item["generation"]
        times = self.samples[:,0]
        index = int(np.searchsorted(times,t,side="right")-1)
        if index < 0 or t-times[index] > MAX_BODY_AGE_S or not np.isfinite(self.samples[index,1:]).all():
            self.reset()
            return sections, None
        if self.last is not None:
            clock, previous, prior_generation, j = self.last
            if previous+1 != frame or prior_generation != generation or not 0 < t-clock <= .18:
                self.reset()
            else:
                while clock < t-1e-10:
                    end = min(t,times[j+1]) if j+1 < len(times) else t
                    speed,yaw = self.samples[j,1:]
                    if end-times[j] > MAX_BODY_AGE_S or not np.isfinite([speed,yaw]).all():
                        self.memory.reset()
                    else:
                        self.memory.advance(float(speed),float(yaw),end-clock)
                    clock = end
                    if j+1 < len(times) and end == times[j+1]:
                        j += 1
        self.last = t, frame, generation, index
        return self.memory.update(sections), float(self.samples[index,1])


def choose_line(observed, remembered, speed, policy, *, clearance_time_budget=None):
    limits = dict(LIMITS, clearance_time_budget=clearance_time_budget)
    if policy == "all":
        line, status = plan_local_line(remembered, speed, **limits)
        return remembered, line, status
    line, status = plan_local_line(observed, speed, **limits)
    if policy == "contrast-near":
        if line is not None or not remembered[0]["valid"]:
            return observed, line, status
        recovery = [
            dict(section, valid=False)
            if section.get("source") == "carried"
            and (section["age_s"] > .24 or section["forward_x_m"] > 8.)
            else section
            for section in remembered
        ]
        line, status = plan_local_line(recovery, speed, **limits)
        return recovery, line, status
    if line is not None or not all(section["valid"] for section in observed[:2]):
        return observed, line, status
    fallback, fallback_status = plan_local_line(remembered, speed, **limits)
    if fallback is not None:
        used = [s for s in remembered if s["forward_x_m"] <= fallback["x_m"][-1]+1e-9]
        if all(s["source"] != "carried" or s["age_s"] <= .24 for s in used):
            return remembered, fallback, fallback_status
    return observed, line, status


def plan(archives, forecasts, features, camera, output, native=None, memory_policy="all",
         clearance_time_budget=None, front_far=False, corner_guided_line=False, extend_corner_line=False):
    if memory_policy != "all" and native is None:
        raise ValueError("An observation-supported memory policy requires native replay")
    proof = read(forecasts.with_name("result.json"))
    feature_proof = read(features)
    if (proof["rows_sha256"] != digest(forecasts)
            or digest(features) not in proof["inputs_sha256"].values()
            or feature_proof["camera_config_sha256"] != digest(camera)):
        raise ValueError("Forecasts, feature receipt or camera calibration differ")
    folders = {name: Path(path) for name, path in read(archives).items()}
    forecast_rows = [json.loads(line) for line in forecasts.read_text(encoding="utf8").splitlines()]
    indexed = {(r["run"], r["frame"]): r for r in forecast_rows}
    if len(indexed) != len(forecast_rows) or set(folders) != {run for run, _ in indexed}:
        raise ValueError("Expected unique forecasts with matching camera archives")
    cv2.setNumThreads(1)
    contrast = memory_policy == "contrast-near"
    if front_far and not contrast:
        raise ValueError("Front-only far boundaries require the contrast-near policy")
    if extend_corner_line and not front_far:
        raise ValueError("Corner line extension requires the front-far contrast-near policy")
    mount = read(camera)
    if contrast:
        detector = SurfaceContrastRoadBoundary(mount, **({"front_far": True} if front_far else {}))
    else:
        detector = PixelRoadBoundary(mount)
    episodes = {name: Path(path) for name,path in read(native).items()} if native else {}
    if native and set(episodes) != set(folders):
        raise ValueError("Native episodes must match camera archive names")
    native_inputs = [native] if native else []
    if contrast:
        native_inputs.append(Path("src/flybrain_drive/surface_road_boundary.py"))
    if extend_corner_line:
        native_inputs.append(Path("src/flybrain_drive/local_line_extension.py"))
    records = {}
    for name, folder in folders.items():
        motion = ArchivedMotion(folder,episodes[name],name) if native else None
        if motion:
            native_inputs.extend(motion.inputs)
        with CameraArchive(folder) as archive:
            if archive.bundle_sha256 != feature_proof["reports"][name]["bundle_sha256"]:
                raise ValueError("Camera bundle differs from inference source")
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key = name, item["capture_id"]
                forecast = indexed.get(key)
                if forecast is None and motion is None:
                    continue
                views = {view["view_id"]: view for view in item["views"]}
                record = dict(run=name, frame=key[1], preview_alert=bool(forecast and forecast["alert"]),
                              status="camera_unavailable", sections=None, line=None, preview_speed_mps=None)
                if forecast is not None:
                    records[key] = record
                if item["ideal_line_shown"] or item["aid_probe_only"] or set(views) != set(VIEW_ORDER):
                    if motion:
                        motion.reset()
                    continue
                images = {view: cv2.imdecode(np.frombuffer(archive.read(Path(meta["file"]).name),
                                                           np.uint8), cv2.IMREAD_COLOR)
                          for view, meta in views.items()}
                if any(image is None or image.shape != (180, 320, 3) for image in images.values()):
                    if motion:
                        motion.reset()
                    continue
                if contrast:
                    sections, observed = detector.line_sections(images)
                else:
                    sections = observed = detector.detect(images)
                speed = None
                if motion:
                    sections,speed = motion.update(item,sections)
                if forecast is None:
                    continue
                record.update(body_speed_mps=speed, maneuver=None)
                if motion and speed is None:
                    record.update(sections=sections,status="body_unavailable")
                    continue
                sections,line,status = choose_line(observed,sections,speed if speed is not None else 0.,
                                                   memory_policy, clearance_time_budget=clearance_time_budget)
                if corner_guided_line and line is not None and forecast["alert"]:
                    side = 1 if forecast["direction"] == 1 else -1
                    line, status = plan_local_line(sections, speed if speed is not None else 0.,
                                                   **LIMITS, clearance_time_budget=clearance_time_budget,
                                                   corner_preview=(side, forecast["distance_m"]))
                if extend_corner_line and forecast["alert"] and line is not None and line["x_m"][-1] == 20.:
                    extension = extend_line(line, detector.detect(images, extended=True), LIMITS)
                    if extension is not None:
                        line = extension
                        status = ("above_entry_speed_bound" if speed > line["speed_bound_mps"][0]
                                  else "within_entry_speed_bound")
                record.update(sections=sections, line=line, status=status)
                if line is not None and forecast["alert"]:
                    envelope = speed_bound_with_corner_preview(line, forecast["distance_m"],
                        corner_speed_mps=20/3.6, braking_mps2=LIMITS["braking_mps2"])
                    record["preview_speed_mps"] = envelope.tolist()
                if line is not None and speed is not None:
                    record["maneuver"] = maneuver_from_line(line,speed,lookahead_m=12.,
                        speed_bound=record["preview_speed_mps"]).tolist()
    if records.keys() != indexed.keys():
        raise ValueError("Some forecast captures were absent from the source archives")
    output.mkdir(parents=True, exist_ok=False)
    jsonl(output/"rows.jsonl", (records[key] for key in indexed))
    write(output/"result.json", dict(revision=proof["revision"], seed=None,
        n=len(records), paths=sum(r["line"] is not None for r in records.values()),
        inputs_sha256={str(p):digest(p) for p in (archives,forecasts,features,camera,Path(__file__),
                        Path("src/flybrain_drive/pixel_road_boundary.py"),
                        Path("src/flybrain_drive/local_racing_line.py"), *native_inputs)},
        rows_sha256=digest(output/"rows.jsonl"), limits=LIMITS, memory_policy=memory_policy,
        clearance_time_budget=clearance_time_budget, front_far=front_far, corner_guided_line=corner_guided_line, extend_corner_line=extend_corner_line,
        termination="archive-only geometric paths and optional speed envelopes produced"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archives", "forecasts", "features", "camera", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--memory-policy", choices=("all", "direct-first", "contrast-near"), default="all")
    parser.add_argument("--front-far", action="store_true", help="Use only the front view from 12m in contrast-near")
    parser.add_argument("--extend-corner-line", action="store_true",
                        help="Extend a 20m path after a corner alert when far boundaries support it")
    parser.add_argument("--corner-guided-line", action="store_true",
                        help="Select an experimental corner-guided geometric line when feasible")
    parser.add_argument("--clearance-time-budget", type=float,
                        help="Prioritize clearance within this fractional time-score increase")
    plan(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
