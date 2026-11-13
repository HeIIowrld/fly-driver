"""Evaluate image-only local lines across archived Monza corner phases."""
import json
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.local_racing_line import plan_local_line
from extend import extend_line
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from flybrain_drive.road_boundary_memory import RoadBoundaryMemory
from plan_frozen_corner_archive import LIMITS
from train_next_corner import BASE, MONZA, CURVATURE_THRESHOLD
import train_vision_teacher_line as teacher
from train_vision_teacher_line import at, digest, edge_polygons, jsonl, read, write


class SurfaceContrastRoadBoundary(PixelRoadBoundary):
    """Compare same-view strips either side of a candidate surface transition."""

    def __init__(self, mount):
        super().__init__(mount)
        self.stations = np.array([4., 8., 10., 12., 15., 20., 24., 28., 32.])
        self.lookup = self._sampling(mount, self.stations)

    def detect(self, images):
        baseline = super().detect(images) + [None] * 3
        center = 0.
        projected = []
        for name, (u, v, valid) in self.lookup.items():
            image = cv2.cvtColor(images[name], cv2.COLOR_BGR2LAB)
            height, width = image.shape[:2]
            samples = image[np.rint(np.clip(v, 0, 1)*(height-1)).astype(int),
                            np.rint(np.clip(u, 0, 1)*(width-1)).astype(int)].astype(float)
            projected.append((name, samples, valid))
        candidates = np.arange(9, len(self.lateral)-9)
        for station, (distance, section_index) in enumerate(zip(self.stations, (0, 2, 3, 4, 5, 6, 7, 8, 9), strict=True)):
            offset = np.arange(1, 5) if distance == 4. else np.arange(2, 9)
            boundaries = []
            for side in (-1, 1):
                inside = candidates[:, None]-side*offset
                outside = candidates[:, None]+side*offset
                supported = np.zeros(len(candidates), dtype=bool)
                for name, samples, valid in projected:
                    if distance >= 12. and name != "front":
                        continue
                    row = samples[station]
                    inner, outer = row[inside], row[outside]
                    a, b = np.median(inner, axis=1), np.median(outer, axis=1)
                    difference = b-a
                    chroma = np.linalg.norm(difference[:, 1:], axis=1)
                    inner_chroma = np.linalg.norm(a[:, 1:]-128., axis=1)
                    outer_chroma = np.linalg.norm(b[:, 1:]-128., axis=1)
                    support = valid[station, inside].all(1) & valid[station, outside].all(1)
                    eligible = support & (inner_chroma <= 13) & (outer_chroma-inner_chroma >= 6)
                    eligible &= (chroma >= 9) | ((chroma >= 6) & (abs(difference[:, 0]) >= 16))
                    eligible &= (self.lateral[candidates]-center)*side >= 1.3
                    supported |= eligible
                # Select the nearest transition supported at three lateral samples.
                contiguous = np.convolve(supported.astype(int), np.ones(3), mode="same") == 3
                indices = candidates[contiguous]
                edge = float(self.lateral[indices[0 if side > 0 else -1]]) if len(indices) else None
                boundaries.append(edge)
            right, left = boundaries
            if distance == 4.:
                original = baseline[0]
                right = original["right_y_m"] if original["right_y_m"] is not None else right
                left = original["left_y_m"] if original["left_y_m"] is not None else left
            valid = right is not None and left is not None and 3. < left-right < 20.
            if valid and distance > 4.:
                center = .5*(left+right)
            baseline[section_index] = dict(forward_x_m=float(distance), right_y_m=right, left_y_m=left, valid=valid)
        return baseline

    def line_sections(self, images):
        """Return the stations used by memory and the directly observed fallback."""
        detected = self.detect(images)
        observed = [section for section in detected if section["forward_x_m"] not in (6., 10.)]
        if observed[0]["valid"] or not detected[1]["valid"] or not observed[1]["valid"]:
            return observed, observed
        grass, visible = self._classify(images, self.near_lut)
        near = self._sections(self.near, grass, visible, near=True)[1]
        fallback = ([near] if near["valid"] else []) + [detected[1]] + observed[1:]
        return observed, fallback


out=Path(__file__).parent
source=out.parent/"vision-next-corner-20261110-03/dataset.json"
reference=out.parent/"vision-corner-label-geometry-20261110-01/probe.py"
track=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
config=Path("configs/ac-csp-exterior-front-v3-retina.json")
assert digest(track)==read(source)["monza_map_sha256"]
teacher.CORNER=CURVATURE_THRESHOLD
code=reference.read_text(encoding="utf-8-sig")
exec(compile(code[code.index("def periodic_line("):code.index("def near(")],str(reference),"exec"))
outer,inner=edge_polygons(track)
data=np.genfromtxt(track,names=True,delimiter=",")
center=np.c_[(data["left_border_x"]+data["right_border_x"])/2,
             (data["left_border_y"]+data["right_border_y"])/2]
length,spans,stations,tree=periodic_line(center)
samples=tree.data
detector=SurfaceContrastRoadBoundary(read(config))
short=out.parent/"vision-monza-corner-guided-replay-20261111-01"
forecast_path=short/"forecasts.jsonl"
assert digest(forecast_path)==read(short/"forecasts-result.json")["rows_sha256"]
forecasts={(r["label"]["run"],r["label"]["frame"]):r for line in
           forecast_path.read_text(encoding="utf8").splitlines() if (r:=json.loads(line))}
short_rows=short/"rows.jsonl"
assert digest(short_rows)==read(short/"result.json")["rows_sha256"]
baseline={(r["run"],r["frame"]):r for line in short_rows.read_text(encoding="utf8").splitlines()
          if (r:=json.loads(line))["policy"]=="guided"}
full=out.parent/"vision-monza-full-horizon-alerts-20261111-01"
all_forecasts=full/"forecasts.jsonl"
assert digest(all_forecasts)==read(full/"forecast-result.json")["rows_sha256"]
alerts={(r["run"],r["frame"]):r["alert"] for line in all_forecasts.read_text(encoding="utf8").splitlines()
        if (r:=json.loads(line))["available"]}
rows=[]
inputs=[]
cv2.setNumThreads(1)
for run,qualification in MONZA:
    episode=BASE/run
    memories={sign:RoadBoundaryMemory(.24) for sign in (1,)}
    previous=None
    qual=episode/qualification
    receipt=qual/"result.json"
    pairings=qual/"pairings.json"
    proof=read(receipt)
    assert digest(pairings)==proof["pairings_sha256"]
    inputs.extend((receipt,pairings))
    allowed={p["camera_index"] for p in read(pairings) if p["eligible"]}
    with CameraArchive(episode/"demonstration/camera/vision") as archive:
        assert archive.bundle_sha256==proof["camera_bundle_sha256"]
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            frame=item["capture_id"]
            if frame not in allowed or item["ideal_line_shown"] or item["aid_probe_only"]:
                continue
            meta=archive.json(Path(item["file"]).name)
            assert meta["index"]==frame and meta["track_id"]=="monza"
            pose=max(meta["views"],key=lambda v:v["render_call_finished_clock"])["source_body_after_render"]
            origin=np.asarray(pose["renderer_only_position"])[[2,0]]
            forward=np.asarray(pose["renderer_only_look"])[[2,0]]
            forward/=np.linalg.norm(forward)
            residual,i=tree.query(origin)
            tangent=samples[(i+1)%len(samples)]-samples[i-1]
            if residual>7 or tangent@forward<.7*np.linalg.norm(tangent):
                continue
            station=stations[i]
            options=[(event,station-a,b-a) for event,(a,b,_) in enumerate(spans)
                     if -20<=station-a<=b-a+10]
            if not options:
                continue
            event,relative,span=min(options,key=lambda v:abs(v[1]-v[2]/2))
            phase="approach" if relative<0 else "early_turn" if relative<span/3 else (
                "mid_turn" if relative<2*span/3 else "late_turn" if relative<span else "exit")
            images={v["view_id"]:cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name),np.uint8),
                                               cv2.IMREAD_COLOR) for v in item["views"]}
            detected=detector.detect(images)
            standard=[s for s in detected if s["forward_x_m"] not in (6.,10.)]
            plans={"standard":standard}
            if not standard[0]["valid"] and detected[1]["valid"] and standard[1]["valid"]:
                # With no 4m observation, use the nearer observed 2m section only if 6m is also supported.
                near=detector._sections(detector.near,*detector._classify(images,detector.near_lut),near=True)[1]
                plans["fallback"]=([near] if near["valid"] else [])+[detected[1]]+standard[1:]
            else:
                plans["fallback"]=standard
            t=float(pose["source_clock_finished"])
            omega=pose.get("local_angular_velocity")
            speed=pose.get("speed_m_s")
            body_valid=bool(pose["physics_available"] and omega is not None and speed is not None
                            and np.isfinite([t,speed,omega[1]]).all() and speed>=0)
            yaw_delta=None
            prior_yaw=None
            if previous and body_valid:
                frame0, gen0, t0, speed0, omega0, heading0=previous
                dt=t-t0
                if frame==frame0+1 and gen0==item["generation"] and .045<dt<=.18:
                    count=int(np.ceil(dt/.08))
                    for sign,memory in memories.items():
                        for _ in range(count):
                            memory.advance(speed0, sign*omega0, dt/count)
                    yaw_delta=float(np.arctan2(heading0[0]*forward[1]-heading0[1]*forward[0],heading0@forward))
                    prior_yaw=omega0*dt
            if yaw_delta is None:
                for memory in memories.values():
                    memory.reset()
            previous=(frame,item["generation"],t,speed,omega[1],forward) if body_valid else None
            for sign,memory in memories.items():
                remembered=memory.update(standard) if body_valid else standard
                plans[f"memory_{sign}"]=remembered if remembered[0]["valid"] else plans["fallback"]
            direct_line,direct_status=plan_local_line(plans["fallback"],0.,**LIMITS)
            for max_age in (.12,.24):
                recovery=[]
                for section in plans["memory_1"]:
                    if section.get("source")=="carried" and (
                        section["age_s"]>max_age or section["forward_x_m"]>8.
                    ):
                        recovery.append(dict(section,valid=False))
                    else:
                        recovery.append(section)
                plans[f"gated_{max_age}"]=plans["fallback"] if direct_line is not None else recovery
            sections=plans["gated_0.24"]
            original=baseline[run,frame]
            rows.append(dict(original,policy="short"))
            forecast=forecasts.get((run,frame))
            guarded=[]
            width0=previous_width=None
            for section in sections:
                station=section["forward_x_m"]
                if not section["valid"]:
                    guarded.append(section)
                    break
                if station>=20.:
                    width=section["left_y_m"]-section["right_y_m"]
                    if station==20.:
                        width0=width
                    if previous_width is not None and (width>previous_width+1.5 or width>width0+2.5):
                        guarded.append(dict(section,valid=False))
                        break
                    previous_width=width
                guarded.append(section)
            for policy,use in (("guarded",guarded),("alert_guarded",guarded if alerts.get((run,frame),False) else [])):
                line=extend_line(original["line"],use,LIMITS)
                if line is None:
                    line=original["line"]
                elif original["line"] is not None:
                    prefix=len(original["line"]["x_m"])
                    for field in ("x_m","y_m"):
                        assert original["line"][field]==line[field][:prefix]
                status=original["status"]
                margin=None
                if line is not None:
                    left=np.array([-forward[1],forward[0]])
                    world=origin+np.asarray(line["x_m"])[:,None]*forward+np.asarray(line["y_m"])[:,None]*left
                    margin=min(min(cv2.pointPolygonTest(outer,tuple(p),True),
                                   -cv2.pointPolygonTest(inner,tuple(p),True)) for p in world.tolist())
                rows.append(dict(run=run,frame=frame,event=event,station_relative_m=float(relative),
                                 phase=phase,status=status,map_margin_m=margin,policy=policy,
                                 first_constrained_m=line["first_constrained_x_m"] if line else None,
                                 yaw_delta=yaw_delta, prior_yaw=prior_yaw, received=item["received"], generation=item["generation"],
                                 split=forecast["label"]["split"] if forecast else None, line=line))
summary={}
phase_order=("approach","early_turn","mid_turn","late_turn","exit")
for policy in ("short","guarded","alert_guarded"):
    summary[policy]={}
    for run,_ in MONZA:
        grouped=defaultdict(list)
        subset=[r for r in rows if r["run"]==run and r["policy"]==policy]
        for row in subset:
            grouped[row["event"]].append(row)
        events={event:{phase:dict(frames=sum(r["phase"]==phase for r in group),
            bounded_paths=sum(r["phase"]==phase and r["map_margin_m"] is not None and r["map_margin_m"]>=0 for r in group))
            for phase in phase_order} for event,group in grouped.items()}
        summary[policy][run]=dict(n=len(subset), bounded=sum(r["map_margin_m"] is not None and r["map_margin_m"]>=0 for r in subset),
            outside=sum(r["map_margin_m"] is not None and r["map_margin_m"]<0 for r in subset),
            bounded_all_phases=sum(all(p["bounded_paths"] for p in e.values()) for e in events.values()),
            status=dict(Counter(r["status"] for r in subset)),events=events)
before={(r["run"],r["frame"]):r for r in rows if r["policy"]=="short"}
after={(r["run"],r["frame"]):r for r in rows if r["policy"]=="alert_guarded"}
assert before.keys()==after.keys()
paired_summary={}
for split in ("train","validation"):
    cases=[(before[k],r) for k,r in after.items() if r["split"]==split]
    changed=[(a,b) for a,b in cases if a["line"]!=b["line"]]
    paired_summary[split]=dict(frames=len(cases),changed=len(changed),
        changed_inside_before=sum(a["map_margin_m"] is not None and a["map_margin_m"]>=0 for a,b in changed),
        changed_inside_after=sum(b["map_margin_m"] is not None and b["map_margin_m"]>=0 for a,b in changed),
        extended_past_20m=sum(b["line"] is not None and b["line"]["x_m"][-1]>20. for a,b in cases))
jsonl(out/"rows.jsonl",rows)
write(out/"result.json",dict(summary=summary,paired_summary=paired_summary,seed=None,termination="archive_end",
    revision=read(out.parent/"vision-monza-midpoint-raw-head-20261110-01/result.json")["revision"],
    geometry_evaluation_only=True, body_source="recorded CSP local speed and angular rate; replay only, actor timing unverified", phase_labels="road centerline curvature span thirds; mid-turn is not apex",
    hashes={str(p):digest(p) for p in (track,source,reference,config,forecast_path,short_rows,all_forecasts,Path(__file__),out/"extend.py",*inputs)},
    rows_sha256=digest(out/"rows.jsonl")))
print({policy:{run:{k:v for k,v in r.items() if k!="events"} for run,r in data.items()} for policy,data in summary.items()})

print(paired_summary,flush=True)
