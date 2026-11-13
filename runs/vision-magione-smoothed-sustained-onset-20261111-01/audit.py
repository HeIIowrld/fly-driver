"""Offline centerline bend diagnostic. Labels are used only after prediction."""
import json
from collections import Counter
from scipy.spatial import cKDTree
from train_next_corner import CURVATURE_THRESHOLD
import train_vision_teacher_line as teacher
from train_vision_teacher_line import at
from audit_road_edge_projection import road_edges,section_points
from pathlib import Path

import cv2
import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from train_vision_teacher_line import digest, jsonl, read, write


class SurfaceContrast(PixelRoadBoundary):
    """Compare same-view strips either side of a candidate surface transition."""

    def __init__(self, mount, *, near=False):
        super().__init__(mount)
        self.stations = np.array([2.,4.,6.,8.]) if near else np.arange(8., 26., 2.)
        self.close = near
        self.lookup = self._sampling(mount, self.stations)

    def detect(self, images):
        sections = []
        center = 0.
        projected = []
        for name, (u, v, valid) in self.lookup.items():
            if not self.close and name != "front":
                continue
            image = cv2.cvtColor(images[name], cv2.COLOR_BGR2LAB)
            height, width = image.shape[:2]
            samples = image[np.rint(np.clip(v, 0, 1)*(height-1)).astype(int),
                            np.rint(np.clip(u, 0, 1)*(width-1)).astype(int)].astype(float)
            projected.append((name, samples, valid))
        candidates = np.arange(9, len(self.lateral)-9)
        for station, distance in enumerate(self.stations):
            offset = np.arange(1, 5) if self.close else np.arange(2, 9)
            boundaries = []
            for side in (-1, 1):
                inside = candidates[:, None]-side*offset
                outside = candidates[:, None]+side*offset
                scores = np.full(len(candidates), -np.inf)
                for name, samples, valid in projected:
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
                    score = chroma + .25*abs(difference[:, 0])
                    scores = np.maximum(scores, np.where(eligible, score, -np.inf))
                if np.isfinite(scores).any():
                    # Select the closest transition supported at three nearby lateral samples.
                    supported = np.convolve(np.isfinite(scores).astype(int), np.ones(3), mode="same") == 3
                    indices = candidates[supported]
                    edge = float(self.lateral[indices[0 if side > 0 else -1]]) if len(indices) else None
                else:
                    edge = None
                boundaries.append(edge)
            right, left = boundaries
            valid = right is not None and left is not None and 3. < left-right < 20.
            if valid and distance > 4.:
                center = .5*(left+right)
            sections.append(dict(forward_x_m=float(distance), right_y_m=right, left_y_m=left, valid=valid))
        return sections


from train_next_corner import BASE, MONZA

out=Path(__file__).parent
root=out.parent
source=root/"vision-monza-presence-temporal-ablation-20261111-01/rows.jsonl"
proof=read(source.with_name("result.json"))
assert digest(source)==proof["rows_sha256"]
config=Path("configs/ac-csp-exterior-front-v3-retina.json")
detector=SurfaceContrast(read(config))
close_detector=SurfaceContrast(read(config),near=True)
rows=[r for r in map(json.loads,source.read_text().splitlines()) if r["mode"]=="latest_and_changes" and r["cohort"]=="magione_development"]
indexed={(r["run"],r["frame"]):r for r in rows}
assert len(indexed)==len(rows)
qualified=[]
frozen=root/"vision-frozen-next-corner-20261110-01"
archive_list=frozen/"archives.json"
features=frozen/"archive-parity/original/flyvis-result.json"
qualified.extend((run,Path(folder),features,read(features)["reports"][run]["bundle_sha256"])
                 for run,folder in read(archive_list).items())
cv2.setNumThreads(1)

def onset_candidate(sections, minimum=10.):
    """Fit a shared straight-to-curved breakpoint to both detected edges."""
    valid=[sec for sec in sections if sec["valid"]]
    if len(valid)<5:
        return None
    x=np.array([sec["forward_x_m"] for sec in valid])
    y=np.array([[sec[side] for side in ("right_y_m","left_y_m")] for sec in valid])
    line=np.column_stack((np.ones(len(x)),(x-8.)/10.))
    baseline=line@np.linalg.lstsq(line,y,rcond=None)[0]
    residual=float(np.mean((y-baseline)**2))
    if residual<=1e-10:
        return None
    candidates=[]
    for onset in np.arange(minimum,21.):
        if sum(x<=onset)<2 or sum(x>onset)<3:
            continue
        design=np.column_stack((line,(np.maximum(x-onset,0.)/10.)**2))
        coef=np.linalg.lstsq(design,y,rcond=None)[0]
        if coef[2,0]*coef[2,1]<=0:
            continue
        error=float(np.mean((y-design@coef)**2))
        gain=1.-error/residual
        if gain<.50 or min(abs(coef[2]))<.5:
            continue
        candidates.append((error,float(onset),1 if coef[2,0]>0 else 2,gain))
    if not candidates:
        return None
    error,onset,direction,gain=min(candidates)
    return dict(distance_m=onset,direction=direction,gain=gain,residual_m=error**.5)


thresholds=(.002,.004,.008)

def smoothed_onsets(sections, radius):
    """Estimate curvature from physical windows with observations on both sides."""
    valid=[sec for sec in sections if sec["valid"]]
    x=np.array([sec["forward_x_m"] for sec in valid])
    y=np.array([[sec["right_y_m"],sec["left_y_m"]] for sec in valid])
    if len(x)<4:
        return dict.fromkeys(map(str,thresholds))
    samples=[]
    for station in np.arange(4.,22.):
        offset=x-station
        mask=abs(offset)<=radius
        u=offset[mask]
        if len(u)<4 or not (np.any(u<0) and np.any(u>0)) or max(np.diff(u))>4.:
            continue
        basis=np.column_stack((np.ones(len(u)),u,u*u))
        coef=np.linalg.lstsq(basis,y[mask],rcond=None)[0]
        k=2.*coef[2]/(1.+coef[1]**2)**1.5
        value=float(np.copysign(min(abs(k)),k[0])) if k[0]*k[1]>0 else 0.
        samples.append((float(station),value))
    result={}
    for cut in thresholds:
        result[str(cut)]=None
        for (x0,k0),(x1,k1) in zip(samples,samples[1:]):
            if x0>20.:
                break
            if x1-x0==1. and abs(k0)>=cut and abs(k1)>=cut and k0*k1>0:
                result[str(cut)]=dict(distance_m=x0,direction=1 if k0>0 else 2)
                break
    return result


def geometric_evidence(sections):
    bends=[]
    for a,b,c in zip(sections,sections[2:],sections[4:]):
        if not all(s["valid"] for s in (a,b,c)) or b["forward_x_m"]>20.:
            continue
        x=np.array([s["forward_x_m"] for s in (a,b,c)])
        edges=np.array([[s[side] for s in (a,b,c)] for side in ("right_y_m","left_y_m")])
        turn=np.diff(np.arctan(np.diff(edges,axis=1)/np.diff(x)),axis=1).ravel()
        if turn[0]*turn[1]>0:
            bends.append((x[1],float(np.copysign(min(abs(turn)),turn[0]))))
    adjacent=max((min(abs(a),abs(b)) for (x,a),(y,b) in zip(bends,bends[1:])
                  if y-x==2. and a*b>0),default=0.)
    return dict(bends=bends,peak=max((abs(a) for _,a in bends),default=0.),adjacent=adjacent,
                valid_sections=sum(s["valid"] for s in sections),onset=onset_candidate(sections))

def carried_section(history, x):
    """Interpolate both sides only between earlier observed 8m points."""
    if len(history)<2:
        return None
    values=[]
    for side in range(2):
        points=np.asarray([points[side] for _,points in history])
        order=np.argsort(points[:,0])
        points=points[order]
        i=int(np.searchsorted(points[:,0],x))
        if i==0 or i==len(points):
            return None
        (a,y0),(b,y1)=points[i-1:i+1]
        if not a<=x<=b or b-a>2.5:
            return None
        values.append(float(y0+(y1-y0)*(x-a)/(b-a)))
    right,left=values
    if not 3.<left-right<20.:
        return None
    return dict(forward_x_m=x,right_y_m=right,left_y_m=left,valid=True)

geometry=BASE/"heldout-magione-geometry-01/geometry.json"
assert digest(geometry)==read(geometry.with_name("result.json"))["geometry_sha256"]
reference_edges=road_edges(read(geometry)["rows"])
ages=(.85,)
inventory=out.parent/"vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl"
inventory_proof=read(inventory.with_name("result.json"))
assert digest(inventory)==inventory_proof["rows_sha256"]
reference_script=out.parent/"vision-corner-label-geometry-20261110-01/probe.py"
teacher.CORNER=CURVATURE_THRESHOLD
code=reference_script.read_text(encoding="utf-8-sig")
exec(compile(code[code.index("def periodic_line("):code.index("def near(")],str(reference_script),"exec"))
length,spans,stations,tree=periodic_line(np.mean(reference_edges,axis=0)[:,[2,0]])
assert [list(s) for s in spans]==inventory_proof["spans"]
labels={(r["run"],r["frame"]):r for r in map(json.loads,inventory.read_text().splitlines())}
def oracle_sections(pose, origin, left, distances):
    boundaries=(section_points(reference_edges,pose,distances)-origin)[:,:,[2,0]]@left
    return [dict(forward_x_m=float(x),left_y_m=float(y[0]),right_y_m=float(y[1]),valid=bool(np.isfinite(y).all()))
            for x,y in zip(distances,boundaries,strict=True)]

evidence={}
for run,folder,receipt,sha in qualified:
    wanted={f for name,f in indexed if name==run}
    start,end=min(wanted),max(wanted)
    history=[]
    previous=None
    with CameraArchive(folder) as archive:
        assert archive.bundle_sha256==sha
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            frame=item["capture_id"]
            key=run,frame
            if frame<start-20 or frame>end:
                continue
            if item["ideal_line_shown"] or item["aid_probe_only"]:
                history.clear()
                previous=None
                continue
            meta=archive.json(Path(item["file"]).name)
            assert meta["index"]==frame and meta["track_id"]=="magione"
            pose=max(meta["views"],key=lambda v:v["render_call_finished_clock"])["source_body_after_render"]
            t=float(pose["source_clock_finished"])
            omega=pose.get("local_angular_velocity")
            speed=pose.get("speed_m_s")
            valid=bool(pose["physics_available"] and omega is not None and speed is not None
                       and np.isfinite([t,speed,omega[1]]).all() and speed>=0)
            contiguous=False
            if previous is not None and valid:
                old,generation,t0,speed0,yaw0=previous
                dt=t-t0
                contiguous=frame==old+1 and generation==item["generation"] and .045<dt<=.18
                if contiguous:
                    turn=yaw0*dt
                    move=speed0*dt*np.array([np.cos(turn/2),np.sin(turn/2)])
                    c,s=np.cos(turn),np.sin(turn)
                    rotate=np.array([[c,-s],[s,c]])
                    history=[(stamp,capture,(points-move)@rotate) for stamp,capture,points in history if t-stamp<=max(ages)]
            if not contiguous:
                history.clear()
            previous=(frame,item["generation"],t,speed,omega[1]) if valid else None
            images={v["view_id"]:cv2.imdecode(np.frombuffer(
                archive.read(Path(v["file"]).name),np.uint8),cv2.IMREAD_COLOR) for v in item["views"]}
            near=close_detector.detect(images)
            current8=near[-1]
            if key in indexed:
                far=detector.detect({"front":images["front"]})[1:]
                origin=np.asarray(pose["renderer_only_position"])
                forward=np.asarray(pose["renderer_only_look"])[[2,0]]
                forward/=np.linalg.norm(forward)
                left=np.array([-forward[1],forward[0]])
                points=section_points(reference_edges,pose,(4.,6.))
                lateral=(points-origin)[:,:,[2,0]]@left
                choices={}
                for age in ages:
                    history_slice=[(stamp,pts) for stamp,capture,pts in history
                                   if t-stamp<=age and (age!=.45 or capture>=start-8)]
                    carried={x:carried_section(history_slice,x) for x in (4.,6.)}
                    combined=[near[0],*(sec for sec in carried.values() if sec is not None),current8,*far]
                    scores={}
                    for i,x in enumerate((4.,6.)):
                        sec=carried[x]
                        error=None
                        if sec is not None and np.isfinite(lateral[i]).all():
                            error=float(max(abs(np.array([sec["left_y_m"],sec["right_y_m"]])-lateral[i])))
                        scores[str(int(x))]=dict(available=sec is not None,max_error_m=error)
                    observed_x=[sec["forward_x_m"] for sec in combined if sec["valid"]]
                    options={f"video_{radius:g}":smoothed_onsets(combined,radius) for radius in (4.,6.)}
                    if indexed[key]["near"]:
                        same=oracle_sections(pose,origin,left,observed_x)
                        dense=oracle_sections(pose,origin,left,np.arange(2.,25.))
                        options.update({f"{name}_{radius:g}":smoothed_onsets(sections,radius) for name,sections in (("supported",same),("dense",dense)) for radius in (4.,6.)})
                    choices[str(age)]=dict(onset=onset_candidate(combined,4.),edges=scores,options=options)
                evidence[key]=choices
            if valid and current8["valid"]:
                history.append((t,frame,np.array([[8.,current8["right_y_m"]],[8.,current8["left_y_m"]]])))
assert evidence.keys()==indexed.keys()

rescue=root/"vision-corner-causal-geometry-rescue-20261111-01/rows.jsonl"
assert digest(rescue)==read(rescue.with_name("result.json"))["rows_sha256"]
comparison={(r["run"],r["frame"]):r for r in map(json.loads,rescue.read_text().splitlines())}
assert indexed.keys()<=comparison.keys()
comparison={key:comparison[key] for key in indexed}
reference=out.parent/"vision-magione-near-edge-memory-duration-20261111-01/rows.jsonl"
assert digest(reference)==read(reference.with_name("result.json"))["rows_sha256"]
prior={(r["run"],r["frame"]):r for r in map(json.loads,reference.read_text().splitlines())}
assert indexed.keys()<=prior.keys()
reports=[]
for key,r in comparison.items():
    item=evidence[key]["0.85"]
    assert item["onset"]==prior[key]["choices"]["0.85"]["onset"],key
    assert item["edges"]==prior[key]["choices"]["0.85"]["edges"],key
    old=r["policies"]["frozen"]
    geo=r["policies"]["geometry"]
    reports.append(dict(run=key[0],frame=key[1],actual=r["actual"],truth_m=r["truth_distance"],
        near=r["near"],previous_geometry_rescue=bool(not old and geo and geo["direction"]==r["actual"]),
        frozen=old,geometry=geo,**item))

def summarize(group,get):
    hits=[]
    outside=0
    for r in group:
        option=get(r)
        if option is None:
            continue
        if not r["near"]:
            outside+=1
        elif option["direction"]==r["actual"]:
            hits.append(option["distance_m"]-r["truth_m"])
    return dict(direction_hits=len(hits),within_5m=sum(abs(e)<=5. for e in hits),
        outside=outside,bias_m=float(np.median(hits)) if hits else None)

summary={"frozen":summarize(reports,lambda r:r["frozen"]),"previous_fit":summarize(reports,lambda r:r["onset"])}
for cut in map(str,thresholds):
    summary[cut]={}
    for name,group in (("all",reports),("under_10m",[r for r in reports if r["near"] and r["truth_m"]<10.]),
                       ("near_previous_rescue",[r for r in reports if r["near"] and r["truth_m"]<10.
                                                                         and r["previous_geometry_rescue"]])):
        result={mode:summarize(group,lambda r:r["options"].get(mode,{}).get(cut))
                for mode in ("video_4","video_6","supported_4","supported_6","dense_4","dense_6")}
        for mode in ("video_4","video_6"):
            def supplement(r):
                option=r["options"][mode][cut]
                confirmation=r["geometry"]
                return r["frozen"] or (option if option and confirmation and option["direction"]==confirmation["direction"] else None)
            result["frozen_plus_"+mode]=summarize(group,supplement)
        summary[cut][name]=result
jsonl(out/"rows.jsonl",reports)
write(out/"result.json",dict(summary=summary,seed=None,termination="archive_end",
    revision=proof["revision"],rows_sha256=digest(out/"rows.jsonl"),diagnostic_reference_only=True,
    hashes={str(p):digest(p) for p in (source,config,archive_list,rescue,reference,geometry,
                                      inventory,reference_script,Path(__file__),
                                      *(receipt for _,_,receipt,_ in qualified))}))
for cut in map(str,thresholds):
    print(cut,summary[cut]["all"])
