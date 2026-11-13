"""Offline centerline bend diagnostic. Labels are used only after prediction."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
from train_vision_teacher_line import digest, jsonl, read, write


class SurfaceContrast(PixelRoadBoundary):
    """Compare same-view strips either side of a candidate surface transition."""

    def __init__(self, mount):
        super().__init__(mount)
        self.stations = np.arange(8., 26., 2.)
        self.lookup = self._sampling(mount, self.stations)

    def detect(self, images):
        sections = []
        center = 0.
        projected = []
        for name, (u, v, valid) in self.lookup.items():
            if name != "front":
                continue
            image = cv2.cvtColor(images[name], cv2.COLOR_BGR2LAB)
            height, width = image.shape[:2]
            samples = image[np.rint(np.clip(v, 0, 1)*(height-1)).astype(int),
                            np.rint(np.clip(u, 0, 1)*(width-1)).astype(int)].astype(float)
            projected.append((name, samples, valid))
        candidates = np.arange(9, len(self.lateral)-9)
        for station, distance in enumerate(self.stations):
            offset = np.arange(1, 5) if distance == 4. else np.arange(2, 9)
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
rows=[r for r in map(json.loads,source.read_text().splitlines()) if r["mode"]=="latest_and_changes"]
indexed={(r["run"],r["frame"]):r for r in rows}
assert len(indexed)==len(rows)
qualified=[(run,BASE/run/"demonstration/camera/vision",BASE/run/sub/"result.json",
            read(BASE/run/sub/"result.json")["camera_bundle_sha256"]) for run,sub in MONZA]
frozen=root/"vision-frozen-next-corner-20261110-01"
archive_list=frozen/"archives.json"
features=frozen/"archive-parity/original/flyvis-result.json"
qualified.extend((run,Path(folder),features,read(features)["reports"][run]["bundle_sha256"])
                 for run,folder in read(archive_list).items())
cv2.setNumThreads(1)

def onset_candidate(sections):
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
    for onset in np.arange(10.,21.):
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

evidence={}
for run,folder,receipt,sha in qualified:
    with CameraArchive(folder) as archive:
        assert archive.bundle_sha256==sha
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            key=run,item["capture_id"]
            if key not in indexed:
                continue
            assert not item["ideal_line_shown"] and not item["aid_probe_only"]
            front=next(v for v in item["views"] if v["view_id"]=="front")
            image=cv2.imdecode(np.frombuffer(archive.read(Path(front["file"]).name),np.uint8),cv2.IMREAD_COLOR)
            evidence[key]=geometric_evidence(detector.detect({"front":image}))
assert evidence.keys()==indexed.keys()

rescue=root/"vision-corner-causal-geometry-rescue-20261111-01/rows.jsonl"
assert digest(rescue)==read(rescue.with_name("result.json"))["rows_sha256"]
comparison={(r["run"],r["frame"]):r for r in map(json.loads,rescue.read_text().splitlines())}
assert comparison.keys()==indexed.keys()
reports=[]
for key,r in comparison.items():
    item=evidence[key]
    old=r["policies"]["frozen"]
    geometric=r["policies"]["geometry"]
    new=item["onset"]
    # All original alerts remain in place. A fit may supply a new onset only
    # after the separate two-capture bend direction check has passed.
    replacement=old or (new if geometric and new and geometric["direction"]==new["direction"] else None)
    reports.append(dict(run=key[0],frame=key[1],cohort=r["cohort"],near=r["near"],actual=r["actual"],
        truth_distance=r["truth_distance"],onset=new,frozen=old,geometry=geometric,segmented=replacement))

def summarize(rows,policy):
    metrics=Counter()
    errors=[]
    for r in rows:
        p=r[policy]
        if p is None:
            continue
        if not r["near"]:
            metrics["outside"]+=1
        elif p["direction"]==r["actual"]:
            error=p["distance_m"]-r["truth_distance"]
            metrics["direction_hit"]+=1
            metrics["within_5m"]+=abs(error)<=5.
            if r["frozen"] is None:
                errors.append(error)
    return dict(metrics,new_hit_n=len(errors),new_bias_m=float(np.median(errors)) if errors else None,
                new_mae_m=float(np.mean(np.abs(errors))) if errors else None)

summary={}
for name in sorted({r["cohort"] for r in reports}):
    cohort=[r for r in reports if r["cohort"]==name]
    summary[name]={p:summarize(cohort,p) for p in ("frozen","geometry","segmented","onset")}
    added=[r for r in cohort if r["near"] and r["frozen"] is None and r["geometry"] is not None
           and r["geometry"]["direction"]==r["actual"]]
    summary[name]["previously_added_under_10m"]=sum(r["truth_distance"]<10. for r in added)
    paired=[r for r in added if r["segmented"] is not None]
    summary[name]["common_new_cases"]=dict(n=len(paired),**{
        p:float(np.mean([abs(r[p]["distance_m"]-r["truth_distance"]) for r in paired])) if paired else None
        for p in ("geometry","segmented")})
jsonl(out/"rows.jsonl",reports)
write(out/"result.json",dict(summary=summary,seed=None,termination="archive_end",
    revision=proof["revision"],rows_sha256=digest(out/"rows.jsonl"),
    hashes={str(p):digest(p) for p in (source,config,archive_list,rescue,Path(__file__),
                                      *(receipt for _,_,receipt,_ in qualified))}))
print(summary)
