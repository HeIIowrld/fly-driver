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
                valid_sections=sum(s["valid"] for s in sections))

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

truth_files=(root/"vision-monza-corner-event-readout-20261111-01/rows.jsonl",
             root/"vision-magione-left-readout-stages-20261111-01/rows.jsonl")
eligible={}
for path in truth_files:
    assert digest(path)==read(path.with_name("result.json"))["rows_sha256"]
    eligible.update({(r["run"],r["frame"]):r["in_range"] for r in map(json.loads,path.read_text().splitlines())})
assert eligible.keys()==indexed.keys()
negative=root/"vision-monza-presence-negative-captures-20261111-01/rows.jsonl"
assert digest(negative)==read(negative.with_name("result.json"))["rows_sha256"]
straight={(r["run"],r["frame"]) for r in map(json.loads,negative.read_text().splitlines())
          if r["kind"]=="onset_beyond_60m"}

train=[r for r in rows if r["cohort"]=="monza_train_oof"]
budget=proof["train_outside_budget"]
base_cut=proof["summary"]["latest_and_changes"]["threshold"]
for r in rows:
    key=r["run"],r["frame"]
    r.update(evidence[key],eligible=eligible[key],previous_straight=key in straight)

def counts(group, alerts):
    return dict(left=sum(a and r["near"] and r["actual"]==1 for r,a in zip(group,alerts,strict=True)),
                right=sum(a and r["near"] and r["actual"]==2 for r,a in zip(group,alerts,strict=True)),
                outside=sum(a and not r["near"] for r,a in zip(group,alerts,strict=True)),
                straight_recurrences=sum(a and r["previous_straight"] for r,a in zip(group,alerts,strict=True)))

def measure(group,kind,minimum,cut):
    return counts(group,[r["eligible"] and r["score"]>=cut and r[kind]>=minimum for r in group])

report={}
for kind in ("peak","adjacent"):
    comparisons={}
    for minimum in (.005,.012,.025,.05,.10):
        cuts=sorted({r["score"] for r in train if r[kind]>=minimum})
        cuts=[*cuts,np.nextafter(max(r["score"] for r in train),np.inf)]
        candidates=[]
        for cut in cuts:
            result=measure(train,kind,minimum,cut)
            if result["outside"]<=budget:
                balance=(min(result["left"]/18,result["right"]/34),result["left"]+result["right"],-result["outside"],cut)
                candidates.append((balance,cut))
        _,selected=max(candidates)
        comparisons[str(minimum)]={name:dict(baseline=counts(group,[r["alert"] for r in group]),
            same_cut=measure(group,kind,minimum,base_cut),refit_cut=measure(group,kind,minimum,selected))
            for name in sorted({r["cohort"] for r in rows}) for group in [[r for r in rows if r["cohort"]==name]]}
        comparisons[str(minimum)]["threshold"]=float(selected)
    report[kind]=comparisons
jsonl(out/"rows.jsonl",rows)
write(out/"result.json",dict(summary=report,seed=None,termination="archive_end",
    revision=proof["revision"],rows_sha256=digest(out/"rows.jsonl"),
    hashes={str(p):digest(p) for p in (source,config,archive_list,negative,Path(__file__),
                                      *truth_files,*(receipt for _,_,receipt,_ in qualified))}))
for kind,group in report.items():
    for minimum,values in group.items():
        print(kind,minimum,values)
