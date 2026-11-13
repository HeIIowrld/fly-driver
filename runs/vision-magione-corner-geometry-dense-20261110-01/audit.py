"""Offline centerline bend diagnostic. Labels are used only after prediction."""
import json
import subprocess
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
            image = cv2.cvtColor(images[name], cv2.COLOR_BGR2LAB)
            height, width = image.shape[:2]
            samples = image[np.rint(np.clip(v, 0, 1)*(height-1)).astype(int),
                            np.rint(np.clip(u, 0, 1)*(width-1)).astype(int)].astype(float)
            projected.append((samples, valid))
        candidates = np.arange(9, len(self.lateral)-9)
        for station, distance in enumerate(self.stations):
            offset = np.arange(1, 5) if distance == 4. else np.arange(2, 9)
            boundaries = []
            for side in (-1, 1):
                inside = candidates[:, None]-side*offset
                outside = candidates[:, None]+side*offset
                scores = np.full(len(candidates), -np.inf)
                for samples, valid in projected:
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


out = Path(__file__).parent
root = Path("runs/vision-frozen-next-corner-20261110-01")
inventory = Path("runs/vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl")
forecast = root / "archive-inference/rows.jsonl"
archives = root / "archives.json"
features = root / "archive-parity/original/flyvis-result.json"
config = Path("configs/ac-csp-exterior-front-v3-retina.json")
def indexed(path):
    assert digest(path) == read(path.with_name("result.json"))["rows_sha256"]
    return {(r["run"],r["frame"]): r for r in map(json.loads,path.read_text(encoding="utf8").splitlines())}

labels, previous = indexed(inventory), indexed(forecast)
detector = SurfaceContrast(read(config))
rows = []
thresholds = (.025, .05, .10)
cv2.setNumThreads(1)
for run, folder in read(archives).items():
    with CameraArchive(Path(folder)) as archive:
        assert archive.bundle_sha256 == read(features)["reports"][run]["bundle_sha256"]
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            key = run,item["capture_id"]
            if key not in previous:
                continue
            assert not item["ideal_line_shown"] and not item["aid_probe_only"]
            images = {v["view_id"]:cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name),np.uint8),
                                               cv2.IMREAD_COLOR) for v in item["views"]}
            sections = detector.detect(images)
            far = sections
            turns = []
            for a,b,c in zip(far,far[2:],far[4:]):
                if not all(s["valid"] for s in (a,b,c)):
                    continue
                x = [s["forward_x_m"] for s in (a,b,c)]
                edge_slopes = np.array([np.diff([s[side] for s in (a,b,c)])/np.diff(x)
                                        for side in ("right_y_m","left_y_m")])
                changes = np.diff(np.arctan(edge_slopes),axis=1).ravel()
                if changes[0]*changes[1] > 0:
                    turns.append((x[1], float(np.copysign(min(abs(changes)), changes[0]))))
            candidates = {}
            for threshold in thresholds:
                hit = next(((x, turn) for x,turn in turns if abs(turn)>=threshold), None)
                candidates[str(threshold)] = None if hit is None else {
                    "direction":1 if hit[1]>0 else 2,"distance_m":hit[0]} if hit[0]<=20. else None
            label=labels.get(key)
            near=bool(label and label["direction"] and 4<=label["distance"]<=20)
            rows.append(dict(run=run,frame=key[1],near=near,label=label,turns=turns,
                             candidates=candidates,baseline=previous[key]))
assert len(rows)==1195 and sum(r["near"] for r in rows)==152
summary={}
for name in ("baseline",*map(str,thresholds)):
    counts=Counter()
    for r in rows:
        prior=r["baseline"]
        candidate=(prior if prior["alert"] else None) if name=="baseline" else r["candidates"][name]
        if candidate is None:
            continue
        if not r["near"]:
            counts["out_of_band"]+=1
        else:
            d=r["label"]["direction"]
            counts["in_band_alert"]+=1
            if d==candidate["direction"]:
                counts[f"direction_{d}"]+=1
                if abs(r["label"]["distance"]-candidate["distance_m"])<=5.:
                    counts[f"direction_and_distance_{d}"]+=1
    summary[name]=dict(counts)
jsonl(out/"rows.jsonl",rows)
write(out/"result.json",dict(summary=summary,seed=None,termination="archive_end",
    revision=subprocess.check_output(["git","-c","safe.directory=*","rev-parse","HEAD"],text=True).strip(),
    onset_is_sample_station_heuristic=True, stations_m=detector.stations.tolist(), triplet_spacing_m=4., labels_used_for_scoring_only=True,
    hashes={str(p):digest(p) for p in (inventory,forecast,archives,features,config,Path(__file__))},
    rows_sha256=digest(out/"rows.jsonl")))
print(summary)