"""Join out-of-fold near-corner false alarms to original Monza captures and geometry."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from flybrain_drive.ac_camera_archive import CameraArchive
from train_next_corner import BASE, MONZA, CURVATURE_THRESHOLD
import train_vision_teacher_line as teacher
from train_vision_teacher_line import at, digest, jsonl, read, write

out=Path(__file__).parent
root=out.parent
scores=root/"vision-monza-presence-temporal-ablation-20261111-01/rows.jsonl"
inventory=root/"vision-monza-two-lap-presence-20261111-01/rows.jsonl"
track=Path("assets/external/ac-teacher-geometry-20261027/monza.csv")
ref=root/"vision-corner-label-geometry-20261110-01/probe.py"
source_proof=read(scores.with_name("result.json"))
assert digest(scores)==source_proof["rows_sha256"]
assert digest(inventory) in source_proof["hashes"].values()
assert digest(track)==read(root/"vision-next-corner-20261110-03/dataset.json")["monza_map_sha256"]
teacher.CORNER=CURVATURE_THRESHOLD
text=ref.read_text(encoding="utf-8-sig")
exec(compile(text[text.index("def periodic_line("):text.index("def near(")],str(ref),"exec"))
raw=np.genfromtxt(track,names=True,delimiter=",")
mid=np.column_stack([(raw[f"left_border_{axis}"]+raw[f"right_border_{axis}"])/2 for axis in ("x","y")])
length,spans,stations,tree=periodic_line(mid)
labels={(r["run"],r["frame"]):r for r in map(json.loads,inventory.read_text().splitlines())}
bad={(r["run"],r["frame"]):r for r in map(json.loads,scores.read_text().splitlines())
     if r["mode"]=="latest_and_changes" and r["cohort"]=="monza_train_oof" and not r["near"] and r["alert"]}
assert len(bad)==22
rows=[]
receipts=[]
images={}
for run,qual in MONZA:
    receipt=BASE/run/qual/"result.json"
    receipts.append(receipt)
    with CameraArchive(BASE/run/"demonstration/camera/vision") as archive:
        assert archive.bundle_sha256==read(receipt)["camera_bundle_sha256"]
        for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
            key=run,item["capture_id"]
            if key not in bad:
                continue
            meta=archive.json(Path(item["file"]).name)
            assert meta["index"]==key[1] and meta["track_id"]=="monza"
            pose=max(meta["views"],key=lambda v:v["render_call_finished_clock"])["source_body_after_render"]
            origin=np.asarray(pose["renderer_only_position"])[[2,0]]
            forward=np.asarray(pose["renderer_only_look"])[[2,0]]
            forward/=np.linalg.norm(forward)
            error,i=tree.query(origin)
            tangent=tree.data[(i+1)%len(stations)]-tree.data[i-1]
            assert error<=7. and tangent@forward>=.7*np.linalg.norm(tangent)
            station=stations[i]
            active=next((n for n,(a,b,_) in enumerate(spans) if (station-a)%length<b-a),None)
            next_corner=min(range(len(spans)),key=lambda n:(spans[n][0]-station)%length)
            gap=float((spans[next_corner][0]-station)%length)
            kind="inside_corner" if active is not None else "onset_4_to_20m" if 4<=gap<=20 else (
                 "onset_under_4m" if gap<4 else "onset_20_to_30m" if gap<=30 else
                 "onset_30_to_60m" if gap<=60 else "onset_beyond_60m")
            r=dict(bad[key],group=labels[key]["physical_group"],station_m=float(station),
                   active_event=active,next_event=next_corner,next_distance_m=gap,kind=kind,
                   inventory_label_distance_m=labels[key]["distance"],projection_error_m=float(error))
            rows.append(r)
            front=next(v for v in item["views"] if v["view_id"]=="front")
            images[key]=cv2.imdecode(np.frombuffer(archive.read(Path(front["file"]).name),np.uint8),cv2.IMREAD_COLOR)
assert len(rows)==22
# Display representative raw frames, selected deterministically by earliest capture per class and run.
sample={}
for r in sorted(rows,key=lambda r:(r["run"],r["frame"])):
    sample.setdefault((r["run"],r["kind"]),r)
thumbs=[]
for r in sample.values():
    im=images[r["run"],r["frame"]]
    tile=cv2.copyMakeBorder(cv2.resize(im,(320,180)),0,40,0,0,cv2.BORDER_CONSTANT)
    label=f"{r['frame']}  {r['kind']}  {r['next_distance_m']:.0f}m"
    cv2.putText(tile,label,(5,203),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    thumbs.append(tile)
if len(thumbs)%2:
    thumbs.append(np.zeros_like(thumbs[0]))
sheet=out/"captures.png"
assert cv2.imencode(".png",np.vstack([np.hstack(thumbs[i:i+2]) for i in range(0,len(thumbs),2)]))[1].tofile(sheet) is None
jsonl(out/"rows.jsonl",rows)
summary=dict(kinds=dict(Counter(r["kind"] for r in rows)),
             next_event=dict(Counter(r["next_event"] for r in rows)),
             by_group={str(g):dict(Counter(r["kind"] for r in rows if r["group"]==g))
                       for g in sorted({r["group"] for r in rows})},
             distance_range_m=[min(r["next_distance_m"] for r in rows),max(r["next_distance_m"] for r in rows)])
write(out/"result.json",dict(summary=summary,seed=None,termination="diagnostic_subset",
    revision=source_proof["revision"],reference_only=True,
    hashes={str(p):digest(p) for p in (scores,inventory,track,ref,Path(__file__),*receipts)},
    rows_sha256=digest(out/"rows.jsonl"),sheet_sha256=digest(sheet)))
print(summary)
