from pathlib import Path
import json
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from audit_road_edge_projection import road_edges
from train_next_corner import BASE,MAGIONE,assign,CURVATURE_THRESHOLD
import train_vision_teacher_line as teacher
from train_vision_teacher_line import CameraArchive,at,digest,jsonl,read,write

out=Path(__file__).parent
source=Path("runs/vision-next-corner-20261110-03")
receipt=read(source/"result.json")
assert digest(source/"rows.jsonl")==receipt["rows_sha256"]
dataset=read(source/"dataset.json")
teacher.CORNER=CURVATURE_THRESHOLD
code=(out/"probe.py").read_text(encoding="utf-8-sig")
exec(compile(code[code.index("def periodic_line("):code.index("data=np.genfromtxt")],
             str(out/"probe.py"),"exec"))
p=BASE/"heldout-magione-geometry-01/geometry.json"
assert digest(p)==dataset["magione_geometry_sha256"]
g=read(p)["rows"]
ref=np.array([r["native_xyz"] for r in g])[:,[2,0]]
mid=np.mean(road_edges(g),axis=0)[:,[2,0]]
length,spans,station,tree=periodic_line(ref)
ml,ms,mg,mt=periodic_line(mid)
assert [list(s) for s in spans]==dataset["magione_map_corner_spans_m"]
labels={(r["run"],r["frame"]):r for r in
        (json.loads(line) for line in (source/"rows.jsonl").read_text().splitlines())
        if r["split"]=="external"}
rows=[]
for run,sub in MAGIONE:
    with CameraArchive(BASE/run/sub/"vision") as archive:
        assert archive.bundle_sha256==dataset["sources"][run]["provenance"]["bundle_sha256"]
        for row in (json.loads(l) for l in archive.read("stream.jsonl").splitlines()):
            key=(run,row["capture_id"])
            if key not in labels:
                continue
            label=labels[key]
            anchor=max(row["views"],key=lambda v:v["render_started_lower"])
            xy=np.asarray(anchor["source_render_metadata"]["source_body_after_render"]["renderer_only_position"])[[2,0]]
            _,i=tree.query(xy)
            residual,j=mt.query(xy)
            a=assign(float(station[i]),spans,length,True)
            b=assign(float(mg[j]),ms,ml,True) if residual<=7 else None
            assert a[:2]==(label["event"],label["direction"])
            assert not a[1] or a[2]==label["distance"]
            rows.append(dict(run=run,frame=key[1],reference=a,edge_midpoint=b,
                             edge_midpoint_residual_m=float(residual)))
    print(run,flush=True)
assert len(rows)==len(labels)
result=compare([r["reference"] for r in rows],[r["edge_midpoint"] for r in rows])
jsonl(out/"capture-labels.jsonl",rows)
write(out/"capture-comparison.json",dict(**result,n=len(rows),program_sha256=digest(Path(__file__)),
    rows_sha256=digest(out/"capture-labels.jsonl"),source_sha256=digest(out/"result.json"),
    original_label_sha256=receipt["rows_sha256"],scope="teacher/diagnostics only"))
print(result,flush=True)
