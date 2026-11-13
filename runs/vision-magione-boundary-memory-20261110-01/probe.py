from pathlib import Path
import base64
import hashlib
import json
from collections import Counter
import numpy as np

from audit_road_edge_projection import road_edges,section_points
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.local_racing_line import plan_local_line
from flybrain_drive.road_boundary_memory import RoadBoundaryMemory
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,decode_navigation_body_prefix
from train_next_corner import BASE,MAGIONE
from train_vision_teacher_line import CameraArchive,digest,jsonl,read,write

out=Path("runs/vision-magione-boundary-memory-20261110-01")
source=Path("runs/vision-magione-pixel-boundary-20261110-02")
pixel=read(source/"result.json")
map_file=BASE/"heldout-magione-geometry-01/geometry.json"
assert digest(map_file)==read(map_file.parent/"result.json")["geometry_sha256"]
edges=road_edges(read(map_file)["rows"])
limits=dict(clearance_m=1.1,max_curvature=.15,lateral_accel_mps2=4.,braking_mps2=4.,speed_ceiling_mps=25.)

def get_native(root,name):
    contract=Path("configs/ac-virtual-navigation-body-v2.json")
    receipt=read(root/"result.json")
    manifest=read(root/"manifest.json")["config"]
    assert manifest["identity"]["track"]=="magione"
    identity={k:manifest["identity"][k] for k in ("car_model","ac_version","shared_memory_version")}
    encoder=NavigationBodyEncoder(contract,native_identity=identity)
    path=root/"owner-native-copy.json"
    assert digest(path)==receipt["native_sha256"]
    seen={}
    values=[]
    for q in read(path):
        a=q["authority"]
        raw=base64.b64decode(a["prefix_base64"],validate=True)
        assert a["same_native_copy"] and hashlib.sha256(raw).hexdigest()==a["prefix_sha256"]
        prefix=raw[:PHYSICS_PREFIX.size]
        assert hashlib.sha256(prefix).hexdigest()==a["body_prefix_sha256"]
        pid=a["packet_id"]
        if pid in seen:
            assert prefix==seen[pid]
            continue
        seen[pid]=prefix
        t=a["received_monotonic_seconds"]
        obs=encoder.encode(decode_navigation_body_prefix(prefix,t),episode_id=name,decision_monotonic_seconds=t)
        assert obs.packet_id==pid and obs.valid[[4,16]].all()
        values.append((t,float(obs.values[16]),float(obs.values[4])))
    samples=np.array(values)
    assert np.all(np.diff(samples[:,0])>0)
    return samples,path

def advance_to(memory,samples,clock,end,index):
    resets=0
    while clock<end-1e-10:
        next_clock=min(end,samples[index+1,0] if index+1<len(samples) else end)
        age=next_clock-samples[index,0]
        if age>.05:
            memory.reset()
            resets+=1
        else:
            memory.advance(samples[index,1],samples[index,2],next_clock-clock)
        clock=next_clock
        if index+1<len(samples) and clock==samples[index+1,0]:
            index+=1
    return clock,index,resets

counts={}
for name,sub in MAGIONE:
    root=BASE/name/sub
    native,native_path=get_native(root,name)
    frame_file=source/f"{name}.jsonl"
    assert digest(frame_file)==pixel["sources"][name]["output_sha256"]
    rows={a["frame"]:a for a in map(json.loads,frame_file.read_text().splitlines())}
    memory=RoadBoundaryMemory(.60)
    clock=None
    records=[]
    status=Counter()
    with CameraArchive(root/"vision") as archive:
        assert archive.bundle_sha256==pixel["sources"][name]["bundle_sha256"]
        stream=[json.loads(s) for s in archive.read("stream.jsonl").splitlines()]
        for q in stream:
            frame=q["capture_id"]
            if frame not in rows:
                continue
            record=rows[frame]
            t=q["received"]
            if clock is None:
                index=int(np.searchsorted(native[:,0],t,side="right")-1)
                assert index>=0
                clock=t
            clock,index,n=advance_to(memory,native,clock,t,index)
            status["native_gap_resets"]+=n
            fresh=t-native[index,0]<=.05
            if not fresh:
                memory.reset()
                status["stale_native_frame"]+=1
            raw=record["sections"]
            filled=memory.update(raw) if fresh else [dict(s,source="observed" if s["valid"] else "missing",age_s=None) for s in raw]
            assert all((a["left_y_m"],a["right_y_m"])==(b["left_y_m"],b["right_y_m"])
                       for a,b in zip(raw,filled,strict=True) if a["valid"])
            path,path_status=plan_local_line(filled,0.,**limits)
            if record["path_generated"]:
                assert path is not None
            errors=[]
            if any(s["source"]=="carried" for s in filled):
                view=max(q["views"],key=lambda v:v["render_started_lower"])
                body=view["source_render_metadata"]["source_body_after_render"]
                truth=section_points(edges,body)
                origin=np.array(body["renderer_only_position"])
                look=np.array(body["renderer_only_look"])[[2,0]]
                look/=np.linalg.norm(look)
                y=(truth-origin)[:,:,[2,0]]@np.array([-look[1],look[0]])
                for j,s in enumerate(filled):
                    if s["source"]=="carried" and np.isfinite(y[j]).all():
                        errors.append(dict(distance_m=s["forward_x_m"],age_s=s["age_s"],
                                           abs_error_m=np.abs(y[j,::-1]-[s["right_y_m"],s["left_y_m"]]).tolist()))
            records.append(dict(frame=frame,t=t,body_age_s=t-native[index,0],sections=filled,
                           prior_path=record["path_generated"],memory_path=path is not None,status=path_status,
                           carried_map_proxy_errors=errors))
    dest=out/f"{name}.jsonl"
    jsonl(dest,records)
    assert len(records)==len(rows)
    accepted=[a for a in records if a["memory_path"]]
    added=[a for a in accepted if not a["prior_path"]]
    all_errors=[x for a in records for x in a["carried_map_proxy_errors"]]
    counts[name]=dict(input_sha256={str(p):digest(p) for p in (native_path,frame_file)},
       frames=len(records),path_without_memory=sum(a["prior_path"] for a in records),
       path_with_memory=len(accepted),added_frames=len(added),native_status=dict(status),
       carried_sections=sum(s["source"]=="carried" for a in records for s in a["sections"]),
       carried_map_proxy_mae_m=float(np.mean([x["abs_error_m"] for x in all_errors])) if all_errors else None,
       output_sha256=digest(dest))
    print(name,counts[name],flush=True)
write(out/"result.json",dict(revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",seed=None,
    input_sha256={str(p):digest(p) for p in (Path(__file__),map_file,source/"result.json",
      Path("src/flybrain_drive/road_boundary_memory.py"),Path("src/flybrain_drive/local_racing_line.py"))},
    counts=counts,memory_age_limit_s=.60,actuator_messages=0,
    termination="stored Magione four-episode geometry and native motion replay completed"))
