"""Evaluate sealed original, phase and compact heads on a fresh qualified teacher lap."""
import argparse
import base64
import gc
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.compact_flight_pedal_readout import CompactFlightPedalReadout
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector, VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.launch_pedal_candidate import load_control_actor_runtime
from flybrain_drive.motion_spatial_runtime import MotionSpatialObservation
from flybrain_drive.native_multisensory_extension import decode_native_extension
from flybrain_drive.sac_unified_student import assemble_student,read
from flybrain_drive.episode_storage import load_array
from flybrain_drive.spatial_pool_residual import discarded_retinal_component
from scripts.audit_flight_braking_phase_errors import evaluate
from scripts.train_qualified_sac_unified import load_episode,context

SPATIAL=Path("runs/thermal-control-improvement-20261104/newhot-spatial-01")


def inputs(base, out):
    actor=base/"actor-data-01"
    d,c,prov=load_episode(actor)
    src=Path(read(actor/"manifest.json")["config"]["arguments"]["recording"])
    qr=read(base/"qualification-02/result.json")
    motion=base/"motion-data-01"
    mr=read(motion/"result.json")
    mcfg=read(motion/"manifest.json")["config"]
    if (mr["status"]!="completed_qualified_motion_spatial_recovery"
            or sha256(motion/"motion-spatial.npz")!=mr["features_sha256"]
            or mcfg["actor_arrays_sha256"]!=prov["arrays_sha256"]):
        raise ValueError("Exact fresh actor motion recovery required")
    with np.load(motion/"motion-spatial.npz") as archive:
        fine=archive["discarded_motion_grid8"].copy()
        coarse=archive["original_motion_pool"].copy()
        ids=archive["camera_indices"].copy()
        mapping=archive["event_to_camera"].copy()
    np.testing.assert_array_equal(ids[mapping],d["camera_index"])
    path=SPATIAL/"pool-geometry.npz"
    if sha256(path)!=read(SPATIAL/"result.json")["geometry_sha256"]:
        raise ValueError("Original frozen geometry differs")
    with np.load(path) as z:g={k:z[k].copy() for k in ("old","inverse","grid8_narrow","uv")}
    model,_=assemble_student()
    with load_array(model.motor.backbone.spec["scene_registration_file"],allow_pickle=False) as z:
        projector=FlyvisCameraProjector(read("configs/ac-csp-exterior-front-v3-retina.json"),z["u"],z["v"])
    del model
    np.testing.assert_allclose(projector.uv,g["uv"],rtol=0,atol=1e-8)
    scene=d["scene"].reshape(-1,4,9,16)[:,:,0]
    first={int(v):int(np.flatnonzero(d["camera_index"]==v)[0]) for v in ids}
    static=[]
    with CameraArchive(src/"camera/vision") as archive:
        if archive.bundle_sha256!=qr["camera_bundle_sha256"]:
            raise ValueError("Original qualified camera bundle differs")
        stream=[json.loads(line) for line in archive.read("stream.jsonl").decode("utf8").splitlines()]
        for camera_index in ids:
            item=stream[int(camera_index)]
            frame=archive.json(Path(item["file"]).name)
            views={v["view_id"]:v for v in frame["views"]}
            retina=np.asarray([np.tanh(projector.sample(np.asarray(Image.open(io.BytesIO(
                archive.read(views[name]["file"]))).convert("RGB")).copy())) for name in VIEW_ORDER])
            expected=scene[first[int(camera_index)]]
            np.testing.assert_allclose(retina@g["old"].T,expected,atol=2e-6,rtol=0)
            residual=discarded_retinal_component(retina,g["old"],g["inverse"])
            static.append(np.concatenate([expected.reshape(-1),(residual@g["grid8_narrow"].T).reshape(-1)]))
    static=np.asarray(static,dtype=np.float32)
    native=src/"camera/native-body.jsonl"
    if sha256(native)!=qr["native_body_sha256"]:
        raise ValueError("Qualified original native hash differs")
    samples={}
    needed=set(map(int,d["body_packet"]))
    for line in native.open(encoding="utf-8"):
        a=json.loads(line)["authority"];pid=a["packet_id"]
        if pid in needed and pid not in samples:
            raw=base64.b64decode(a["prefix_base64"],validate=True)
            if hashlib.sha256(raw).hexdigest()!=a["prefix_sha256"]:
                raise ValueError("Original native prefix hash differs")
            samples[pid]=decode_native_extension(raw[:PHYSICS_PREFIX.size],a["received_monotonic_seconds"])
    if set(samples)!=needed:raise ValueError("Fresh native coverage incomplete")
    event_path=src/"teacher/events.jsonl"
    if sha256(event_path)!=read(src/"teacher/result.json")["events_sha256"]:
        raise ValueError("Genuine teacher event history differs")
    requests=[x for l in event_path.open(encoding="utf-8") if (x:=json.loads(l))["kind"]=="teacher_request"]
    genuine=np.zeros(len(d["labels"]),dtype=bool)
    for j in np.flatnonzero(d["label_valid"]):
        e=requests[int(d["teacher_request_index"][j])]
        u=e["request"]["requested_control"]
        np.testing.assert_allclose(d["labels"][j],[u["steering"]/.6,u["throttle"],u["brake"]],rtol=0,atol=1e-8)
        genuine[j]=e["unlimited"]==e["limited"] and not e["automatic_shift_coast"]
    item=dict(data=d,receipt=dict(role="fresh_development_same_Monza"),
              high=genuine&d["fit_mask"],
              low=genuine&d["label_valid"]&~d["fit_mask"]&d["body_valid"][:,16]&(d["body"][:,16]<13))
    write_json(out/"input-receipt.json",dict(actor=prov,motion_sha256=mr["features_sha256"],
        camera_sha256=qr["camera_bundle_sha256"],native_sha256=qr["native_body_sha256"],
        teacher_events_sha256=sha256(event_path),fit_allowed=False))
    return item,c,static,fine,coarse,mapping,samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fresh",required=True)
    p.add_argument("--compact",required=True,action="append")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    base=Path(a.fresh)
    role=base/"preregistered-role.json"
    out=begin_run(a.output,dict(args=vars(a),program_sha256=sha256(__file__),seed=None,
        role_receipt_sha256=sha256(role),fit_allowed=False,vehicle_control_sent=False))
    result=dict(status="failed",fit_allowed=False,vehicle_control_sent=False,models={})
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    gc.disable()
    try:
        r=read(role)
        if r["role"]!="evaluation_only_new_epoch_same_Monza" or r["fit_allowed"] is not False:
            raise ValueError("Prospectively assigned independent evaluation role required")
        item,c,static,fine,coarse,mapping,samples=inputs(base,out)
        d=item["data"]; arrays={}
        for path in map(Path,a.compact):
            cfg=read(path/"manifest.json")["config"]; digest=read(path/"result.json")
            if digest["status"]!="completed_compact_braking_fit" or sha256(path/"pedals.pt")!=digest["checkpoint_sha256"]:
                raise ValueError("Sealed completed compact fit required")
            cp=torch.load(path/"pedals.pt",map_location="cpu",weights_only=True)
            phase=Path(cfg["arguments"]["phase_fit"])
            lineage=cp["actor_kind"]
            runtime,_=load_control_actor_runtime("balanced_"+lineage,phase/"pedals.pt",cp["phase_checkpoint_sha256"])
            try:
                compact=CompactFlightPedalReadout(runtime.head,cp["centers"],cp["radii"],cp["coefficients"]).eval()
                runtime.begin_episode(c["episode_id"])
                runtime.monotonic=lambda:0. # Offline compute only; source clocks remain original.
                muscles=[]; phase_output=[]
                for i,frame in enumerate(mapping):
                    m=MotionSpatialObservation(c["episode_id"],float(d["camera_receipt"][i]),
                         tuple(map(float,d["render_lowers"][i])),coarse[frame],fine[frame])
                    v=runtime.decide(decision_time=float(d["decision_time"][i]),scene_features=d["scene"][i],
                        static_features=static[frame],body_context=context(d,c,i),
                        native_sample=samples[int(d["body_packet"][i])],motion_observation=m)
                    muscles.append(v["muscles"]);phase_output.append(v["prediction"])
                    if i%1000==0:write_json(out/"progress.json",dict(lineage=lineage,decisions=i,total=len(mapping)))
                phase_output=np.asarray(phase_output)
                with torch.no_grad():
                    mu=torch.tensor(np.asarray(muscles))
                    original=runtime.head.parent(mu).numpy()
                    phase_batch=runtime.head(mu).numpy()
                    alternative=compact(mu).numpy()
                if (not np.array_equal(phase_batch[:,0],alternative[:,0])
                        or np.max(abs(phase_batch-phase_output))>2e-6):
                    raise ValueError("Compact pedal fit changed steering")
                row=evaluate(item,alternative[:,1:],original[:,1:])
                phase_row=evaluate(item,phase_output[:,1:],original[:,1:])
                result["models"][lineage]=dict(compact_sha256=digest["checkpoint_sha256"],
                    phase_sha256=cp["phase_checkpoint_sha256"],phase=phase_row,compact=row,decisions=len(mapping))
                arrays.update({lineage+"_original":original,lineage+"_phase":phase_output,
                              lineage+"_compact":alternative,lineage+"_muscles":mu.numpy()})
            finally:runtime.close()
        np.savez_compressed(out/"predictions.npz",**arrays)
        result.update(status="completed_fresh_recorded_evaluation",predictions_sha256=sha256(out/"predictions.npz"))
    except Exception as exc:
        result["reason"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
