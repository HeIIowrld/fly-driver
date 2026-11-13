"""Recompute every body-admitted saved Imola frame for paired frozen profiles."""
from pathlib import Path
import json
import traceback

import cv2
import numpy as np
import torch

from extract_next_corner_flyvis import PairedFlyVisFrames, setup
from flybrain_drive.ac_camera_archive import CameraArchive
from infer_frozen_next_corner import FrozenCornerStream, forecast_values, load_readout
from train_vision_teacher_line import digest, jsonl, read, write

out=Path(__file__).parent
root=out.parent
profiles=dict(
    parent=root/"vision-frozen-next-corner-20261110-01/pins.json",
    background=root/"vision-corner-background-candidate-20261111-02/pins.json",
    blended=root/"vision-corner-blended-candidate-20261111-02/pins.json")
pins=read(profiles["parent"])
inputs=[Path(__file__), *profiles.values(),Path("scripts/infer_frozen_next_corner.py"),
        Path("scripts/extract_next_corner_flyvis.py"),Path("src/flybrain_drive/flyvis_compat.py"),
        Path("src/flybrain_drive/flyvis_fractional_tail.py")]
outputs={name:[] for name in profiles}
features=[]
index=[]
reports={}
mapping={}
try:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    net,projector,indices,pool,step,provenance=setup(root/"vision-next-corner-flyvis-cache-20261110",batch_size=8)
    predictors={name:load_readout(read(path)) for name,path in profiles.items()}
    for suffix in ("01","02","03","05","06","07"):
        source=root/("vision-imola-moving-live-readout-20261110-"+suffix)
        tag=source.name
        trace=source/"live-frozen-readout.jsonl"
        receipt=source/"result.json"
        proof=read(receipt)
        assert digest(trace)==proof["live_frozen_readout"]["rows_sha256"]
        assert digest(profiles["parent"]) in proof["live_input_hashes"].values()
        inputs.extend((trace,receipt,source/"manifest.json"))
        recorded=list(map(json.loads,trace.read_text().splitlines()))
        visual=PairedFlyVisFrames(net,projector,indices,pool,step,batched=True)
        stream=FrozenCornerStream(profiles["parent"],visual,provenance)
        mismatches=[]
        available=0
        with CameraArchive(source/"vision") as archive:
            assert archive.bundle_sha256==proof["camera_archive"]["bundle_sha256"]
            mapping[tag]=dict(directory=str(source/"vision"),bundle_sha256=archive.bundle_sha256)
            captures={r["capture_id"]:r for r in map(json.loads,archive.read("stream.jsonl").splitlines())}
            for row in recorded:
                frame=row["frame"]
                forecasts={name:forecast_values(None,threshold) for name,(_,threshold) in predictors.items()}
                if row["body_valid"]:
                    def image_at(_,view):
                        return cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name),np.uint8),
                                            cv2.IMREAD_COLOR)
                    parent=stream.push(captures[frame],image_at)
                    forecasts["parent"]=parent
                    if parent["available"]:
                        available+=1
                        direct,mirror=(np.concatenate(branch)[None] for branch in zip(*stream.history,strict=True))
                        features.append(np.stack((direct[0],mirror[0])))
                        index.append(dict(run=tag,frame=frame))
                        for name,(predict,threshold) in predictors.items():
                            forecasts[name]=forecast_values(tuple(value[0] for value in predict(direct,mirror)),threshold)
                actual=forecasts["parent"]
                if (actual["available"]!=row["available"] or
                    (row["available"] and (actual["direction"]!=row["direction"] or actual["alert"]!=row["alert"]))):
                    mismatches.append(dict(frame=frame,recorded={k:row.get(k) for k in
                        ("available","direction","alert","presence_score")},replay=actual))
                for name,prediction in forecasts.items():
                    outputs[name].append(dict(run=tag,frame=frame,**prediction))
        reports[tag]=dict(trace_frames=len(recorded),available=available,mismatches=mismatches)
        write(out/"progress.json",dict(completed=reports,current=None))
        print(tag,available,"mismatches",len(mismatches),flush=True)
    np.savez_compressed(out/"features.npz",features=np.array(features,dtype=np.float32))
    jsonl(out/"index.jsonl",index)
    write(out/"archives.json",{tag:r["directory"] for tag,r in mapping.items()})
    hashes={str(p):digest(p) for p in inputs}
    for name,predictions in outputs.items():
        dest=out/name
        dest.mkdir(exist_ok=False)
        jsonl(dest/"rows.jsonl",predictions)
        profile_hashes={path:value for path,value in hashes.items()
                        if path not in {str(p) for p in profiles.values()}}
        profile_hashes[str(profiles[name])]=digest(profiles[name])
        write(dest/"result.json",dict(revision=pins["revision"],seed=None,
            pins_sha256=digest(profiles[name]),inputs_sha256=profile_hashes,rows_sha256=digest(dest/"rows.jsonl"),verified_track_id="imola",
            camera_bundle_sha256_by_run={tag:r["bundle_sha256"] for tag,r in mapping.items()},
            n=len(predictions),available=sum(r["available"] for r in predictions),
            profile=name,termination="archive_end"))
    write(out/"result.json",dict(status="completed_saved_imola_feature_replay",reports=reports,
        hashes=hashes,features_sha256=digest(out/"features.npz"),index_sha256=digest(out/"index.jsonl"),
        profile_results_sha256={name:digest(out/name/"result.json") for name in profiles},
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,seed=None,
        revision=pins["revision"],termination="archive_end"))
except Exception as error:
    write(out/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),completed=reports,
        inputs_sha256={str(p):digest(p) for p in inputs if p.exists()},termination="exception"))
    raise
