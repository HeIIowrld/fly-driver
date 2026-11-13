import json
from pathlib import Path

import cv2
import numpy as np
import torch

from extract_next_corner_flyvis import PairedFlyVisFrames, setup
from flybrain_drive.ac_camera_archive import CameraArchive
from infer_frozen_next_corner import FEATURE_DIM, FrozenCornerStream, forecast_values, load_readout
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
base = Path("runs/vision-frozen-next-corner-20261110-01")
pins = base/"pins.json"
predict, threshold = load_readout(read(pins))
inputs = [pins,Path(__file__)]
modes = ("original","reverse_slots","repeat_latest")

def variants(direct, reflected):
    for mode in modes:
        converted = []
        for features in (direct,reflected):
            blocks = features.reshape(-1,3,FEATURE_DIM//3)
            if mode=="reverse_slots":
                blocks = blocks[:,::-1]
            elif mode=="repeat_latest":
                blocks = np.repeat(blocks[:,-1:],3,axis=1)
            converted.append(blocks.reshape(-1,FEATURE_DIM))
        yield mode,predict(*converted)

torch.set_num_threads(1)
cv2.setNumThreads(1)
net,projector,indices,pool,step,provenance = setup(Path("runs/vision-next-corner-flyvis-cache-20261110"),batch_size=8)

imola = []
features = []
mismatches = []
for tag in ("02","03"):
    readout = FrozenCornerStream(pins,PairedFlyVisFrames(net,projector,indices,pool,step,batched=True),provenance)
    run = Path(f"runs/vision-imola-moving-live-readout-20261110-{tag}")
    trace = run/"live-frozen-readout.jsonl"
    receipt = run/"result.json"
    proof = read(receipt)
    assert digest(trace)==proof["live_frozen_readout"]["rows_sha256"]
    assert digest(pins) in proof["live_input_hashes"].values()
    inputs.extend((trace,receipt))
    with CameraArchive(run/"vision") as archive:
        assert archive.bundle_sha256 == proof["camera_archive"]["bundle_sha256"]
        frames = {r["capture_id"]:r for r in map(json.loads,archive.read("stream.jsonl").splitlines())}
        for row in map(json.loads,trace.read_text(encoding="utf8").splitlines()):
            if not row["body_valid"]:
                continue
            frame = frames[row["frame"]]
            def pixels(_, view):
                return cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name),np.uint8),cv2.IMREAD_COLOR)
            forecast = readout.push(frame,pixels)
            if not row["available"] or row["speed_mps"] < 1/3.6:
                continue
            if not (forecast["available"] and forecast["alert"] == row["alert"]
                    and forecast["direction"] == row["direction"]):
                mismatches.append(dict(run=tag,frame=row["frame"],archived=row["direction"],
                                       replay=forecast))
                continue
            direct,mirror = (np.concatenate(branch)[None] for branch in zip(*readout.history,strict=True))
            result = {mode:forecast_values(tuple(x[0] for x in response),threshold)
                      for mode,response in variants(direct,mirror)}
            features.append(np.stack((direct[0],mirror[0])))
            imola.append(dict(run=tag,frame=row["frame"],forecasts=result))

parity = base/"archive-parity"
paths = [parity/"index.jsonl",parity/"original/features.npz",parity/"reflected/features.npz"]
source = base/"archive-inference/result.json"
source_proof = read(source)
assert all(digest(p) in source_proof["inputs_sha256"].values() for p in paths)
labels = Path("runs/vision-magione-edge-midpoint-inventory-20261110-01/rows.jsonl")
label_proof = labels.with_name("result.json")
assert digest(labels)==read(label_proof)["rows_sha256"]
inputs.extend((*paths,source,labels,label_proof))
truth = {(r["run"],r["frame"]):r for r in map(json.loads,labels.read_text().splitlines())}
records = [truth[r["run"],r["frame"]] for r in map(json.loads,paths[0].read_text().splitlines())]
with np.load(paths[1]) as a,np.load(paths[2]) as b:
    assert a["available"].all() and b["available"].all()
    magione = {}
    for mode,response in variants(a["features"],b["features"]):
        f = [forecast_values(x,threshold) for x in zip(*response,strict=True)]
        near = [r["direction"] in (1,2) and r["distance"] is not None and 4<=r["distance"]<=20 for r in records]
        magione[mode] = dict(near=sum(near),correct=sum(n and p["alert"] and p["direction"]==r["direction"]
            for n,p,r in zip(near,f,records,strict=True)),outside=sum(not n and p["alert"] for n,p in zip(near,f,strict=True)))
assert magione["original"] == dict(near=152,correct=66,outside=26)
assert len(imola)+len(mismatches)==64
report={mode:sum(r["forecasts"][mode]["alert"] for r in imola) for mode in modes}
jsonl(out/"imola-rows.jsonl",imola)
np.savez_compressed(out/"imola-features.npz",features=np.array(features))
write(out/"result.json",dict(status="completed_readout_time_slot_contrast",imola_alerts=report,magione=magione,imola_reproduced=len(imola),mismatches=mismatches,
    seed=None,termination="archive_end", hashes={str(p):digest(p) for p in inputs},
    imola_rows_sha256=digest(out/"imola-rows.jsonl"), feature_sha256=digest(out/"imola-features.npz"),actor_deployed=False))
print(report,magione)
