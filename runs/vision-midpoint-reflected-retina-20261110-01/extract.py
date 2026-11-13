from pathlib import Path
import json,time
import cv2
import numpy as np
import torch
from extract_next_corner_flyvis import setup, advance
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.retinal_reflection import reflection_permutation
from train_vision_teacher_line import check, digest, read, write

@torch.no_grad()
def extract(directory,required,net,projector,indices,pool,step,read_at):
    permutation=reflection_permutation(projector.uv)
    reflected_view=(0,2,1,3)
    features={}
    prior,generation=None,None
    held=torch.full((4,len(projector.uv)),.5,device="cuda")
    receipt_delays = []
    with CameraArchive(directory) as archive:
        stream=[json.loads(line) for line in archive.read("stream.jsonl").splitlines()]
        for row in stream:
            i=row["capture_id"]
            if i>max(required):
                break
            if row["ideal_line_shown"] or row["aid_probe_only"]:
                continue
            views={v["view_id"]:v for v in row["views"]}
            if set(views)!=set(VIEW_ORDER):
                continue
            events=sorted((float(views[n]["render_started_lower"]),j,n) for j,n in enumerate(VIEW_ORDER))
            if generation!=row["generation"]:
                prior=None
                generation=row["generation"]
                state=net.steady_state(1.,.005,batch_size=4,value=.5)
                held.fill_(.5)
            if prior is not None and events[0][0]<=prior:
                continue
            check(events[-1][0]<=row["received"],"frame clock after receipt")
            for t,j,name in events:
                if prior is not None:
                    state = advance(state, held, t-prior, step)
                raw=archive.read(Path(views[name]["file"]).name)
                bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                check(bgr is not None,"image decode")
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                held[reflected_view[j]]=torch.as_tensor(projector.sample(rgb)[permutation],device="cuda")
                prior=t
            if i in required:
                delay = row["received"] - prior
                check(np.isfinite(delay), "nonfinite receipt delay")
                receipt_delays.append(delay)
                # Branch the readout on delivered frames. Keep the carried state
                # on the render clock, which can precede an earlier receipt.
                snapshot = advance(state, held, delay, step) if read_at == "receipt" else state
                current=(torch.tanh(snapshot.nodes.activity[:,indices])@pool).reshape(-1)
                features[i]=current.detach().cpu().numpy()
        return features,archive.bundle_sha256,receipt_delays



out=Path(__file__).parent
inventory=Path("runs/vision-monza-midpoint-raw-head-20261110-01")
receipt=read(inventory/"result.json")
assert digest(inventory/"rows.jsonl")==receipt["rows_sha256"]
rows=[json.loads(line) for line in (inventory/"rows.jsonl").read_text().splitlines()]
torch.set_num_threads(1)
cv2.setNumThreads(1)
net,projector,indices,pool,step,source=setup(Path("runs/vision-next-corner-flyvis-cache-20261110"))
previous=read(Path("runs/vision-next-corner-20261110-03/flyvis-source.json"))
assert all(previous[k]==v for k,v in source.items())
x=np.zeros((len(rows),1536),dtype=np.float32)
available=np.zeros(len(rows),dtype=bool)
reports={}
start=time.monotonic()
from train_next_corner import BASE,MONZA
run,_=MONZA[0]
subset=[(i,r["history"]) for i,r in enumerate(rows) if r["split"]=="train"]
result,bundle,delays=extract(BASE/run/"demonstration/camera/vision",{f for _,h in subset for f in h},
                             net,projector,indices,pool,step,"last-render")
assert bundle==receipt["provenance"]["camera_bundle_sha256"]
for i,history in subset:
    if all(f in result for f in history):
        x[i]=np.concatenate([result[f] for f in history])
        available[i]=True
reports[run]=dict(bundle_sha256=bundle,requested=len(subset),available=int(available.sum()))
print(reports,flush=True)
assert np.isfinite(x).all() and available.any()
np.savez_compressed(out/"features.npz",features=x,available=available)
write(out/"flyvis-result.json",dict(**source,revision=receipt["revision"],seed=None,
    inventory_sha256=digest(inventory/"result.json"),rows_sha256=receipt["rows_sha256"],
    program_sha256=digest(Path(__file__)),extractor_sha256=digest(Path("scripts/extract_next_corner_flyvis.py")),
    features_sha256=digest(out/"features.npz"),reports=reports,elapsed_s=time.monotonic()-start,
    read_at="last-render",optimizer_updates=0,reflection="input receptor permutation and swapped side eyes; unchanged visual network",termination="completed"))
