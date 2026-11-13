"""Extract frozen FlyVis motion from independent, causally timed camera views."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import yaml

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.flyvis_camera import FlyvisCameraProjector, VIEW_ORDER, integration_intervals
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flyvis_fractional_tail import FractionalVisualTail
from flybrain_drive.retinotopic_readout import column_indices
from train_next_corner import BASE, MONZA, MAGIONE
from train_vision_teacher_line import check, digest, read, write


def setup(cache):
    config=read(Path("configs/flight-stream-batched-v1-20261029.json"))
    mount=Path("configs/ac-csp-exterior-front-v3-retina.json")
    model=Path(config["author_model_dir"])
    checkpoint=model/"best_chkpt"
    check(digest(checkpoint)==config["author_weights_sha256"],"author checkpoint differs")
    cache.mkdir(parents=True,exist_ok=True)
    os.environ["FLYVIS_ROOT_DIR"]=str(cache.resolve())
    os.environ["MPLBACKEND"]="Agg"
    install_datamate_cache_compat(cache)
    import flyvis
    torch.set_default_device("cuda")
    cfg=yaml.safe_load((model/"_meta.yaml").read_text(encoding="utf-8"))["config"]["network"]
    network=flyvis.Network(**cfg).to("cuda")
    network.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["network"],strict=True)
    network.eval().requires_grad_(False)
    nodes=network.connectome.nodes
    retinal=network.stimulus.input_index[0]
    u,v=nodes.u[:][retinal],nodes.v[:][retinal]
    indices=column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,
                           [f"{a}{b}" for a in ("T4","T5") for b in "abcd"])
    indices=torch.as_tensor(indices,device="cuda")
    projector=FlyvisCameraProjector(read(mount),u,v)
    uv=np.clip((projector.uv*4).astype(int),0,3)
    cells=uv[:,1]*4+uv[:,0]
    pool=np.eye(16,dtype=np.float32)[cells]
    pool/=np.maximum(pool.sum(axis=0),1)
    pool=torch.as_tensor(pool,device="cuda")
    initial=network.steady_state(1.,.005,batch_size=4,value=.5)
    step=FractionalVisualTail(network,initial).simulate
    return network,projector,indices,pool,step,dict(author_checkpoint_sha256=digest(checkpoint),
            camera_config_sha256=digest(mount))


@torch.no_grad()
def extract(directory,required,net,projector,indices,pool,step):
    features={}
    prior,generation=None,None
    held=torch.full((4,len(projector.uv)),.5,device="cuda")
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
                    elapsed=t-prior
                    while elapsed>1e-12:
                        span=min(elapsed,.08)
                        steps=integration_intervals(span)
                        whole=sum(dt==.005 for dt in steps)
                        if whole:
                            state=step(held[:,None,None,:].expand(-1,whole,-1,-1),.005,state,as_states=True)[-1]
                        if steps and steps[-1]!=.005:
                            state=step(held[:,None,None,:],steps[-1],state,as_states=True)[-1]
                        elapsed-=span
                raw=archive.read(Path(views[name]["file"]).name)
                bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                check(bgr is not None,"image decode")
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                held[j]=torch.as_tensor(projector.sample(rgb),device="cuda")
                prior=t
            if i in required:
                current=(torch.tanh(state.nodes.activity[:,indices])@pool).reshape(-1)
                features[i]=current.detach().cpu().numpy()
        return features,archive.bundle_sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(1)
    check(torch.cuda.is_available(),"GPU runtime required")
    cv2.setNumThreads(1)
    rows=[json.loads(l) for l in (args.dataset/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    folders={n:BASE/n/"demonstration/camera/vision" for n,_ in MONZA}
    folders.update({n:BASE/n/s/"vision" for n,s in MAGIONE})
    net,projector,indices,pool,step,source=setup(args.cache)
    x=np.zeros((len(rows),3*4*8*16),np.float32)
    present=np.zeros(len(rows),bool)
    provenance={}
    start=time.monotonic()
    for name,directory in folders.items():
        subset=[(j,r["frame"]) for j,r in enumerate(rows) if r["run"]==name]
        if not subset:
            continue
        required={i for _,c in subset for i in (c-2,c-1,c)}
        features,bundle=extract(directory,required,net,projector,indices,pool,step)
        for j,i in subset:
            if all(k in features for k in (i-2,i-1,i)):
                x[j]=np.concatenate([features[k] for k in (i-2,i-1,i)])
                present[j]=True
        provenance[name]=dict(bundle_sha256=bundle,requested=len(subset),available=sum(bool(present[j]) for j,_ in subset))
        write(args.dataset/"flyvis-progress.json",dict(sources=provenance,elapsed_s=time.monotonic()-start))
        print(name,provenance[name],flush=True)
    check(np.isfinite(x).all(),"nonfinite FlyVis features")
    np.savez_compressed(args.dataset/"flyvis-features.npz",features=x,available=present)
    write(args.dataset/"flyvis-source.json",dict(**source,rows_sha256=digest(args.dataset/"rows.jsonl"),
        features_sha256=digest(args.dataset/"flyvis-features.npz"),program_sha256=digest(Path(__file__)),
        sources=provenance,elapsed_s=time.monotonic()-start,optimizer_updates=0,termination="completed"))


if __name__=="__main__":
    main()
