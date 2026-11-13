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


def setup(cache, *, batch_size=4, extra_chunks=()):
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
    initial=network.steady_state(1.,.005,batch_size=batch_size,value=.5)
    step=FractionalVisualTail(network,initial,batch_size=batch_size,extra_chunks=extra_chunks).simulate
    return network,projector,indices,pool,step,dict(author_checkpoint_sha256=digest(checkpoint),
            camera_config_sha256=digest(mount))


def advance(state, held, elapsed, step):
    while elapsed > 1e-12:
        span = min(elapsed, .08)
        steps = integration_intervals(span)
        whole = sum(dt == .005 for dt in steps)
        if whole:
            state = step(held[:, None, None, :].expand(-1, whole, -1, -1),
                         .005, state, as_states=True)[-1]
        if steps and steps[-1] != .005:
            state = step(held[:, None, None, :], steps[-1], state, as_states=True)[-1]
        elapsed -= span
    return state


@torch.no_grad()
def extract(directory,required,net,projector,indices,pool,step,read_at,*,reflected=False,profile=None):
    if reflected:
        from flybrain_drive.retinal_reflection import reflection_permutation
        permutation = reflection_permutation(projector.uv)
        reflected_view = (0, 2, 1, 3)
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
            measured = profile is not None and i in required
            started = time.perf_counter() if measured else 0.
            image_s = 0.
            for t,j,name in events:
                if prior is not None:
                    state = advance(state, held, t-prior, step)
                image_start = time.perf_counter() if measured else 0.
                raw=archive.read(Path(views[name]["file"]).name)
                bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                check(bgr is not None,"image decode")
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                sample = projector.sample(rgb)
                if reflected:
                    held[reflected_view[j]] = torch.as_tensor(sample[permutation], device="cuda")
                else:
                    held[j] = torch.as_tensor(sample, device="cuda")
                if measured:
                    image_s += time.perf_counter()-image_start
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
                if measured:
                    profile[i] = dict(received=row["received"], elapsed_s=time.perf_counter()-started,
                                      image_and_input_s=image_s)
        return features,archive.bundle_sha256,receipt_delays


class PairedFlyVisFrames:
    """Carry independent original and reflected states across delivered frame groups."""

    def __init__(self, net, projector, indices, pool, step, *, batched=False):
        from flybrain_drive.retinal_reflection import reflection_permutation

        self.projector, self.indices, self.pool = projector, indices, pool
        self.step, self.batched = step, batched
        self.permutation = reflection_permutation(projector.uv)
        self.held = torch.full((2, 4, len(projector.uv)), .5, device="cuda")
        # The frozen step copies its input activity into graph-owned buffers.
        self.neutral = net.steady_state(1., .005, batch_size=8 if batched else 4, value=.5)
        self.prior = self.generation = self.states = None

    @torch.no_grad()
    def push(self, row, image_at, *, readout=True):
        """Accept a capture row and a callable returning a BGR image by view name."""
        if row["ideal_line_shown"] or row["aid_probe_only"]:
            return None
        views = {view["view_id"]: view for view in row["views"]}
        if set(views) != set(VIEW_ORDER):
            return None
        events = sorted((float(views[name]["render_started_lower"]), i, name)
                        for i, name in enumerate(VIEW_ORDER))
        if self.generation != row["generation"]:
            self.generation = row["generation"]
            self.prior = None
            self.states = (self.neutral,) if self.batched else (self.neutral, self.neutral)
            self.held.fill_(.5)
        if self.prior is not None and events[0][0] <= self.prior:
            return None
        check(events[-1][0] <= row["received"], "frame clock after receipt")
        reflected_view = (0, 2, 1, 3)
        for stamp, index, name in events:
            if self.prior is not None:
                pixels = (self.held.flatten(0, 1),) if self.batched else self.held
                self.states = tuple(advance(state, image, stamp-self.prior, self.step)
                                    for state, image in zip(self.states, pixels, strict=True))
            bgr = image_at(name, views[name])
            check(bgr is not None, "image decode")
            sample = self.projector.sample(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self.held[0, index] = torch.as_tensor(sample, device="cuda")
            self.held[1, reflected_view[index]] = torch.as_tensor(sample[self.permutation], device="cuda")
            self.prior = stamp
        if not readout:
            return None
        activity = [state.nodes.activity for state in self.states]
        if self.batched:
            activity = activity[0].split(4)
        return tuple((torch.tanh(response[:, self.indices])@self.pool).reshape(-1).cpu().numpy()
                     for response in activity)


@torch.no_grad()
def extract_pair(directory, required, net, projector, indices, pool, step, profile=None, *, batched=False, on_frame=None):
    """Decode each archive image once using the delivered-frame state machine."""
    stream = PairedFlyVisFrames(net, projector, indices, pool, step, batched=batched)
    outputs = {}, {}
    last_required = max(required)
    with CameraArchive(directory) as archive:
        for row in map(json.loads, archive.read("stream.jsonl").splitlines()):
            frame = row["capture_id"]
            if frame > last_required:
                break
            started = time.perf_counter()
            images = {} if on_frame is not None else None

            def image_at(name, view):
                data = archive.read(Path(view["file"]).name)
                bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if images is not None:
                    images[name] = bgr
                return bgr

            features = stream.push(row, image_at, readout=frame in required)
            if features is None:
                continue
            for output, feature in zip(outputs, features, strict=True):
                output[frame] = feature
            if on_frame is not None:
                on_frame(row, features, images)
            if profile is not None:
                profile[frame] = time.perf_counter()-started
        return outputs, archive.bundle_sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--read-at", choices=("last-render", "receipt"), default="last-render")
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
        features,bundle,delays=extract(directory,required,net,projector,indices,pool,step,args.read_at)
        for j,i in subset:
            if all(k in features for k in (i-2,i-1,i)):
                x[j]=np.concatenate([features[k] for k in (i-2,i-1,i)])
                present[j]=True
        provenance[name]=dict(bundle_sha256=bundle,requested=len(subset),available=sum(bool(present[j]) for j,_ in subset),
                              receipt_delay_ms_quantiles=(np.quantile(delays,[0,.5,1])*1000).tolist() if delays else [])
        write(args.dataset/"flyvis-progress.json",dict(sources=provenance,elapsed_s=time.monotonic()-start))
        print(name,provenance[name],flush=True)
    check(np.isfinite(x).all(),"nonfinite FlyVis features")
    np.savez_compressed(args.dataset/"flyvis-features.npz",features=x,available=present)
    write(args.dataset/"flyvis-source.json",dict(**source,rows_sha256=digest(args.dataset/"rows.jsonl"),
        features_sha256=digest(args.dataset/"flyvis-features.npz"),program_sha256=digest(Path(__file__)),
        sources=provenance,elapsed_s=time.monotonic()-start,read_at=args.read_at,
        optimizer_updates=0,termination="completed"))


if __name__=="__main__":
    main()
