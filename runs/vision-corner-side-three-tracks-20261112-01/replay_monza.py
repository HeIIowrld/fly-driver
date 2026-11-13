"""Check the frozen three-track side model on the archived Monza transfer."""
from pathlib import Path
from collections import Counter
import sys

import cv2
import numpy as np
import torch

from train_three import OUT,old,DEV,SideHeat,crops,candidates,infer,sampling,patch,load_observer,observe,frames
ARCHIVE=OUT.parent/"vision-corner-archive-observer-20261112-01"
sys.path.insert(0,str(OUT.parent/"vision-corner-paired-risk-20261112-01"))
from paired_risk import reference

METHODS=("temporal","t4t5_confirm")
MONZA="monza"


def score(rows,truth):
    by_key={(r["run"],r["frame"]):r for r in rows}
    metrics={m:Counter() for m in METHODS}
    paired=Counter()
    for ref in truth:
        item=by_key[ref["run"],ref["frame"]]
        diff=[]
        for method in METHODS:
            forecast=item["forecasts"][method]
            direction={"left":1,"right":2}.get(forecast["state"])
            offer=(direction,forecast["distance_m"]) if direction is not None else None
            category=old.category(ref,offer)
            metrics[method][category]+=1
            if category=="joint_hit":
                metrics[method][f"joint_{ref['label']}"]+=1
            diff.append(category)
        if diff[0]!=diff[1]:
            paired[tuple(diff)]+=1
    return dict(metrics={k:dict(v) for k,v in metrics.items()},
                paired=[dict(edge=k[0],motion=k[1],frames=n) for k,n in sorted(paired.items())])


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    trained=OUT/"training.json"
    frozen=old.read(trained)
    assert MONZA not in frozen["development"] and frozen["development"]==list(DEV)
    weight=OUT/"side-three.pt"
    assert old.digest(weight)==frozen["checkpoint_sha256"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SideHeat().to(device)
    model.load_state_dict(torch.load(weight,map_location=device,weights_only=True))
    model.eval()
    root=ARCHIVE/"monza"
    mapping=ARCHIVE/"monza-transfer-inputs/archive-map.json"
    receipt=old.read(root/"result.json")
    raw_path=root/"sensors.npz"
    assert old.digest(raw_path)==receipt["inputs_sha256"]
    with np.load(raw_path,allow_pickle=False) as data:
        raw={k:data[k] for k in data.files}
    keys=list(zip(raw["run"],raw["frame"],strict=True))
    bundles=receipt["archive_bundles"]
    images,valid=crops(MONZA,mapping,bundles,keys)
    options=list(candidates(MONZA,mapping,bundles,keys,raw["sections"],raw["motion"]))
    assert all(valid[i[0]] for i,_,_ in options)
    heat=infer(model,images,device)
    lut=sampling(old.read(old.MOUNT))
    models,settings,sources=load_observer()
    known,path=reference(MONZA)
    prior=root/"predictions/forecasts.jsonl"
    replay=frames(prior,old.read(prior.with_name("result.json"))["forecasts_sha256"])
    assert len(replay)==len(keys)==3152
    results={"archive":score(replay,known)}
    for name,mode in (("side_contrast","contrast"),("three_track",frozen["choice"])):
        sections=patch(options,raw["sections"],mode,heat,lut)
        sensor=OUT/f"monza-{name}-sensors.npz"
        np.savez_compressed(sensor,**dict(raw,sections=sections))
        with np.load(sensor,allow_pickle=False) as data:
            rows,_,_=observe(data,models,settings)
        assert [(r["run"],r["frame"]) for r in rows]==keys
        trace=OUT/f"monza-{name}-frames.jsonl"
        old.jsonl(trace,rows)
        results[name]=dict(mode=mode,**score(rows,known),frames_sha256=old.digest(trace),sensors_sha256=old.digest(sensor))
    sources.update((Path(__file__),mapping,path,raw_path,trained,weight,prior))
    old.write(OUT/"monza.json",dict(status="archival_transfer_verified",frames=len(replay),labels=len(known),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},methods=results,
              termination="three_track_side_model_monza_replayed"))
    print(results,flush=True)


if __name__=="__main__":
    main()
