"""Diagnose Silverstone corner misses by substituting physical edges offline."""
from collections import Counter
from pathlib import Path

import numpy as np

from train_three import OUT,RAW_ARCHIVE,ROUTING,MESH,TEST,old,frames,load_observer,observe
from replay_corner_boundary_states import sensor_rows,proposals,alt

METHODS=("temporal","t4t5_confirm")


def forecast(item):
    side={"left":1,"right":2}.get(item["state"])
    return (side,item["distance_m"]) if side is not None else None


def score(rows,truth,raw,aligned,settings):
    metadata=sensor_rows(raw,settings)
    raw_pool=proposals(metadata,raw["sections"],raw["motion"],raw["heads"])[0]
    added_pool=proposals(metadata,aligned,raw["motion"],raw["heads"])[0]
    result=[]
    for i,(decision,ref) in enumerate(zip(rows,truth,strict=True)):
        assert (decision["run"],decision["frame"])==(ref["run"],ref["frame"])
        if ref["truth"]!="approach" or ref["label"]!="2":
            continue
        available=[c["offer"] for c in raw_pool[i]+added_pool[i]]
        joint=any(old.category(ref,offer)=="joint_hit" for offer in available)
        chosen=old.category(ref,decision["candidate"])
        result.append(dict(frame=ref["frame"],truth_m=ref["truth_endpoints_m"],candidate=decision["candidate"],
                           available_joint=joint,selection=chosen,
                           confirmed={m:old.category(ref,forecast(decision["forecasts"][m])) for m in METHODS}))
    summary=dict(frames=len(result),pool_joint=int(sum(r["available_joint"] for r in result)),
                 selected=dict(Counter(r["selection"] for r in result)),
                 confirmed={m:dict(Counter(r["confirmed"][m] for r in result)) for m in METHODS})
    return dict(summary=summary,details=result)


def inject(observed,edge,eligible,mode):
    output=observed.copy()
    use=eligible.copy()
    if mode in ("fill_right","replace_right"):
        use[:,:,1]=False
    if mode=="fill_right":
        use &= ~np.isfinite(observed)
    output[use]=edge[use]
    width=output[:,:,1]-output[:,:,0]
    invalid=np.isfinite(width)&((width<=3.)|(width>=20.))
    output[invalid]=observed[invalid]
    return output,int((use&~invalid[:,:,None]).sum())


def main():
    source=RAW_ARCHIVE/"silverstone"
    archived=old.read(source/"result.json")
    bundle=source/"sensors.npz"
    assert old.digest(bundle)==archived["inputs_sha256"]
    assert old.digest(MESH)==old.read(MESH.with_suffix(".json"))["labels_sha256"]
    with np.load(bundle,allow_pickle=False) as data:
        base={k:data[k] for k in data.files}
    keys=list(zip(base["run"],base["frame"],strict=True))
    with np.load(MESH,allow_pickle=False) as data:
        assert keys==list(zip(data["run"],data["frame"],strict=True))
        assert np.array_equal(data["stations"],base["stations"])
        edge=data["lateral_m"]
        eligible=data["adjacent_invalid"]&np.isfinite(data["image_px"]).all(-1).any(axis=1)
    prior=ROUTING/f"{TEST}-state-distance.jsonl"
    reference=frames(prior,old.read(ROUTING/"straight-verification.json")["tracks"][TEST]["rows_sha256"])
    models,settings,sources=load_observer()
    result={}
    # Use the full chronological stream for each diagnostic so neighbouring frames follow the same rule.
    for mode in ("original","fill_right","replace_right","replace_both"):
        sensor=OUT/f"oracle-{mode}-sensors.npz"
        variant,n=(base["sections"],0) if mode=="original" else inject(base["sections"],edge,eligible,mode)
        np.savez_compressed(sensor,**dict(base,sections=variant))
        with np.load(sensor,allow_pickle=False) as data:
            forecasts,aligned,_=observe(data,models,settings)
            result[mode]=dict(physical_substitutions=n,sensor_sha256=old.digest(sensor),**score(forecasts,reference,data,aligned,settings))
    assert all(m["summary"]["frames"]==13 for m in result.values())
    assert result["original"]["summary"]["confirmed"]["t4t5_confirm"]["joint_hit"]==2
    sources.update((Path(__file__),bundle,MESH,MESH.with_suffix(".json"),prior,source/"result.json"))
    old.write(OUT/"geometry-intervention.json",dict(status="diagnostic_only",tracks=result,
             source_sha256={str(p):old.digest(p) for p in sorted(sources)},
             termination="oracle_interventions_without_weight_or_cut_changes"))
    print({m:r["summary"] for m,r in result.items()},flush=True)


if __name__=="__main__":
    main()
