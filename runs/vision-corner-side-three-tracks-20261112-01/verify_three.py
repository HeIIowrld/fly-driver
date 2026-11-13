"""Verify the three-circuit split, side-edge labels, and same-archive comparisons."""
from pathlib import Path
from collections import Counter

import numpy as np

from train_three import OUT,DENSE,EXTRA,DEV,TEST,VAL,RAW_ARCHIVE,ROUTING,old,frames
from verify_side import classify

METHODS=("temporal","t4t5_confirm")


def label_audit(track):
    path=EXTRA/f"{track}-train-labels.npz"
    meta=old.read(path.with_suffix(".json"))
    assert meta["track"]==track and meta["status"]=="offline_only"
    assert old.digest(path)==meta["labels_sha256"]
    for filename,sha in meta["source_sha256"].items():
        assert old.digest(Path(filename))==sha
    with np.load(path,allow_pickle=False) as data:
        keys=list(zip(data["run"],data["frame"],strict=True))
        assert len(keys)==len(set(keys))==meta["frames"]
        np.testing.assert_array_equal(data["stations"],old.STATIONS)
        witness=data["adjacent_invalid"]
        assert np.isfinite(data["lateral_m"][witness]).all()
        p=data["image_px"]
        supported=witness[:,None]&np.isfinite(p).all(axis=-1)
        assert (p[supported]>=0.).all() and (p[supported]<=[319.,179.]).all()
        counts=dict(zip(data["views"].tolist(),supported.sum(axis=(0,2)).tolist(),strict=True))
        assert counts==meta["counts_by_view"]
    return dict(frames=len(keys),right=counts["right"][0],left=counts["left"][1],
                withheld=meta["rejected_surface_witnesses"],labels_sha256=old.digest(path))


def outcomes(actual,reference):
    output=Counter()
    for r,y in zip(actual,reference,strict=True):
        assert (r["run"],r["frame"])==(y["run"],y["frame"])
        a,b=(classify(r,y,mode) for mode in METHODS)
        if a!=b:
            output[y["truth"],y["label"],a,b]+=1
    return [dict(truth=k[0],label=k[1],edge=k[2],motion=k[3],frames=v) for k,v in sorted(output.items())]


def main():
    training_path,result_path=OUT/"training.json",OUT/"result.json"
    training,result=old.read(training_path),old.read(result_path)
    assert old.digest(training_path)==result["training_sha256"]
    assert training["development"]==list(DEV) and training["excluded"]==[TEST,VAL]
    assert old.digest(OUT/"side-three.pt")==training["checkpoint_sha256"]
    for name,sha in training["source_sha256"].items():
        assert TEST not in name and VAL not in name and old.digest(Path(name))==sha
    route=old.read(ROUTING/"straight-verification.json")
    original=old.read(DENSE/"result.json")
    replay={}
    for track,folder in ((TEST,"silverstone"),(VAL,"vallelunga")):
        entry=result["tracks"][track]
        for path,sha in entry["source_sha256"].items():
            assert old.digest(Path(path))==sha
        raw=RAW_ARCHIVE/folder/"sensors.npz"
        with np.load(raw,allow_pickle=False) as source:
            observed=source["sections"]
        expected=frames(ROUTING/f"{track}-state-distance.jsonl",route["tracks"][track]["rows_sha256"])
        checked={}
        for mode,record in entry["methods"].items():
            path=OUT/f"{track}-{mode}-sensors.npz"
            assert old.digest(path)==record["sensors_sha256"]
            with np.load(path,allow_pickle=False) as data:
                assert len(data["frame"])==1200
                np.testing.assert_array_equal(data["sections"][np.isfinite(observed)],observed[np.isfinite(observed)])
            actual=frames(OUT/f"{track}-{mode}-frames.jsonl",record["frames_sha256"])
            checked[mode]=outcomes(actual,expected)
            if mode in ("contrast","two_track"):
                before="contrast" if mode=="contrast" else "heat_and_contrast"
                assert record["frames_sha256"]==original["tracks"][track]["methods"][before]["frames_sha256"]
        replay[track]=checked
    labels={t:label_audit(t) for t in ("imola","mugello","ks_zandvoort")}
    old.write(OUT/"verification.json",dict(status="verified",result_sha256=old.digest(result_path),
              paired=replay,physical_label_audit=labels,termination="frozen_three_circuit_replay_verified"))
    print(labels,flush=True)


if __name__=="__main__":
    main()
