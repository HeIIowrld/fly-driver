"""Verify dense-side replay provenance and the guarded extra supervision."""
from pathlib import Path
import sys

import numpy as np

from train_heat import OUT, CHAINS, DEV, TEST, VAL, ROUTING, old, frames
EXTRA=OUT.parent/"vision-corner-extra-side-labels-20261112-01"
sys.path.insert(0,str(EXTRA))
from export_guarded_labels import overlaps_ground, TRACKS


def checks():
    edges=[dict(lateral_m=-5.,world=[0.,0.,0.]),dict(lateral_m=5.,world=[0.,0.,0.])]
    origin=np.zeros(3)
    def test(span,height):
        return overlaps_ground(np.array([[[span[0],height],[span[1],height]]]),edges,origin)
    assert test((-6.,-5.1),0.)
    assert not test((-8.,-6.1),0.)
    assert not test((-4.,4.),1.5)
    assert test((-4.,4.),.3)


def labels_verified():
    checks()
    summary={}
    for track in TRACKS:
        file=EXTRA/f"{track}-train-labels.npz"
        info=old.read(file.with_suffix(".json"))
        assert info["status"]=="offline_only" and info["track"]==track
        assert old.digest(file)==info["labels_sha256"]
        for name,sha in info["source_sha256"].items():
            assert old.digest(Path(name))==sha
        with np.load(file,allow_pickle=False) as data:
            assert len(data["run"])==len(set(zip(data["run"],data["frame"],strict=True)))==info["frames"]
            assert np.array_equal(data["stations"],old.STATIONS)
            witness=data["adjacent_invalid"]
            assert np.isfinite(data["lateral_m"][witness]).all()
            pixel=data["image_px"]
            valid=witness[:,None]&np.isfinite(pixel).all(-1)
            assert (pixel[valid]>=0).all() and (pixel[valid]<=[319.,179.]).all()
            counts=dict(zip(data["views"].tolist(),valid.sum(axis=(0,2)).tolist(),strict=True))
            assert counts==info["counts_by_view"]
            summary[track]=dict(frames=info["frames"],right=counts["right"][0],left=counts["left"][1],
                               rejected=info["rejected_surface_witnesses"],labels_sha256=old.digest(file))
    return summary


def main():
    training,result_path=OUT/"training.json",OUT/"result.json"
    trained,result=old.read(training),old.read(result_path)
    assert old.digest(training)==result["training_sha256"]
    assert old.digest(OUT/"side-heat.pt")==trained["checkpoint_sha256"]
    assert trained["development"]==list(DEV) and trained["excluded"]==[TEST,VAL]
    for name,sha in trained["source_sha256"].items():
        assert TEST not in name and VAL not in name and old.digest(Path(name))==sha
    previous=old.read(CHAINS/"result.json")
    reference=old.read(ROUTING/"straight-verification.json")
    for track,report in result["tracks"].items():
        for name,sha in report["source_sha256"].items():
            assert old.digest(Path(name))==sha
        prior=frames(ROUTING/f"{track}-state-distance.jsonl",reference["tracks"][track]["rows_sha256"])
        for method,model in report["methods"].items():
            assert old.digest(OUT/f"{track}-{method}-sensors.npz")==model["sensors_sha256"]
            replay=frames(OUT/f"{track}-{method}-frames.jsonl",model["frames_sha256"])
            assert len(replay)==len(prior)==1200
            assert [(r["run"],r["frame"]) for r in replay]==[(r["run"],r["frame"]) for r in prior]
            if method=="contrast":
                assert model["frames_sha256"]==previous["tracks"][track]["methods"]["baseline"]["frames_sha256"]
    additional=labels_verified()
    old.write(OUT/"verification.json",dict(status="verified",result_sha256=old.digest(result_path),
              tracks=[TEST,VAL],additional_offline_supervision=additional,
              termination="dense_replay_and_guarded_label_export_verified"))
    print(additional)


if __name__=="__main__":
    main()
