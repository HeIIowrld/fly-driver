"""Verify fixed candidate support, prior observation preservation, and paired outcomes."""
from collections import Counter
from pathlib import Path

import numpy as np

from rank_side_pairs import OUT, SIDE, RAW_ARCHIVE, ROUTING, DEV, TEST, VAL, old, frames
from verify_side import classify

METHODS = ("temporal","t4t5_confirm")


def main():
    path, result_path = OUT/"training.json", OUT/"result.json"
    training, result = old.read(path), old.read(result_path)
    assert old.digest(path)==result["training_sha256"]
    assert old.digest(OUT/"rankers.joblib")==training["checkpoint_sha256"]
    assert training["development"]==list(DEV) and training["excluded"]==[TEST,VAL]
    for name,sha in training["source_sha256"].items():
        assert TEST not in name and VAL not in name and old.digest(Path(name))==sha
    route=old.read(ROUTING/"straight-verification.json")
    output={}
    for track,folder in ((TEST,"silverstone"),(VAL,"vallelunga")):
        report=result["tracks"][track]
        for name,sha in report["source_sha256"].items():
            assert old.digest(Path(name))==sha
        raw_path=RAW_ARCHIVE/folder/"sensors.npz"
        with np.load(raw_path,allow_pickle=False) as source:
            raw=source["sections"]
        prior=frames(ROUTING/f"{track}-state-distance.jsonl",route["tracks"][track]["rows_sha256"])
        base=frames(OUT/f"{track}-baseline-frames.jsonl",report["methods"]["baseline"]["frames_sha256"])
        gathered={}
        for mode,variant in report["methods"].items():
            sensor=OUT/f"{track}-{mode}-sensors.npz"
            assert old.digest(sensor)==variant["sensors_sha256"]
            with np.load(sensor,allow_pickle=False) as d:
                patched=d["sections"]
            np.testing.assert_array_equal(patched[np.isfinite(raw)],raw[np.isfinite(raw)])
            replay=frames(OUT/f"{track}-{mode}-frames.jsonl",variant["frames_sha256"])
            assert len(base)==len(replay)==len(prior)==1200
            changes=Counter()
            camera=Counter()
            for r,b,label in zip(replay,base,prior,strict=True):
                assert (r["run"],r["frame"])==(b["run"],b["frame"])==(label["run"],label["frame"])
                for method in METHODS:
                    old_state=classify(b,label,method)
                    new_state=classify(r,label,method)
                    if old_state!=new_state:
                        changes[(method,label["truth"],label["label"],old_state,new_state)]+=1
                edge,motion=(classify(r,label,m) for m in METHODS)
                if edge!=motion:
                    camera[(label["truth"],label["label"],edge,motion)]+=1
            gathered[mode]=dict(changes=[dict(method=k[0],truth=k[1],label=k[2],before=k[3],after=k[4],frames=n)
                                          for k,n in sorted(changes.items())],
                                camera=[dict(truth=k[0],label=k[1],edge=k[2],motion=k[3],frames=n)
                                        for k,n in sorted(camera.items())])
        output[track]=gathered
    old.write(OUT/"verification.json",dict(status="verified",tracks=output,
                result_sha256=old.digest(result_path),termination="shared_replay_and_observation_preservation"))


if __name__=="__main__":
    main()
