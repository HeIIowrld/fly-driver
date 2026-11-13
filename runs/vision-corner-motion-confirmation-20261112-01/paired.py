"""Compare the same local confirmation threshold with and without T4/T5 inputs."""
from collections import Counter
from pathlib import Path

from baseline import SILVER
from confirm import EVAL,RECOVERY,signals
from evaluate_fused_corner import frame_key,read_rows
from motion_contribution import boundary_features
from paired_motion import inputs
import profile_replay as profile
from train_vision_teacher_line import digest,jsonl,read,write

OUT=Path(__file__).parent


def load_boundary(track,hashes):
    if track==SILVER:
        return inputs(track,hashes)["boundaries"]
    group=profile.load(track,hashes)
    group["x"]=boundary_features(track,group,hashes)
    if track not in EVAL:
        group["labels"]=profile.labels(track,group,hashes)
    return group


def main():
    hashes={Path(__file__),Path(signals.__code__.co_filename),Path(profile.__file__),
            Path(boundary_features.__code__.co_filename)}
    receipt=OUT/"result.json"
    result=read(receipt)
    path=OUT/"proposals.jsonl"
    assert digest(path)==result["rows_sha256"]
    hashes.update((receipt,path))
    proposals=read_rows(path)
    training=tuple(profile.CALIBRATION_TRACKS)
    assert set(training).isdisjoint(EVAL) and list(training)==result["training"]
    model=profile.fit([load_boundary(t,hashes) for t in training],1)
    fallback_receipt=RECOVERY/"fallback-result.json"
    old=read(fallback_receipt)
    hashes.add(fallback_receipt)
    results={}
    details=[]
    for track in EVAL:
        group=load_boundary(track,hashes)
        p=profile.predict(group,model)
        profiles={frame_key(row):value for row,value in zip(group["selected"],p,strict=True)}
        base=dict(old["tracks"][track]["original"]["metrics"]["candidate"])
        totals=dict(base)
        changes=Counter()
        for row in proposals:
            if row["track"]!=track:
                continue
            side=row["proposal_side"]
            value=signals(profiles.get(frame_key(row)),side,row["distance_m"])
            accepted=value["available"] and value["local"]>=.25
            combined=row["accepted"]["combined_local"]
            kind=row["label"]
            changes[kind,accepted,combined]+=1
            if accepted:
                if kind in ("approach_joint","approach_direction"):
                    totals["left" if side==1 else "right"]+=1
                totals["joint"]+=int(kind=="approach_joint")
                totals["outside"]+=int(kind in ("straight_false_alert","out_of_range_false_alert"))
            details.append(dict(track=track,run=row["run"],frame=row["frame"],label=kind,geometry=value,
                                 with_boundary=accepted,with_t4t5=combined))
        results[track]=dict(boundary_local=totals,combined_local=result["tracks"][track]["combined_local"]["counts"],
              transitions=[dict(label=k[0],boundary_local=k[1],combined_local=k[2],frames=v)
                           for k,v in sorted(changes.items())])
        print(track,totals,flush=True)
    rows=OUT/"paired-confirmation.jsonl"
    jsonl(rows,details)
    write(OUT/"paired-result.json",dict(status="evaluated",seed=None,training=training,threshold=.25,
          tracks=results,rows_sha256=digest(rows),source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
