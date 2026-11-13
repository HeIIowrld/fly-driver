"""Pair error transitions when adding T4/T5 inputs to the same road profile fit."""
from collections import Counter
from pathlib import Path

import numpy as np

from baseline import SILVER,load_silver,publish
from evaluate_fused_corner import frame_key
from motion_contribution import MODES,CUTOFF,boundary_features,silver_labels
import profile_replay as profile
from train_vision_teacher_line import digest,jsonl,write

OUT=Path(__file__).parent
EVALUATION=("spa",SILVER)


def inputs(track,hashes):
    group=load_silver(hashes) if track==SILVER else profile.load(track,hashes)
    if track not in EVALUATION:
        group["labels"]=profile.labels(track,group,hashes)
    edge=boundary_features(track,group,hashes)
    motion=group["x"]
    half=motion.shape[-1]//2
    step=half//3
    assert 3*step==half
    latest=np.r_[0:step,half:half+step]
    return {mode:dict(group,x=features) for mode,features in zip(MODES,(
        edge,motion,np.concatenate((edge,motion[:,:,latest]),axis=2),
        np.concatenate((edge,motion),axis=2)),strict=True)}


def score(group,model):
    values=iter(profile.predict(group,model))
    for row,valid in zip(group["rows"],group["valid"],strict=True):
        p=profile.onset(next(values),CUTOFF,True) if valid else None
        yield publish(row,p,valid,"features_unavailable")


def classify(row):
    alert=row["state"] in ("left","right")
    if row["truth"]=="approach":
        if row["joint_hit"]:
            return "direction_and_distance"
        if row["direction_hit"]:
            return "distance_error"
        if alert:
            return "wrong_direction"
        return "missed_approach" if row["state"]=="straight" else "approach_abstained"
    if row["truth"] in ("straight","out_of_range"):
        return "false_alert" if alert else "quiet"
    return "other"


def main():
    hashes={Path(__file__),Path(inputs.__code__.co_filename),Path(profile.__file__),
            Path(boundary_features.__code__.co_filename),Path(publish.__code__.co_filename)}
    training=tuple(profile.CALIBRATION_TRACKS)
    train={track:inputs(track,hashes) for track in training}
    models={mode:profile.fit([train[track][mode] for track in training],1) for mode in MODES}
    comparison={}
    details=[]
    for track in EVALUATION:
        assert track not in training
        data=inputs(track,hashes)
        predictions={mode:list(score(group,models[mode])) for mode,group in data.items()}
        counts={}
        for a,b in (("boundaries","boundaries_temporal_t4t5"),
                    ("boundaries_latest_t4t5","boundaries_temporal_t4t5")):
            pairs=Counter()
            for before,after in zip(predictions[a],predictions[b],strict=True):
                assert frame_key(before)==frame_key(after)
                truth=before["truth"]
                if truth not in ("approach","straight","out_of_range"):
                    continue
                start,stop=classify(before),classify(after)
                cohort=(("left" if before["label"]=="1" else "right") if truth=="approach" else truth)
                pairs[cohort,start,stop]+=1
                if start!=stop or truth=="approach":
                    details.append(dict(track=track,run=before["run"],frame=before["frame"],cohort=cohort,
                        before_model=a,after_model=b,before=start,after=stop,before_state=before["state"],
                        after_state=after["state"],before_distance_m=before["predicted_distance_m"],
                        after_distance_m=after["predicted_distance_m"],truth_m=before["truth_distance_m"]))
            counts[f"{a}_to_{b}"]=[dict(cohort=k[0],before=k[1],after=k[2],frames=n)
                                   for k,n in sorted(pairs.items())]
        comparison[track]=counts
        print(track,"paired",len(data["boundaries"]["rows"]),flush=True)
    path=OUT/"motion-error-transitions.jsonl"
    jsonl(path,details)
    write(OUT/"motion-error-transitions.json",dict(status="evaluated",seed=None,training=training,cutoff=CUTOFF,
          comparisons=comparison,rows_sha256=digest(path),
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
