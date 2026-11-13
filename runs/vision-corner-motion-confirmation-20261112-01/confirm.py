"""Measure whether T4/T5 can confirm geometry proposals that fill a prior abstention."""
from collections import Counter
from pathlib import Path

import numpy as np

from baseline import SILVER
from evaluate_fused_corner import frame_key,read_rows
from paired_motion import inputs
import profile_replay as profile
from train_vision_teacher_line import digest,jsonl,read,write

OUT=Path(__file__).parent
RECOVERY=Path("runs/vision-corner-image-normal-20261112-01")
EVAL=(SILVER,"spa")
MODES=("geometry","temporal_geometry","motion_veto","motion_local","motion_onset",
       "combined_local","combined_onset")
CUTOFF=.5


def signals(values,direction,distance):
    if values is None:
        return dict(available=False,local=None,onset=None)
    sign=1 if direction==1 else -1
    close=(profile.OFFSETS>=distance-2)&(profile.OFFSETS<=distance+4)
    return dict(available=True,local=float(np.mean(values[close])*sign),
                onset=profile.onset(values,CUTOFF,True))


def gates(motion,combined,direction,distance,temporally_consistent):
    def agrees(value):
        p=value["onset"]
        return p is not None and p[0]==direction and abs(p[1]-distance)<=5.
    return dict(geometry=True,temporal_geometry=temporally_consistent,
                motion_veto=motion["available"] and motion["local"]>=-.25,
                motion_local=motion["available"] and motion["local"]>=.25,
                motion_onset=agrees(motion),combined_local=combined["available"] and combined["local"]>=.25,
                combined_onset=agrees(combined))


def corrections(row):
    if row["truth"]=="approach":
        return ("approach_joint" if row["joint_hit"] else
                "approach_direction" if row["direction_hit"] else "approach_wrong_side")
    if row["truth"] in ("straight","out_of_range"):
        return row["truth"]+"_false_alert"
    return row["truth"]+"_alert"


def main():
    hashes={Path(__file__),Path(inputs.__code__.co_filename),Path(profile.__file__)}
    fallback_path=RECOVERY/"fallback-result.json"
    fallback=read(fallback_path)
    raw_path=RECOVERY/"result.json"
    raw=read(raw_path)
    alerts_path=RECOVERY/"fallback-alerts.jsonl"
    assert digest(alerts_path)==fallback["rows_sha256"]
    hashes.update((fallback_path,raw_path,alerts_path))
    alerts=read_rows(alerts_path)
    training=tuple(profile.CALIBRATION_TRACKS)
    assert not set(training)&set(EVAL)
    groups={name:inputs(name,hashes) for name in training}
    feature_modes=("t4t5","boundaries_temporal_t4t5")
    models={mode:profile.fit([groups[name][mode] for name in training],1) for mode in feature_modes}
    del groups
    results={}
    details=[]
    for track in EVAL:
        groups=inputs(track,hashes)
        geometry_path=RECOVERY/f"{track}-frames.jsonl"
        assert digest(geometry_path)==raw["tracks"][track]["rows_sha256"]
        hashes.add(geometry_path)
        geometry={frame_key(row):row for row in read_rows(geometry_path)}
        g=groups["t4t5"]
        rows={frame_key(r):r for r in g["rows"]}
        profiles={}
        for mode in feature_modes:
            predicted=profile.predict(groups[mode],models[mode])
            profiles[mode]={frame_key(row):v for row,v in zip(g["selected"],predicted,strict=True)}
        totals={name:Counter() for name in MODES}
        for alert in alerts:
            if alert["track"]!=track:
                continue
            key=frame_key(alert)
            row=rows[key]
            known=geometry[key]["methods"]
            assert known["baseline"]["state"]=="abstain"
            assert alert["truth"]==row["truth"] and alert["label"]==row["label"]
            candidate=known["normal_fill"]
            assert candidate["state"] in ("left","right") and alert["distance_m"]==candidate["distance_m"]
            direction=1 if candidate["state"]=="left" else 2
            distance=alert["distance_m"]
            motion,combined=(signals(profiles[mode].get(key),direction,distance) for mode in feature_modes)
            prior_key=(key[0],key[1]-1)
            previous=geometry.get(prior_key)
            adjacent=prior_key in rows and rows[prior_key]["generation"]==row["generation"] and (
                0.<row["received_s"]-rows[prior_key]["received_s"]<=.18)
            before=previous["methods"]["normal_fill"] if previous else None
            consistent=bool(adjacent and before and before["state"]==candidate["state"]
                            and -1.<=before["distance_m"]-distance<=12.)
            accepted=gates(motion,combined,direction,distance,consistent)
            assert tuple(accepted)==MODES
            label=corrections(alert)
            for mode,accept in accepted.items():
                totals[mode][label]+=int(accept)
            details.append(dict(track=track,run=key[0],frame=key[1],truth=row["truth"],label=label,
                               proposal_side=direction,distance_m=distance,
                               motion=motion,combined=combined,accepted=accepted))
        baseline=fallback["tracks"][track]["original"]
        metrics={}
        for mode,accepted in totals.items():
            metrics[mode]=dict(accepted=dict(accepted),counts=dict(baseline["metrics"]["candidate"]))
            counts=metrics[mode]["counts"]
            for evidence in details:
                if evidence["track"]!=track or not evidence["accepted"][mode]:
                    continue
                if evidence["label"] in ("approach_joint","approach_direction"):
                    counts["left" if evidence["proposal_side"]==1 else "right"]+=1
                counts["joint"]+=int(evidence["label"]=="approach_joint")
                counts["outside"]+=int(evidence["label"] in ("straight_false_alert","out_of_range_false_alert"))
        assert metrics["geometry"]["counts"]==fallback["tracks"][track]["fallback"]["metrics"]["candidate"]
        results[track]=metrics
        print(track,{mode:r["counts"] for mode,r in metrics.items()},flush=True)
    path=OUT/"proposals.jsonl"
    jsonl(path,details)
    write(OUT/"result.json",dict(status="evaluated",seed=None,training=training,
          tracks=results,rows_sha256=digest(path),source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
