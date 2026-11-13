"""Test exchanging old warnings for independently supported curvature warnings."""
from itertools import combinations
from pathlib import Path

import numpy as np

from profile_replay import CALIBRATION_TRACKS, OFFSETS, TRACKS, fit, labels, load, onset, predict
from train_vision_teacher_line import digest, write
from transfer import compare_tracks,score

OUT=Path(__file__).parent
SETTINGS=[(cut,support) for cut in (.2,.35,.5,.65) for support in (0.,.10,.20,.35)]+[(None,None)]

def change(group,profiles,setting):
    cut,minimum=setting
    out=[]
    predicted=iter(profiles)
    for row,valid in zip(group["rows"],group["valid"],strict=True):
        if not valid:
            out.append(row)
            continue
        profile=next(predicted)
        if cut is None:
            out.append(row)
            continue
        selection=onset(profile,cut,True)
        active=row["state"] in ("left","right")
        if active:
            side=1 if row["state"]=="left" else -1
            near=np.abs(OFFSETS-row["predicted_distance_m"])<=5.
            strength=float((side*profile[near]).max())
            if strength<minimum:
                row=dict(row,state="abstain",abstain_reason="profile_rejected",direction_hit=False,
                         joint_hit=False,signed_distance_error_m=None,absolute_distance_error_m=None)
        elif selection and (row["state"]=="straight" or row["state"]=="abstain"
                            and row["abstain_reason"]=="presence_rejected"):
            row=score(row,selection)
        out.append(row)
    return compare_tracks(group["original"],out,group["rules"])

def select(calibration,groups):
    ranked=[]
    for setting in SETTINGS:
        results={name:change(groups[name],estimate,setting) for name,estimate in calibration.items()}
        passed=True
        for name,result in results.items():
            ref=groups[name]["before"]
            base=ref["metrics"]["candidate"]
            check=result["metrics"]["candidate"]
            passed &= (all(check[k]>=base[k] for k in ("left","right","joint"))
                       and check["outside"]<=max(base["outside"],ref["metrics"]["parent"]["outside"])
                       and len(result["stable_leads_m"])>=len(ref["stable_leads_m"]))
        if passed:
            n=lambda k:sum(r["metrics"]["candidate"][k] for r in results.values())
            rank=(n("left"),n("joint"),-n("outside"),setting[0] is None,setting[0] or 0.,setting[1] or 0.)
            ranked.append((rank,setting))
    return max(ranked)[1]

def run():
    sources={Path(__file__),Path("runs/vision-corner-curvature-profile-20261111-01/profile_replay.py"),
             Path("runs/vision-corner-matched-approaches-20261111-01/pair_audit.py")}
    groups={}
    for track in TRACKS:
        group=load(track,sources)
        group["labels"]=labels(track,group,sources)
        groups[track]=group
    reports={}
    for kind,name in enumerate(("curvature","span")):
        models={keys:fit([groups[k] for k in keys],kind) for size in (3,4,5)
                for keys in combinations(CALIBRATION_TRACKS,size)}
        reports[name]={}
        for track,group in groups.items():
            train=tuple(k for k in CALIBRATION_TRACKS if k!=track)
            calibration={k:predict(groups[k],models[tuple(i for i in train if i!=k)]) for k in train}
            setting=select(calibration,groups)
            profile=predict(group,models[train])
            selected=change(group,profile,setting)
            sweep={str(setting):change(group,profile,setting)["metrics"]["candidate"] for setting in SETTINGS}
            reports[name][track]=dict(chosen=setting,comparison=selected,diagnostic=sweep)
            print(name,track,setting,selected["metrics"]["candidate"],flush=True)
    write(OUT/"replacement-result.json",dict(tracks=reports,seed=None,
          source_sha256={str(p):digest(p) for p in sorted(sources)},termination="profile_replacement_comparison_complete"))

if __name__=="__main__":
    run()
