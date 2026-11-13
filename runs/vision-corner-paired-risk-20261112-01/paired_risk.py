"""Calibrate how much same-candidate T4/T5 confirmation to use at a common development risk."""
from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from replay_corner_boundary_states import (load_observer, old, sensor_rows, proposals,
    rank_candidates, frame_features, phase_scores, sequence, alt, ROUTING)
from routing import ALIGNED, LISTWISE, DEVELOP, SILVER, VAL, expand, routed, train, risk_cut
from train_phase import fit_phase
from verify_directional import frames

MONZA = "monza"
ARCHIVE = OUT.parent/"vision-corner-archive-observer-20261112-01"
MONZA_LABELS = OUT.parent/"vision-monza-two-lap-corner-inventory-20261111-01"/"rows.jsonl"
WEIGHTS = (.0,.25,.5,.75,1.)
POLICY = "empty_or_consistent"
METHODS = ("temporal","t4t5_confirm")
OUTSIDE = ("straight_alert","out_of_range_alert")


def counts(scores, labels, cut):
    accepted = scores>=cut
    return dict(outside=int((accepted&np.isin(labels,OUTSIDE)).sum()),
                inside=int((accepted&(labels=="in_corner_alert")).sum()),
                hits=int((accepted&(labels=="joint_hit")).sum()))


def development(settings, sources):
    groups = {track:expand(track,sources)[:2] for track in DEVELOP}
    vectors, labels = [],[]
    for track, (group, base) in groups.items():
        others = [b for name,(_,b) in groups.items() if name!=track]
        methods = routed(group,base,train(others))[POLICY]
        q = phase_scores(base,fit_phase(others))[:,2]
        for edge,motion in zip(*(methods[m] for m in METHODS),strict=True):
            assert edge[:2]==motion[:2]
            i,j = edge[:2]
            assert np.isfinite(q[i])
            vectors.append((edge[2],motion[2]) if q[i]<=settings["phase_cut"] else (-1.,-1.))
            labels.append(group["errors"][j])
    return np.asarray(vectors),np.asarray(labels)


def calibrate(vectors, labels, settings):
    current = {name:counts(vectors[:,i],labels,settings["cuts"][name]) for i,name in enumerate(METHODS)}
    cap = {key:min(c[key] for c in current.values()) for key in ("outside","inside")}
    variants = {}
    for weight in WEIGHTS:
        scores = vectors @ np.array([1.-weight,weight])
        cut = risk_cut(scores,labels,cap["outside"],cap["inside"])
        variants[str(weight)] = dict(weight=weight,cut=cut,**counts(scores,labels,cut))
    winner = max(variants.values(),key=lambda d:(d["hits"],-d["outside"]-d["inside"],-d["weight"]))
    return current,cap,variants,winner


def archive_scores(bundle, settings, models):
    with np.load(bundle,allow_pickle=False) as data:
        rows = sensor_rows(data,settings)
        body, sections, heads, visual = (data[k] for k in ("motion","sections","heads","t4t5"))
    adjacent = [False,*(alt.adjacent(a,b) for a,b in zip(rows,rows[1:]))]
    aligned = sequence(old.STATIONS,adjacent,sections,body)
    phase = frame_features(rows,sections,body)
    q = phase_scores(dict(phase_x=phase),models["phase"])[:,2]
    original = proposals(rows,sections,body,heads)
    new = proposals(rows,aligned,body,heads)
    choices, likelihood = rank_candidates(rows,aligned,sections,body,visual,phase,original,new,models["rank"])
    return rows,choices,likelihood,q


def reference(track):
    if track==MONZA:
        data = old.read(ARCHIVE/"monza/onset-transfer-score.json")
        stored = {Path(path).resolve(): digest for path,digest in data["source_sha256"].items()}
        raw = frames(MONZA_LABELS,stored[MONZA_LABELS.resolve()])
        rows = []
        for r in raw:
            d = r["distance"]
            truth = "approach" if d is not None and 4.<=d<=20. else "out_of_range"
            rows.append(dict(run=r["run"],frame=r["frame"],truth=truth,label=str(r["direction"]),
                             truth_endpoints_m=[] if d is None else [d]))
        return rows,MONZA_LABELS
    path = ROUTING/f"{track}-state-distance.jsonl"
    digest = old.read(ROUTING/"straight-verification.json")["tracks"][track]["rows_sha256"]
    return frames(path,digest),path


def score(track, name, settings, models, calibration):
    directory = ARCHIVE/name
    receipt = old.read(directory/"result.json")
    bundle = directory/"sensors.npz"
    assert old.digest(bundle)==receipt["inputs_sha256"]
    rows,choices,likelihood,q = archive_scores(bundle,settings,models)
    truth,path = reference(track)
    keys = {(r["run"],r["frame"]):i for i,r in enumerate(rows)}
    prior = directory/"predictions/forecasts.jsonl"
    published = frames(prior,old.read(prior.with_name("result.json"))["forecasts_sha256"])
    records = []
    gate = calibration["variants"]
    for r in truth:
        i = keys[r["run"],r["frame"]]
        offer = choices.get(i)
        pre = likelihood.get(i,(np.nan,np.nan))
        pair = np.array(pre) if np.isfinite(q[i]) and q[i]<=settings["phase_cut"] else np.array([-1.,-1.])
        estimates = {}
        for j,method in enumerate(METHODS):
            accepted = offer if offer is not None and pair[j]>=settings["cuts"][method] else None
            observed = published[i]["forecasts"][method]
            actual = ({"left":1,"right":2}[observed["state"]],observed["distance_m"]) if observed["state"] in ("left","right") else None
            assert accepted==actual
            estimates[method]=accepted
        for name,p in gate.items():
            w=p["weight"]
            value = pair[0] if w==0. else float(pair @ np.array([1.-w,w]))
            estimates["blend_"+name]=offer if offer is not None and value>=p["cut"] else None
        records.append(dict(run=r["run"],frame=r["frame"],truth=r["truth"],label=r["label"],candidate=offer,
                            outcomes={m:old.category(r,offer) for m,offer in estimates.items()}))
    metrics = {}
    for name in records[0]["outcomes"]:
        c = Counter()
        for r in records:
            outcome = r["outcomes"][name]
            c[outcome]+=1
            if outcome=="joint_hit":
                c[f"joint_{r['label']}"]+=1
        metrics[name]=dict(c)
    changed = OUT/f"{track}-frames.jsonl"
    old.jsonl(changed,records)
    return dict(metrics=metrics,frames=len(records),rows_sha256=old.digest(changed),
                source_sha256={str(p):old.digest(p) for p in (bundle,path,prior,directory/"result.json")})


def main():
    models,settings,sources=load_observer()
    sources.update((Path(__file__),Path(expand.__code__.co_filename),Path(routed.__code__.co_filename),
                    Path(train.__code__.co_filename),Path(fit_phase.__code__.co_filename)))
    vectors,labels=development(settings,sources)
    assert not any(SILVER in str(p) or VAL in str(p) or MONZA in str(p) for p in sources)
    fixed,caps,variants,winner=calibrate(vectors,labels,settings)
    frozen=OUT/"training.json"
    training=dict(status="frozen_before_replay", seed=20261112,source_revision=old.revision(),
                  development=DEVELOP,excluded=[SILVER,VAL,MONZA],fixed=fixed,caps=caps,variants=variants,
                  winner=winner,source_sha256={str(p):old.digest(p) for p in sorted(sources)},
                  termination="crossfit_and_common_false_alarm_limits")
    old.write(frozen,training)
    print("Frozen",winner,caps,flush=True)
    reports={t:score(t,n,settings,models,training) for t,n in ((SILVER,"silverstone"),(VAL,"vallelunga"),(MONZA,"monza"))}
    old.write(OUT/"result.json",dict(status="three_archival_replays",training_sha256=old.digest(frozen),tracks=reports,
                                    termination="paired_confirmation_risk_comparison"))
    for track,report in reports.items():
        print(track,report["metrics"],flush=True)


if __name__=="__main__":
    main()
