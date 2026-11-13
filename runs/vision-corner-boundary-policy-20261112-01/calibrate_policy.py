"""Choose a missing-boundary policy using boundary and forecast errors on development recordings."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
PREV = OUT.parent/"vision-corner-boundary-candidate-selection-20261112-01"
sys.path.insert(0, str(PREV))
from select_boundary import (ARCHIVE, DEVELOP, DISTANCE_LIMIT_M, EVALUATION, EXTRA, SEED, SIGNS, SILVER,
    SensorArrays, boundary_scores, combine, examples, features, frames, joblib, load_labels, load_observer,
    observe, old, read_development, reference, row_map, selected, train)

POLICIES = {
    "raw": (1.01, None),
    "prior": (.75, None),
    "score_65": (.65, None),
    "score_55": (.55, None),
    "width_45": (.45, 1.25),
    "width_30": (.30, .75),
}
METHODS = ("temporal", "t4t5_confirm")


def patch(raw, ex, choice, policy):
    cut, width_tolerance = policy
    scores, values, origins = choice
    i,j,k = ex["positions"].T
    keep = scores >= cut
    if width_tolerance is not None:
        widths = raw[:,:,1]-raw[:,:,0]
        offsets = np.array([-2,-1,1,2])
        stations = j[:,None]+offsets
        bounded = (stations >= 0) & (stations < raw.shape[1])
        neighbour_widths = widths[i[:,None], stations.clip(0,raw.shape[1]-1)]
        expected = values + raw[i,j,1-k]*SIGNS[1-k]
        comparable = bounded & np.isfinite(neighbour_widths)
        agrees = comparable & (abs(neighbour_widths-expected[:,None]) <= width_tolerance)
        keep &= (agrees.sum(axis=1) >= 2) & (expected > 3.) & (expected < 20.)
    proposal = np.full_like(raw,np.nan)
    proposal[i,j,k] = values
    mask = np.zeros(raw.shape,bool)
    mask[i,j,k] = keep
    filled = combine(raw,proposal,mask,np.zeros(raw.shape,bool))
    np.testing.assert_array_equal(filled[np.isfinite(raw)],raw[np.isfinite(raw)])
    used = np.isfinite(filled[i,j,k])
    np.testing.assert_array_equal(filled[i[used],j[used],k[used]],values[used]*SIGNS[k[used]])
    return filled


def forecast_errors(output, truth):
    counters = {m:Counter() for m in METHODS}
    for row,ref in zip(output,truth,strict=True):
        assert (row["run"],row["frame"]) == (ref["run"],ref["frame"])
        for method,c in counters.items():
            item = row["forecasts"][method]
            direction = {"left":1,"right":2}.get(item["state"])
            offer = None if direction is None else (direction,item["distance_m"])
            category = old.category(ref,offer)
            c[category] += 1
            if category=="joint_hit":
                c[f"joint_{ref['label']}"] += 1
    return counters


def geometry(raw, updated, edge, labels):
    new = labels & ~np.isfinite(raw) & np.isfinite(updated)
    good = int((new & (abs(updated-edge)<=DISTANCE_LIMIT_M)).sum())
    return dict(accepted=int(new.sum()), accurate=good,
                error_sum_m=float(abs(updated[new]-edge[new]).sum()),
                bilateral_added=int((np.isfinite(updated).all(-1) & ~np.isfinite(raw).all(-1)).sum()))


def select_policy(development):
    aggregates = {}
    for policy in POLICIES:
        c = {m:Counter() for m in METHODS}
        g = Counter()
        for track in development.values():
            g.update(track[policy]["geometry"])
            for method in METHODS:
                c[method].update(track[policy]["forecasts"][method])
        summary = {}
        for method,v in c.items():
            summary[method]=dict(outside=v["straight_alert"]+v["out_of_range_alert"],inside=v["in_corner_alert"],
                                 wrong_direction=v["wrong_direction"],joint=v["joint_hit"],right=v["joint_2"])
        aggregates[policy]=dict(geometry=g,forecast=summary)
    control=aggregates["prior"]["forecast"]

    def safe(info):
        return all(info[m][risk] <= max(control[m][risk],aggregates["raw"]["forecast"][m][risk])
                   for m in METHODS for risk in ("outside","inside","wrong_direction"))
    eligible=[name for name,a in aggregates.items() if safe(a["forecast"])]
    def priority(name):
        a=aggregates[name]
        joint=sum(v["joint"] for v in a["forecast"].values())
        g=a["geometry"]
        return joint, g["accurate"]-(g["accepted"]-g["accurate"]), g["accurate"]
    return max(eligible,key=priority),eligible,aggregates


def score_forecasts(track, original, revised, truth):
    refs = row_map(truth)
    compare = []
    counts = Counter()
    effects = Counter()
    right = []
    for a,b in zip(original, revised, strict=True):
        key = a["run"],a["frame"]
        assert key == (b["run"],b["frame"])
        ref = refs.get(key)
        if ref is None:
            continue
        details = {}
        for version,item in (("before",a),("after",b)):
            p = {}
            for method,forecast in item["forecasts"].items():
                side = {"left":1,"right":2}.get(forecast["state"])
                offer = (side,forecast["distance_m"]) if side is not None else None
                if offer is not None:
                    assert tuple(item["candidate"]) == offer
                category = old.category(ref,offer)
                counts[(version,method,category)] += 1
                if category=="joint_hit":
                    counts[(version,method,f"joint_{ref['label']}")] += 1
                p[method] = dict(**forecast,category=category)
            details[version] = dict(candidate=item["candidate"], forecasts=p)
        a_out = details["after"]["forecasts"]["temporal"]["category"]
        b_out = details["after"]["forecasts"]["t4t5_confirm"]["category"]
        if a_out != b_out:
            effects[f"{a_out} -> {b_out}"] += 1
        compare.append(dict(run=key[0], frame=key[1], truth=ref["truth"], label=ref["label"], **details))
        if ref["truth"]=="approach" and ref["label"]=="2":
            right.append(dict(frame=key[1], **details))
    assert len(compare)==len(truth)
    trace=OUT/f"{track}-frames.jsonl"
    old.jsonl(trace, compare)
    return dict(rows=len(compare), paired=dict(effects), right_approaches=right,
                counts={"/".join(key):value for key,value in counts.items()}, trace_sha256=old.digest(trace))



def run():
    observer, settings, sources = load_observer()
    old_training = old.read(PREV/"training.json")
    assert old.digest(PREV/"selector.joblib")==old_training["checkpoint_sha256"]
    sources.update((Path(__file__),Path(features.__code__.co_filename),Path(train.__code__.co_filename),
                    Path(read_development.__code__.co_filename),PREV/"training.json",PREV/"selector.joblib"))
    development={}
    samples={}
    for track in DEVELOP:
        data, truth, receipt = read_development(track,sources,settings)
        edge, visible=load_labels(EXTRA/f"{track}-train-labels.npz",data,sources,receipt["camera_bundle_sha256_by_run"])
        ex=examples(features(SensorArrays(data),settings)[0])
        samples[track]=(ex,edge,visible)
        development[track]=(data,truth)
    scores={}
    for track,(data,truth) in development.items():
        ex,edge,visible=samples[track]
        model=train([v for t,v in samples.items() if t!=track])
        choice=selected(ex,model)
        report={}
        for name,policy in POLICIES.items():
            original=data["sections"]
            updated=patch(original,ex,choice,policy)
            result=observe(SensorArrays(dict(data,sections=updated)),observer,settings)[0]
            report[name]=dict(geometry=geometry(original,updated,edge,visible),forecasts=forecast_errors(result,truth))
            print(track,name,report[name]["geometry"],[(m,c["joint_hit"],c["straight_alert"]+c["out_of_range_alert"],
                  c["in_corner_alert"]) for m,c in report[name]["forecasts"].items()],flush=True)
        scores[track]=report
    policy,eligible,aggregates=select_policy(scores)
    assert all(t not in str(p) for p in sources for t,_ in EVALUATION)
    training=OUT/"training.json"
    old.write(training,dict(development=DEVELOP,excluded=[t for t,_ in EVALUATION],chosen=policy,eligible=eligible,
              policies=POLICIES, development_results=scores, aggregates=aggregates,seed=SEED,source_revision=old.revision(),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="frozen_before_evaluation_replay"))
    print("Frozen",policy,eligible,flush=True)
    fitted=joblib.load(PREV/"selector.joblib")["model"]
    results={}
    for track,folder in EVALUATION:
        root=ARCHIVE/folder
        sensor=root/"sensors.npz"
        receipt=root/"result.json"
        assert old.digest(sensor)==old.read(receipt)["inputs_sha256"]
        with np.load(sensor,allow_pickle=False) as d:
            data=SensorArrays({k:d[k] for k in d.files})
        raw=data["sections"]
        ex=examples(features(data,settings)[0])
        updated=patch(raw,ex,selected(ex,fitted),POLICIES[policy])
        forecasts=observe(SensorArrays(dict(data,sections=updated)),observer,settings)[0]
        archived=root/"predictions/forecasts.jsonl"
        original=frames(archived,old.read(archived.with_name("result.json"))["forecasts_sha256"])
        truth, truth_path=reference(track)
        proof={sensor,receipt,archived,truth_path}
        item=score_forecasts(track,original,forecasts,truth)
        if track==SILVER:
            mesh=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            edge, visible=load_labels(mesh,data,proof)
            refs=row_map(truth)
            right=np.array([refs[r["run"],r["frame"]]["truth"]=="approach" and refs[r["run"],r["frame"]]["label"]=="2" for r in original])
            assert right.sum()==13
            item["edges"]={side:boundary_scores(raw[right,:,i],updated[right,:,i],edge[right,:,i],visible[right,:,i])
                            for i,side in enumerate(("right","left"))}
        output=OUT/f"{track}-boundaries.npz"
        np.savez_compressed(output,run=data["run"],frame=data["frame"],stations=data["stations"],sections=updated)
        item.update(boundaries_sha256=old.digest(output),source_sha256={str(p):old.digest(p) for p in sorted(proof)})
        results[track]=item
        print(track,item["counts"],item.get("edges"),flush=True)
    old.write(OUT/"result.json",dict(policy=policy,tracks=results,training_sha256=old.digest(training),
                                     termination="shared_candidates_scored"))


if __name__=="__main__":
    run()
