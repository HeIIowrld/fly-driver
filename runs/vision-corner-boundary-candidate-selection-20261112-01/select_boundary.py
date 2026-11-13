"""Select a real boundary observation for a missing side/station, then replay the corner observer."""
from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

OUT = Path(__file__).resolve().parent
FUSION = OUT.parent/"vision-corner-supervised-boundary-fusion-20261112-01"
sys.path.insert(0, str(FUSION))
from fit_fusion import (ARCHIVE, DEVELOP, EXTRA, MONZA, SEED, SIGNS, SILVER, VAL, SensorArrays,
    boundary_scores, combine, features, frames, load_labels, load_observer, observe, old, read_development, reference)

DISTANCE_LIMIT_M = 1.25
CUTOFFS = (0., .35, .45, .55, .65, .75, .85, .95, 1.01)
SOURCE_COLUMNS = (1,11,12,13,14)
EVALUATION = ((SILVER,"silverstone"), (VAL,"vallelunga"), (MONZA,"monza"))


def examples(f):
    donor = f[..., SOURCE_COLUMNS]
    targets = ~np.isfinite(f[...,0]) & np.isfinite(donor).any(-1)
    positions = np.argwhere(targets)
    context = f[targets]
    values = donor[targets]
    cell, source = np.nonzero(np.isfinite(values))
    x = values[cell, source]
    agreement = abs(x[:,None]-values[cell])
    consensus = (agreement <= .75).sum(1) - 1
    support = np.isfinite(values[cell]).sum(1)
    columns = np.column_stack((context[cell,1:], source, x, support, consensus,
                               x[:,None]-context[cell][:,[1,2,3,4,5,6,7,8]]))
    return dict(positions=positions, cell=cell, source=source, values=x, inputs=columns,
                agreement=consensus, available=support)


def train(data):
    columns, targets, weights = [], [], []
    for ex, edge, visible in data:
        pos = tuple(ex["positions"].T)
        eligible = visible[pos][ex["cell"]]
        expected = (edge*SIGNS)[pos][ex["cell"]][eligible]
        size = ex["available"][eligible]
        weight = 1./size
        columns.append(ex["inputs"][eligible])
        targets.append(abs(ex["values"][eligible]-expected) <= DISTANCE_LIMIT_M)
        weights.append(weight/weight.sum())
    w = np.concatenate(weights)
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=32,
                                            l2_regularization=2., random_state=SEED)
    return model.fit(np.concatenate(columns), np.concatenate(targets), sample_weight=w/w.mean())


def selected(ex, model=None, rule="learned"):
    cell, source = ex["cell"], ex["source"]
    if rule == "learned":
        score = model.predict_proba(ex["inputs"])[:,1]
    elif rule == "nearest":
        score = 1.-source/10.
    elif rule == "agreement":
        score = ex["agreement"]/4.
    else:
        raise ValueError(rule)
    order = np.lexsort((source,-score,cell))
    _, first = np.unique(cell[order], return_index=True)
    best = order[first]
    assert np.array_equal(cell[best],np.arange(len(ex["positions"])))
    return score[best], ex["values"][best], source[best]


def measure(ex, choice, edge, visible, cut):
    scores, values, source = choice
    pos = tuple(ex["positions"].T)
    match = visible[pos]
    use = match & (scores >= cut)
    errors = abs(values - (edge*SIGNS)[pos])
    good = use & (errors <= DISTANCE_LIMIT_M)
    return dict(eligible=int(match.sum()), accepted=int(use.sum()), accurate=int(good.sum()),
                error_sum_m=float(errors[use].sum()), mae_m=float(errors[use].mean()) if use.any() else None,
                source_counts=dict(Counter(source[use].tolist())))


def freeze(dev):
    folds = {}
    for held, (ex, edge, visible) in dev.items():
        model = train([v for key,v in dev.items() if key != held])
        ranked = selected(ex, model)
        folds[held] = dict(learned={cut:measure(ex, ranked, edge, visible, cut) for cut in CUTOFFS},
                           nearest=measure(ex, selected(ex, rule="nearest"), edge, visible, 0.),
                           agreement=measure(ex, selected(ex, rule="agreement"), edge, visible, 0.))
        print(held, [(cut,m["accepted"],m["accurate"]) for cut,m in folds[held]["learned"].items()],flush=True)
    def quality(cut):
        summaries = [fold["learned"][cut] for fold in folds.values()]
        # Two misses at 1.25 m cost the same as one accepted estimate outside tolerance.
        utility = sum((3*r["accurate"]-2*r["accepted"])/r["eligible"] for r in summaries)
        return utility, sum(r["accurate"] for r in summaries), cut
    cut = max(CUTOFFS, key=quality)
    return train(list(dev.values())), cut, folds, {c:quality(c)[0] for c in CUTOFFS}


def row_map(rows):
    return {(r["run"],r["frame"]):r for r in rows}


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
    sources.update((Path(__file__),Path(features.__code__.co_filename),Path(read_development.__code__.co_filename)))
    dev = {}
    for track in DEVELOP:
        data, _, receipt = read_development(track, sources, settings)
        f,_ = features(SensorArrays(data), settings)
        edge, visible = load_labels(EXTRA/f"{track}-train-labels.npz", data, sources, receipt["camera_bundle_sha256_by_run"])
        dev[track] = examples(f),edge,visible
    model, cut, folds, utility = freeze(dev)
    checkpoint = OUT/"selector.joblib"
    joblib.dump(dict(model=model, cut=cut),checkpoint)
    training = OUT/"training.json"
    assert all(t not in str(p) for p in sources for t in (SILVER, VAL, MONZA))
    old.write(training, dict(development=DEVELOP, excluded=[SILVER,VAL,MONZA], threshold=cut, folds=folds,
              development_utility=utility, seed=SEED, source_revision=old.revision(), checkpoint_sha256=old.digest(checkpoint),
              source_sha256={str(p):old.digest(p) for p in sorted(sources)}, termination="frozen_before_video_replay"))
    print("Frozen",cut,utility,flush=True)
    results={}
    for track,folder in EVALUATION:
        root=ARCHIVE/folder
        path=root/"sensors.npz"
        assert old.digest(path)==old.read(root/"result.json")["inputs_sha256"]
        with np.load(path,allow_pickle=False) as arrays:
            data=SensorArrays({k:arrays[k] for k in arrays.files})
        f,_=features(data,settings)
        ex=examples(f)
        choice=selected(ex,model)
        score,value,source=choice
        pos=tuple(ex["positions"].T)
        raw=data["sections"]
        candidate=np.full_like(raw,np.nan)
        candidate[pos]=value
        accept=np.zeros(raw.shape,bool)
        accept[pos]=score>=cut
        updated=combine(raw,candidate,accept,np.zeros(raw.shape,bool))
        known=np.isfinite(raw)
        filled=~known & np.isfinite(updated)
        np.testing.assert_array_equal(updated[known],raw[known])
        assert np.all(accept[filled])
        chosen = np.zeros(raw.shape,bool)
        chosen[pos]=np.isclose(value,f[pos][:,SOURCE_COLUMNS][np.arange(len(source)),source])
        assert np.all(chosen[filled])
        revised=observe(SensorArrays(dict(data,sections=updated)),observer,settings)[0]
        archive=root/"predictions/forecasts.jsonl"
        original=frames(archive,old.read(archive.with_name("result.json"))["forecasts_sha256"])
        truth, truth_path=reference(track)
        proof={path,root/"result.json",archive,truth_path}
        result=score_forecasts(track,original,revised,truth)
        if track==SILVER:
            mesh=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{track}-train-labels.npz"
            edge,visible=load_labels(mesh,data,proof)
            refs=row_map(truth)
            right=np.array([refs[r["run"],r["frame"]]["truth"]=="approach" and refs[r["run"],r["frame"]]["label"]=="2" for r in original])
            assert right.sum()==13
            result["right_approach_edges"]={side:boundary_scores(raw[right,:,i],updated[right,:,i],edge[right,:,i],visible[right,:,i])
                                            for i,side in enumerate(("right","left"))}
            i,j,k=ex["positions"].T
            focus=right[i] & visible[pos] & (k==0)
            assert focus.sum()==10
            expected=(edge*SIGNS)[pos]
            diagnostic={}
            for rule in ("learned","nearest","agreement"):
                picked=choice if rule=="learned" else selected(ex,rule=rule)
                errors=abs(picked[1]-expected)
                diagnostic[rule]=dict(mae_m=float(errors[focus].mean()), accurate=int((errors[focus]<=DISTANCE_LIMIT_M).sum()))
            result["right_missing_diagnostic"]=diagnostic
            result["right_missing_points"]=[dict(frame=int(data["frame"][i[t]]), station_m=float(data["stations"][j[t]]),
                source=int(source[t]), error_m=float(abs(value[t]-expected[t])), confidence=float(score[t]), filled=bool(filled[pos][t]))
                for t in np.flatnonzero(focus)]
        output=OUT/f"{track}-boundaries.npz"
        np.savez_compressed(output,run=data["run"],frame=data["frame"],stations=data["stations"],sections=updated)
        result.update(new_points=int(filled.sum()), boundaries_sha256=old.digest(output),
                      source_sha256={str(p):old.digest(p) for p in sorted(proof)})
        results[track]=result
        print(track,result["new_points"],result.get("right_missing_diagnostic"),result.get("right_approach_edges"),result["counts"],flush=True)
    old.write(OUT/"result.json",dict(tracks=results,training_sha256=old.digest(training),termination="fixed_selection_replay_scored"))


if __name__=="__main__":
    run()
