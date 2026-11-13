"""Score identical learned-boundary candidates with paired frozen confirmation models."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fit_gate import OUT, DEVELOP, SILVER, MODES, OUTSIDE, data, scores, old
from features import cached, SHAPE
from evaluate_shape import ROOT
from score_corner_captures import summarize
import joblib

OUTPUTS = ("candidate_only","shared_edge")


def compare_outcome(row,offer):
    if offer is None:
        return "quiet"
    return old.category(row,offer)


def evaluate(group, probability, thresholds, sections):
    entry = old.context()
    runner = entry.import_replay()
    timelines = {f"{mode}_{output}":defaultdict(list) for mode in MODES for output in OUTPUTS}
    transitions = {out:Counter() for out in OUTPUTS}
    accepted = {mode:{i:float(v) for i,v in zip(group["index"],probability[mode],strict=True)}
                for mode in MODES}
    assert set(accepted["edge"])==set(accepted["t4t5"])
    pairs = np.isfinite(sections).all(axis=-1)
    width = sections[:,:,1]-sections[:,:,0]
    pairs &= (width>3.)&(width<20.)
    observed = pairs[:, (old.STATIONS>=4.)&(old.STATIONS<=24.)].sum(axis=1)>=6
    recorded = []
    for i,(row,offer,base) in enumerate(zip(group["rows"],group["offers"],group["base"],strict=True)):
        finals, states, values = {},{},{}
        for mode in MODES:
            value = accepted[mode].get(i)
            chosen = offer if value is not None and value>=thresholds[mode] else None
            values[mode] = value
            for method,proposal in (("candidate_only",chosen),("shared_edge",base or chosen)):
                name = f"{mode}_{method}"
                # A rejected turn proposal is explicitly uncertain. A supported
                # profile without an in-range candidate retains the straight output.
                observable = bool(proposal is not None or (offer is None and observed[i]))
                reason = "candidate_rejected" if value is not None else "boundary_or_motion_unavailable"
                published = old.publish(row,proposal,observable,reason)
                timelines[name][row["run"]].append(published)
                finals[name] = proposal
                states[name] = published["state"]
        for method in OUTPUTS:
            before, after = (compare_outcome(row,finals[f"{m}_{method}"]) for m in MODES)
            if before != after:
                transitions[method][row["truth"],row["label"],before,after] += 1
        recorded.append(dict(run=row["run"],frame=row["frame"],truth=row["truth"],label=row["label"],
                             truth_endpoints_m=row["truth_endpoints_m"],candidate=offer,
                             common_edge_warning=base,scores=values,final=finals,states=states))
    metrics = {k:runner.totals(v) for k,v in timelines.items()}
    detailed = {k:{tag:summarize(rows) for tag,rows in by.items()} for k,by in timelines.items()}
    return metrics,detailed,recorded,{method:[dict(truth=k[0],label=k[1],edge=k[2],t4t5=k[3],frames=n)
                                              for k,n in sorted(value.items())] for method,value in transitions.items()}


def comparable_budget(group,probabilities,cap):
    """Retrospective comparison only, never used as a replay threshold."""
    report = {}
    far = np.isin(group["errors"],OUTSIDE)
    right = np.array([group["rows"][i]["label"]=="2" for i in group["index"]])
    for mode,p in probabilities.items():
        choices = []
        for cut in np.r_[np.inf,np.unique(p)]:
            accept = p>=cut
            n = int((accept&far).sum())
            if n<=cap:
                choices.append(dict(outside=n,joint=int((accept&group["target"]).sum()),
                                    right_joint=int((accept&group["target"]&right).sum())))
        report[mode] = dict(best_right_joint=max(c["right_joint"] for c in choices),
                            best_joint=max(c["joint"] for c in choices))
    return report


def main():
    freeze, checkpoint = OUT/"training.json",OUT/"confirmation.joblib"
    training = old.read(freeze)
    assert training["excluded"]==[SILVER] and tuple(training["development"])==DEVELOP
    assert old.digest(checkpoint)==training["checkpoint_sha256"]
    for name,sha in training["source_sha256"].items():
        assert SILVER not in name and old.digest(ROOT/name)==sha
    sources = {Path(__file__),freeze,checkpoint,Path(data.__code__.co_filename),Path(scores.__code__.co_filename)}
    group = data(SILVER,sources)
    _,_,sections,_ = cached(SILVER,sources)
    models = joblib.load(checkpoint)
    assert models["edge"]["motion"] is None and models["t4t5"]["motion"] is not None
    predictions = {mode:scores(group,models[mode]) for mode in MODES}
    metrics,detail,records,paired = evaluate(group,predictions,training["threshold_by_model"],sections)
    assert len(records)==1200
    raw = sum(x is not None for x in group["offers"])
    raw_right = sum(r["truth"]=="approach" and r["label"]=="2" and r["candidate"]
                    is not None and r["candidate"][0]==2 for r in records)
    cap = min(metrics[f"{m}_candidate_only"]["outside"] for m in MODES)
    shape = old.read(SHAPE/"result.json")
    path = OUT/"silverstone-frames.jsonl"
    old.jsonl(path,records)
    report = dict(status="held_out_confirmation_replay",seed=training["seed"],source_revision=old.revision(),
                  frames=len(records),thresholds=training["threshold_by_model"],development=list(DEVELOP),
                  shape_raw_candidates=raw,paired_available=len(group["index"]),raw_right_candidates=raw_right,
                  metrics=metrics,detailed=detail,paired=paired,
                  equal_outside_diagnostic=dict(outside_cap=cap,result=comparable_budget(group,predictions,cap)),
                  historical={k:v for k,v in shape["metrics"].items() if k in
                              ("edge_base","t4t5_base","edge_shape_geometry","t4t5_shape_geometry",
                               "edge_far_wide","t4t5_far_wide")},
                  rows_sha256=old.digest(path),source_sha256={str(p):old.digest(p) for p in sorted(sources)},
                  termination="all_1200_frames_and_common_candidates_scored")
    old.write(OUT/"result.json",report)
    print(metrics,report["equal_outside_diagnostic"],flush=True)


if __name__ == "__main__":
    main()
