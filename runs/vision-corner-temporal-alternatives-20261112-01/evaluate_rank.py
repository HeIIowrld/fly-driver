"""Replay temporal ranking and isolate T4/T5 confirmation on an identical selection."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rank import OUT, METHODS, SILVER, VAL, choose, data, old, joblib
from evaluate_shape import ROOT
from score_corner_captures import summarize

POLICIES = ("candidate_only","shared_edge")


def unpack(value):
    return (int(value[0]),float(value[1])) if np.isfinite(value).all() else None


def group_by_frame(group):
    result = defaultdict(list)
    for j,i in enumerate(group["index"]):
        result[int(i)].append(j)
    return result


def is_right(row):
    return row["truth"]=="approach" and row["label"]=="2"


def available_count(options, row):
    signs = [p for p in options if str(p[0])==row["label"]]
    return bool(signs),any(old.category(row,p)=="joint_hit" for p in signs)


def analyze(group,picks,cuts):
    entry = old.context()
    runner = entry.import_replay()
    matched = {m:{frame:(candidate,score) for frame,candidate,score in picks[m]} for m in METHODS}
    assert matched["temporal"].keys()==matched["t4t5_confirm"].keys()
    assert all(matched["temporal"][i][0]==matched["t4t5_confirm"][i][0] for i in matched["temporal"])
    by_frame = group_by_frame(group)
    timelines = {f"{method}_{policy}":defaultdict(list) for method in METHODS for policy in POLICIES}
    paired = {p:Counter() for p in POLICIES}
    raw = Counter()
    pre_cut = {m:Counter() for m in METHODS}
    records = []
    for i,row in enumerate(group["rows"]):
        observable = np.isfinite(group["sections"][i]).all(axis=1).sum()>=6
        options = group["pools"][i]
        indices = by_frame[i]
        primary = unpack(group["primary"][i])
        base = unpack(group["edge_base"][i])
        if is_right(row):
            raw["right_frames"] += 1
            for label, subset in (("first",[primary] if primary else []),("pool",options)):
                direction,joint = available_count(subset,row)
                raw[f"{label}_right_direction"] += int(direction)
                raw[f"{label}_right_joint"] += int(joint)
        scored, finals, states = {},{},{}
        for method in METHODS:
            j, score = matched[method].get(i,(None,None))
            selected = group["offers"][j] if j is not None else None
            accepted = selected if score is not None and score>=cuts[method] else None
            scored[method] = dict(candidate=selected,score=score,accepted=accepted)
            if is_right(row):
                direction,joint = available_count([selected] if selected else [],row)
                pre_cut[method]["direction"] += int(direction)
                pre_cut[method]["joint"] += int(joint)
            for policy,forecast in (("candidate_only",accepted),("shared_edge",base or accepted)):
                name = f"{method}_{policy}"
                available = forecast is not None or (observable and not options)
                published = old.publish(row,forecast,available,"candidate_rejected_or_unavailable")
                timelines[name][row["run"]].append(published)
                finals[name],states[name] = forecast,published["state"]
        for policy in POLICIES:
            first,second = (old.category(row,finals[f"{m}_{policy}"]) for m in ("temporal","t4t5_confirm"))
            if first!=second:
                paired[policy][row["truth"],row["label"],first,second] += 1
        records.append(dict(run=row["run"],frame=row["frame"],truth=row["truth"],label=row["label"],
                            truth_endpoints_m=row["truth_endpoints_m"],primary=primary,pool=options,
                            history=[dict(candidate=group["offers"][j],lags=group["history"][j].tolist())
                                     for j in indices],
                            common_edge=base,scores=scored,final=finals,states=states))
    report = dict(raw_right=dict(raw),pre_cut_right={m:dict(c) for m,c in pre_cut.items()},
                  metrics={m:runner.totals(v) for m,v in timelines.items()},
                  detailed={m:{r:summarize(rows) for r,rows in v.items()} for m,v in timelines.items()},
                  paired={p:[dict(truth=k[0],label=k[1],edge=k[2],t4t5=k[3],frames=n)
                             for k,n in sorted(c.items())] for p,c in paired.items()})
    return report,records


def main():
    checkpoint, manifest = OUT/"rankers.joblib",OUT/"training.json"
    frozen = old.read(manifest)
    assert frozen["excluded"]==[SILVER,VAL] and old.digest(checkpoint)==frozen["checkpoint_sha256"]
    for name,sha in frozen["source_sha256"].items():
        assert SILVER not in name and VAL not in name and old.digest(ROOT/name)==sha,name
    models = joblib.load(checkpoint)
    assert models["static"]["motion"] is None and models["temporal"]["motion"] is None and models["t4t5"]["motion"] is not None
    result = {}
    for track in (SILVER,VAL):
        hashes = {Path(__file__),checkpoint,manifest,Path(data.__code__.co_filename),Path(choose.__code__.co_filename)}
        group = data(track,hashes)
        picks = choose(group,models)
        report, records = analyze(group,picks,frozen["cuts"])
        assert len(records)==1200
        path = OUT/f"{track}-frames.jsonl"
        old.jsonl(path,records)
        report.update(frames=len(records),rows_sha256=old.digest(path),
                      source_sha256={str(p):old.digest(p) for p in sorted(hashes)})
        result[track] = report
        print(track,report["raw_right"],report["pre_cut_right"],report["metrics"],flush=True)
    old.write(OUT/"result.json",dict(status="two_frozen_replays",seed=frozen["seed"],source_revision=old.revision(),
               cuts=frozen["cuts"], tracks=result,training_sha256=old.digest(manifest),
               termination="two_excluded_circuits_scored_with_identical_confirm_selection"))


if __name__ == "__main__":
    main()
