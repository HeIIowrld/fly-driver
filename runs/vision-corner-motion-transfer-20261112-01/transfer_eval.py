"""Select local confirmation thresholds on other circuits, then score a held-out circuit."""
from functools import lru_cache
from pathlib import Path

import numpy as np

from baseline import publish
from confirm import RECOVERY,signals
from evaluate_fused_corner import frame_key,read_rows
from paired_motion import inputs
import profile_replay as profile
from score_corner_captures import summarize
from train_vision_teacher_line import digest,jsonl,read,write

OUT=Path(__file__).parent
DEVELOPMENT=Path("runs/vision-corner-motion-confirmation-20261112-01")
TRACKS=tuple(profile.CALIBRATION_TRACKS)
MODES=("boundaries","boundaries_temporal_t4t5")
CUTS=(.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.70,.80)
BUDGETS=(0,1,3)
FALSE=("straight","out_of_range")


def load_proposals(hashes):
    result=DEVELOPMENT/"development-bracket.json"
    rows=DEVELOPMENT/"development-bracket-proposals.jsonl"
    assert digest(rows)==read(result)["rows_sha256"]
    hashes.update((result,rows))
    grouped={track:[] for track in TRACKS}
    for row in read_rows(rows):
        grouped[row["track"]].append(row)
    return grouped


def counts(base,proposals,accepted):
    totals=dict(base)
    for row,accept in zip(proposals,accepted,strict=True):
        if not accept:
            continue
        if row["direction_hit"]:
            totals[row["side"]]+=1
        totals["joint"]+=int(row["joint_hit"])
        totals["outside"]+=int(row["truth"] in FALSE)
    return totals


def approach_report(group,saved,accepted):
    by_run={}
    for row in group["rows"]:
        if row["truth"]!="approach":
            continue
        key=frame_key(row)
        original=saved[key]["methods"]["baseline"]
        fallback=accepted.get(key)
        chosen=fallback if fallback is not None else original
        side=chosen.get("side",chosen.get("state"))
        proposal=(1 if side=="left" else 2,chosen["distance_m"]) if side in ("left","right") else None
        observable=bool(proposal) or original["observable"]
        by_run.setdefault(key[0],[]).append(publish(row,proposal,observable))
    events=[]
    pairs=[]
    for run,rows in by_run.items():
        r=summarize(rows)
        events.extend(dict(run=run,**event) for event in r["events"] if event["first_stable_frame"] is not None)
        pairs.extend(dict(run=run,**pair) for pair in r["distance_change"]["rows"])
    return dict(stable=events,stable_corners=len({e["physical_corner"] for e in events}),direction_pairs=len(pairs),
                distance_change_mae_m=float(np.mean([abs(p["delta_error_m"]) for p in pairs])) if pairs else None,
                increasing_pairs=sum(p["truth_delta_m"]<0 and p["forecast_delta_m"]>0 for p in pairs))


def main():
    hashes={Path(__file__),Path(inputs.__code__.co_filename),Path(profile.__file__),
            Path(signals.__code__.co_filename),Path("scripts/score_corner_captures.py")}
    proposals=load_proposals(hashes)
    group={track:inputs(track,hashes) for track in TRACKS}
    saved={}
    baselines={}
    for track in TRACKS:
        r=DEVELOPMENT/f"{track}-development.json"
        f=DEVELOPMENT/f"{track}-frames.jsonl"
        receipt=read(r)
        assert digest(f)==receipt["rows_sha256"]
        hashes.update((r,f))
        saved[track]={frame_key(row):row for row in read_rows(f)}
        baselines[track]=receipt["methods"]["baseline"]["score"]["metrics"]["candidate"]
        assert all(saved[track][frame_key(p)]["methods"]["baseline"]["state"]=="abstain" for p in proposals[track])
        print("Loaded",track,len(proposals[track]),flush=True)

    @lru_cache
    def model(mode,training):
        return profile.fit([group[t][mode] for t in training],1)

    @lru_cache
    def scores(mode,track,training):
        assert track not in training
        g=group[track][mode]
        prediction=profile.predict(g,model(mode,training))
        values={frame_key(r):p for r,p in zip(g["selected"],prediction,strict=True)}
        result=[]
        for row in proposals[track]:
            side=1 if row["side"]=="left" else 2
            local=signals(values.get(frame_key(row)),side,row["distance_m"])["local"]
            result.append(local if local is not None else -np.inf)
        return np.array(result)

    results={}
    details=[]
    for track in TRACKS:
        train=tuple(t for t in TRACKS if t!=track)
        nested={t:tuple(k for k in train if k!=t) for t in train}
        trials={}
        for mode in MODES:
            validation={t:scores(mode,t,peers) for t,peers in nested.items()}
            evaluated=scores(mode,track,train)
            def stats(t,values,cut):
                return counts(baselines[t],proposals[t],values>=cut if cut is not None else np.zeros(len(values),bool))
            grid={cut:{t:stats(t,v,cut) for t,v in validation.items()} for cut in CUTS}
            choices={}
            for budget in BUDGETS:
                options=[]
                for cut,table in grid.items():
                    diff={t:{key:s[key]-baselines[t][key] for key in ("left","right","joint","outside")}
                          for t,s in table.items()}
                    if any(d["outside"]>budget for d in diff.values()):
                        continue
                    gain={k:sum(d[k] for d in diff.values()) for k in ("left","right","joint","outside")}
                    if gain["joint"] or gain["left"] or gain["right"]:
                        options.append(((gain["joint"],gain["right"],gain["left"],-gain["outside"],cut),cut))
                selected=max(options)[1] if options else None
                mask=evaluated>=selected if selected is not None else np.zeros(len(evaluated),bool)
                accepted={frame_key(r):r for r,yes in zip(proposals[track],mask,strict=True) if yes}
                choices[str(budget)]=dict(threshold=selected,counts=stats(track,evaluated,selected),
                    approach=approach_report(group[track][mode],saved[track],accepted))
                for r,v,yes in zip(proposals[track],evaluated,mask,strict=True):
                    details.append(dict(track=track,run=r["run"],frame=r["frame"],mode=mode,budget=budget,
                        truth=r["truth"],side=r["side"],direction_hit=r["direction_hit"],joint_hit=r["joint_hit"],
                        signal=float(v) if np.isfinite(v) else None,accepted=bool(yes)))
            trials[mode]=dict(choices=choices,fixed025=stats(track,evaluated,.25))
        results[track]=dict(training=train,nested=nested,baseline=baselines[track],
            baseline_approach=approach_report(group[track][MODES[0]],saved[track],{}),modes=trials)
        print(track,{m:{b:v["threshold"] for b,v in x["choices"].items()} for m,x in trials.items()},flush=True)
    path=OUT/"proposal-decisions.jsonl"
    jsonl(path,details)
    write(OUT/"result.json",dict(status="evaluated",seed=None,tracks=results,rows_sha256=digest(path),
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__=="__main__":
    main()
