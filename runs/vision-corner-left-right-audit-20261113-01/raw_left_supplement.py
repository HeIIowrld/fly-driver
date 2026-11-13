"""Test previously frozen structured confirmation as a left-only supplemental readout."""
from collections import Counter, defaultdict
from audit_and_refit import (CLI, EVAL, METHODS, OUT, TURNS, Path, SensorArrays, build, frames,
                             joblib, load_observer, np, old, reference, score_archive, turning)
from gate_inputs import scores
from replay_corner_boundary_states import phase_scores
import sys

SOURCE=OUT.parent/"vision-corner-selected-confirmation-20261112-01"
sys.path.insert(0,str(SOURCE))


def run():
    training=SOURCE/"training.json"
    checkpoint=SOURCE/"confirmers.joblib"
    locked=old.read(training)
    assert old.digest(checkpoint)==locked["checkpoint_sha256"]
    assert set(locked["excluded"])=={t for t,_ in EVAL}
    models=joblib.load(checkpoint)["structured"]
    observer,settings,_=load_observer()
    results={}
    for track,name in EVAL:
        root=CLI/("vallelunga-retry" if name=="vallelunga" else name)
        receipt=root/"result.json"
        source=root/"sensors.npz"
        assert old.digest(source)==old.read(receipt)["inputs_sha256"]
        with np.load(source,allow_pickle=False) as data:
            built=build(SensorArrays({k:data[k] for k in data.files}),settings)
        path=root/"predictions/forecasts.jsonl"
        base=frames(path,old.read(path.with_name("result.json"))["forecasts_sha256"])
        original=OUT.parent/"vision-corner-archive-observer-20261112-01"/name/"predictions/forecasts.jsonl"
        raw=frames(original,old.read(original.with_name("result.json"))["forecasts_sha256"])
        group=built["group"]
        interior=phase_scores(group,observer["phase"])[:,2]
        lookup={(int(i),a):j for j,(i,a) in enumerate(zip(group["index"],group["offers"],strict=True))}
        chosen=[]
        for i,r in enumerate(base):
            assert (r["run"],r["frame"])==(built["rows"][i]["run"],built["rows"][i]["frame"])
            assert (raw[i]["run"],raw[i]["frame"])==(r["run"],r["frame"])
            candidate=raw[i]["candidate"]
            if candidate is not None and candidate[0]==1 and not turning(r) and interior[i]<=settings["phase_cut"]:
                chosen.append((i,lookup[(i,tuple(candidate))]))
        p=scores(group,np.array([j for _,j in chosen]),models)
        output=list(base)
        for k,(i,_) in enumerate(chosen):
            forecasts=dict(base[i]["forecasts"])
            for m,kind in zip(METHODS,("temporal","t4t5"),strict=True):
                if p[kind][k]>=locked["cuts"]["structured"][kind]:
                    forecasts[m]=dict(state="left",distance_m=raw[i]["candidate"][1])
            if forecasts!=base[i]["forecasts"]:
                output[i]=dict(base[i],candidate=raw[i]["candidate"],forecasts=forecasts)
        truth,label_path=reference(track)
        changes=[]
        distances={m:defaultdict(list) for m in METHODS}
        by_key={(r["run"],r["frame"]):r for r in truth}
        for a,b in zip(base,output,strict=True):
            if turning(a):
                assert a==b
            ref=by_key.get((a["run"],a["frame"]))
            if a!=b and ref is not None:
                changes.append(dict(run=a["run"],frame=a["frame"],truth=ref["truth"],side=ref["label"],
                    actual_m=ref["truth_endpoints_m"],candidate=b["candidate"],forecasts=b["forecasts"]))
            if ref is None or ref["truth"]!="approach":
                continue
            for m,f in b["forecasts"].items():
                if TURNS.get(f["state"])==int(ref["label"]):
                    distances[m][ref["label"]].append(max(abs(f["distance_m"]-x) for x in ref["truth_endpoints_m"]))
        index=lambda rows:{(a["run"],a["frame"]):a for a in rows}
        score={name:{m:score_archive(truth,index(rows),m) for m in METHODS}
                for name,rows in (("before",base),("after",output))}
        assert all(score["before"][m].get("hit_2",0)==score["after"][m].get("hit_2",0) for m in METHODS)
        trace=OUT/f"{track}-raw-left-supplement.jsonl"
        old.jsonl(trace,output)
        result=dict(scores=score,distances={m:{s:dict(n=len(v),mae_m=float(np.mean(v))) for s,v in by.items()}
                                          for m,by in distances.items()},changes=changes,
                    forecasts_sha256=old.digest(trace),
                    source_sha256={str(p):old.digest(p) for p in (path,source,receipt,label_path,original)})
        results[track]=result
        print(track,score,result["distances"],len(changes),flush=True)
    old.write(OUT/"raw-left-supplement-result.json",dict(tracks=results,
        source_sha256={str(p):old.digest(p) for p in (Path(__file__),training,checkpoint,Path(scores.__code__.co_filename))},
        limitation="retrospective_routing_with_previously_frozen_weights_and_thresholds",termination="archive_end"))


if __name__=="__main__":
    run()
