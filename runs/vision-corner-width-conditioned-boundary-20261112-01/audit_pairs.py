"""Report paired frame changes and focus the added width-derived boundary candidates."""
from collections import Counter
from width_conditioned import EVALUATION, METHODS, OUT, old, frames, reference, row_map


def category(label, forecast):
    side={"left":1,"right":2}.get(forecast["state"])
    return old.category(label,None if side is None else (side,forecast["distance_m"]))


def run():
    path=OUT/"result.json"
    result=old.read(path)
    changes={}
    for track,_ in EVALUATION:
        truth,_=reference(track)
        labels=row_map(truth)
        info=result["tracks"][track]
        summaries={}
        for mode,report in info["modes"].items():
            rows=frames(OUT/f"{track}-{mode}-frames.jsonl",report["trace_sha256"])
            pairs=Counter()
            witness=[]
            for r in rows:
                ref=labels[r["run"],r["frame"]]
                a,b=(category(ref,r["forecasts"][m]) for m in METHODS)
                if a!=b:
                    pairs[f"{a} -> {b}"]+=1
                    witness.append(dict(frame=r["frame"],candidate=r["candidate"],edge=a,t4t5=b))
            summaries[mode]=dict(paired=pairs,changes=witness)
        changes[track]=summaries
        print(track,{k:v["paired"] for k,v in summaries.items()},flush=True)
    old.write(OUT/"paired-audit.json",dict(tracks=changes,result_sha256=old.digest(path)))


if __name__=="__main__":
    run()
