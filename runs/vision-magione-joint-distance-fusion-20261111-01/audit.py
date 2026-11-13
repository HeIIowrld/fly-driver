"""Keep the frozen alert while testing the joint readout as a distance estimate."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean

from train_vision_teacher_line import digest,jsonl,read,write

out=Path(__file__).parent
joint=out.parent/"vision-monza-joint-corner-readout-20261111-01/predictions.jsonl"
geometry=out.parent/"vision-magione-corner-geometry-front-far-20261111-01/rows.jsonl"
crossing=out.parent/"vision-magione-line-width-onset-crossing-20261110-01/rows.jsonl"
def load(path):
    assert digest(path)==read(path.with_name("result.json"))["rows_sha256"]
    return {(r["run"],r["frame"]):r for l in path.read_text(encoding="utf8").splitlines() if (r:=json.loads(l))}
a,b,c=map(load,(joint,geometry,crossing))
assert a.keys()==b.keys() and len(a)==1195
rows=[]
for key,r in a.items():
    frozen=r["predictions"]["frozen"]
    policies={"frozen":frozen}
    for name in ("one_lap","two_laps"):
        alternative=r["predictions"][name]
        agrees=alternative["alert"] and alternative["direction"]==frozen["direction"]
        policies[name]=dict(frozen,distance_m=alternative["distance_m"]) if frozen["alert"] and agrees else frozen
    crossed=key in c and c[key]["options"]["base"]["onset_crossing_offset_m"] is not None
    rows.append(dict(run=key[0],frame=key[1],near=b[key]["near"],truth=b[key]["label"],
                     crossed=crossed,policies=policies))
summary={}
for name in rows[0]["policies"]:
    counts=Counter()
    errors=defaultdict(list)
    for r in rows:
        pred=r["policies"][name]
        if not pred["alert"]:
            continue
        if not r["near"]:
            counts["out_of_band"]+=1
            continue
        truth=r["truth"]
        if pred["direction"]!=truth["direction"]:
            continue
        d=truth["distance"]
        error=abs(pred["distance_m"]-d)
        band="4-8" if d<8 else "8-12" if d<12 else "12-20"
        errors[band].append(error)
        counts["distance_updates"]+=pred["distance_m"]!=r["policies"]["frozen"]["distance_m"]
        if error<=5:
            counts["direction_distance_hit"]+=1
            counts["with_line_crossing"]+=r["crossed"]
            counts[f"distance_hit_{band}"]+=1
            counts[f"line_crossing_{band}"]+=r["crossed"]
    summary[name]=dict(counts,mae_m={key:mean(err) for key,err in errors.items()})
jsonl(out/"rows.jsonl",rows)
write(out/"result.json",dict(summary=summary,n=len(rows),seed=None,termination="archive_end",
    revision=read(joint.with_name("result.json"))["revision"], hashes={str(p):digest(p) for p in
        (joint,geometry,crossing,Path(__file__))},rows_sha256=digest(out/"rows.jsonl")))
print(summary)
