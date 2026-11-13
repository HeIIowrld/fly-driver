"""Select between two saved path horizons using only available forecasts or path geometry."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from train_vision_teacher_line import digest,jsonl,read,write

out=Path(__file__).parent
long=out.parent/"vision-monza-guided-long-horizon-20261111-01/rows.jsonl"
forecast=out/"forecasts.jsonl"
def read_rows(path,receipt):
    assert digest(path)==read(receipt)["rows_sha256"]
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
rows=read_rows(long,long.with_name("result.json"))
predictions=read_rows(forecast,forecast.with_name("forecast-result.json"))
alerts={(r["run"],r["frame"]):r["alert"] for r in predictions if r["available"]}
short={(r["run"],r["frame"]):r for r in rows if r["policy"]=="short"}
extended={(r["run"],r["frame"]):r for r in rows if r["policy"]=="extended"}
assert short.keys()==extended.keys() and len(short)==1318
assert { (r["run"],r["frame"]) for r in predictions }==short.keys()
results=[]
for key,base in short.items():
    other=extended[key]
    a,b=base["line"],other["line"]
    extension=bool(b and b["x_m"][-1]>20)
    difference=None
    if extension and a:
        x,y=np.asarray(a["x_m"]),np.asarray(a["y_m"])
        difference=float(np.max(abs(y-np.interp(x,b["x_m"],b["y_m"]))))
    prefer_alert=extension and alerts.get(key,False)
    prefer_prefix=difference is not None and difference<=.5
    for name,selected in (("short",base),("extended",other),("alert_gate",other if prefer_alert else base),
                         ("prefix_gate",other if prefer_prefix else base),
                         ("both",other if prefer_alert and prefer_prefix else base)):
        results.append(dict(selected,policy=name,forecast_available=key in alerts,
            selected_extension=selected is other and extension,prefix_difference_m=difference))

summary={}
for name in ("short","extended","alert_gate","prefix_gate","both"):
    subset=[r for r in results if r["policy"]==name]
    counts=Counter()
    for r in subset:
        margin=r["map_margin_m"]
        counts["no_line" if margin is None else "inside" if margin>=0. else "outside"]+=1
        counts["long_lines"]+=r["selected_extension"]
        counts["validation_long_lines"]+=r["selected_extension"] and r["split"]=="validation"
    cases=[dict(run=r["run"],frame=r["frame"],extended=r["selected_extension"],difference=r["prefix_difference_m"])
           for r in subset if (r["run"],r["frame"]) in
           {("monza-user-started-teacher-01",1056),("fresh-monza-develop-01",1058)}]
    summary[name]=dict(counts,previous_validation_examples=cases)
jsonl(out/"rows.jsonl",results)
write(out/"result.json",dict(summary=summary,seed=None,termination="archive_end",
    revision=read(long.with_name("result.json"))["revision"], n=len(short),
    prefixes_compared_in_body_frame=True, map_used_for_evaluation_only=True,
    saved_line_candidates=True, available_forecasts=len(alerts),
    hashes={str(p):digest(p) for p in (long,forecast,Path(__file__))},rows_sha256=digest(out/"rows.jsonl")))
print(summary)
