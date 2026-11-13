import json
from collections import Counter
from pathlib import Path
from train_vision_teacher_line import digest,read,write

out=Path(__file__).parent
sources={"multi_view":out.parent/"vision-magione-corner-geometry-dense-20261110-01/rows.jsonl",
         "front_far":out/"rows.jsonl"}
summary={}
for name,p in sources.items():
    assert digest(p)==read(p.with_name("result.json"))["rows_sha256"]
    rows=[json.loads(s) for s in p.read_text(encoding="utf8").splitlines()]
    summary[name]={}
    for threshold in (.005,.012,.025):
        c=Counter()
        for row in rows:
            hit=next(((x,angle) for x,angle in row["turns"] if x<=20 and abs(angle)>=threshold),None)
            label=row["label"]
            if not hit:
                continue
            if not row["near"]:
                c["out_of_band"]+=1
                if label and (not label["direction"] or label["distance"]>30):
                    c["beyond_30_or_no_labelled_corner"]+=1
                continue
            x,angle=hit
            if (1 if angle>0 else 2)==label["direction"] and abs(x-label["distance"])<=5:
                c["near_direction_distance_hit"]+=1
                if label["distance"]>=12:
                    c["12_to_20_direction_distance_hit"]+=1
        summary[name][str(threshold)]=dict(c)
write(out/"comparison.json",dict(summary=summary,hashes={str(p):digest(p) for p in (*sources.values(),Path(__file__))},
    n=1195,near_n=152,near_12_to_20_n=80,seed=None,termination="archive_end"))
print(summary)
