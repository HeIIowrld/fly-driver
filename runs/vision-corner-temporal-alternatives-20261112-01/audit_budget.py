"""Compare score ordering under the same retrospective false-alert limits."""
from collections import Counter
from hashlib import file_digest
from pathlib import Path
import json

OUT = Path(__file__).resolve().parent
TRACKS = ("ks_silverstone1967","ks_vallelunga")
METHODS = ("static","temporal","t4t5_confirm","t4t5_rerank")


def digest(path):
    with path.open("rb") as file:
        return file_digest(file,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def classify(row, offer):
    if row["truth"] in ("straight","out_of_range"):
        return "outside"
    if row["truth"]=="in_corner":
        return "in_corner"
    if row["truth"]!="approach":
        return "other"
    if str(offer[0])!=row["label"]:
        return "wrong_direction"
    if max(abs(offer[1]-v) for v in row["truth_endpoints_m"])>5.:
        return "distance_error"
    return "right_joint" if offer[0]==2 else "left_joint"


def ordering(rows,mode):
    buckets = {}
    for r in rows:
        value = r["scores"][mode]
        if value["score"] is not None:
            buckets.setdefault(value["score"],Counter())[classify(r,value["candidate"])] += 1
    counts = Counter()
    timeline = [dict(outside=0,in_corner=0,right=0,joint=0)]
    for value in sorted(buckets,reverse=True):
        counts.update(buckets[value])
        timeline.append(dict(outside=counts["outside"],in_corner=counts["in_corner"],
                             right=counts["right_joint"],joint=counts["right_joint"]+counts["left_joint"]))
    return timeline


def main():
    reports = read(OUT/"result.json")["tracks"]
    stats = {}
    for track in TRACKS:
        path = OUT/f"{track}-frames.jsonl"
        assert digest(path)==reports[track]["rows_sha256"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
        negative = sum(r["truth"] in ("straight","out_of_range") for r in rows)
        modes = {}
        for mode in METHODS:
            scores = ordering(rows,mode)
            modes[mode] = {}
            for cap in (int(.025*negative),int(.05*negative),int(.10*negative)):
                possible = [p for p in scores if p["outside"]<=cap]
                top = max(possible,key=lambda p:(p["joint"],p["right"],-p["in_corner"],-p["outside"]))
                modes[mode][cap] = top
        stats[track] = dict(outside_denominator=negative,matched_caps=modes)
    report = dict(status="retrospective_only",seed=None,source_revision=read(OUT/"result.json")["source_revision"],
                  tracks=stats,result_sha256=digest(OUT/"result.json"),
                  program_sha256=digest(Path(__file__)),termination="matched_caps_scored")
    (OUT/"equal-outside-audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
