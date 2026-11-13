"""Measure uninterrupted availability of bounded local lines on archived teacher laps."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from flybrain_drive.ac_camera_archive import CameraArchive
from train_next_corner import BASE, MONZA
from train_vision_teacher_line import digest, jsonl, read, write

out=Path(__file__).parent
source=out/"rows.jsonl"
proof=read(source.with_name("result.json"))
assert digest(source)==proof["rows_sha256"]
clocks={(r["run"],r["frame"]):(r["received"],r["generation"]) for r in map(json.loads,source.read_text(encoding="utf8").splitlines())}
groups=defaultdict(list)
for row in map(json.loads,source.read_text(encoding="utf8").splitlines()):
    if row["policy"] in ("short","extended"):
        groups[row["policy"],row["run"],row["event"]].append(row)
phases={"approach","early_turn","mid_turn","late_turn","exit"}
rows=[]
for (policy,run,event),group in groups.items():
    group.sort(key=lambda row:row["frame"])
    available_phases={row["phase"] for row in group}
    if available_phases!=phases:
        continue
    chains=[]
    current=[]
    failures=Counter()
    for row in group:
        frame=row["frame"]
        time,generation=clocks[run,frame]
        if current:
            prev_time,prev_gen=clocks[run,current[-1]["frame"]]
            if frame!=current[-1]["frame"]+1 or generation!=prev_gen or not 0<time-prev_time<=.18:
                chains.append(current)
                current=[]
        if row["map_margin_m"] is None or row["map_margin_m"]<0:
            failures[row["status"] if row["map_margin_m"] is None else "outside_map"]+=1
            if current:
                chains.append(current)
                current=[]
        else:
            current.append(row)
    if current:
        chains.append(current)
    def duration(chain):
        return clocks[run,chain[-1]["frame"]][0]-clocks[run,chain[0]["frame"]][0]
    rows.append(dict(policy=policy,run=run,event=event,frames=len(group),failures=dict(failures),
        all_phases_uninterrupted=any({r["phase"] for r in chain}==phases for chain in chains),
        longest_bounded_s=max(map(duration,chains),default=0.),
        longest_bounded_station_span_m=max((c[-1]["station_relative_m"]-c[0]["station_relative_m"]
                                             for c in chains),default=0.),
        bounded_chains=len(chains)))
summary={}
for policy in ("short","extended"):
    summary[policy]={}
    for run,_ in MONZA:
        subset=[r for r in rows if r["policy"]==policy and r["run"]==run]
        counts=Counter()
        for r in subset:
            counts.update(r["failures"])
        summary[policy][run]=dict(events=len(subset),uninterrupted=sum(r["all_phases_uninterrupted"] for r in subset),
            best_chain_s=max(r["longest_bounded_s"] for r in subset),failure_frames=dict(counts))
jsonl(out/"continuity-rows.jsonl",rows)
write(out/"continuity.json",dict(summary=summary,seed=None,termination="archive_end",
    capture_generation_rechecked=True, revision=proof["revision"],
    hashes={str(p):digest(p) for p in (source,Path(__file__))},rows_sha256=digest(out/"continuity-rows.jsonl")))
print(summary)
