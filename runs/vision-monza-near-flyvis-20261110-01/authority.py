from collections import Counter
from pathlib import Path
import json

import numpy as np
from flybrain_drive.local_racing_line import plan_local_line, speed_bound_with_corner_preview
from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
root = Path("runs")
replay = root/"vision-magione-memory-command-replay-20261110-02"
receipt = read(replay/"result.json")
memory = root/"vision-magione-boundary-memory-20261110-01"
pixel = root/"vision-magione-pixel-boundary-20261110-02"
forecast = out/"augmented-predictions.jsonl"
assert digest(forecast)==read(out/"readout-result.json")["models"]["augmented"]["predictions_sha256"]
agreement = read(memory/"result.json")["current_edge_agreement_gate"]["gate_max_disagreement_m"]
assert agreement == .75

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines()]

prediction = {(r["run"],r["frame"]):r for r in rows(forecast) if r["split"]=="external"}
limits=dict(clearance_m=1.1,max_curvature=.15,lateral_accel_mps2=4.,braking_mps2=4.)
caps=(25,30,35)
records=[]
for name,summary in receipt["summaries"].items():
    direct_path=pixel/f"{name}.jsonl"
    memory_path=memory/f"{name}.jsonl"
    trace_path=replay/f"{name}.jsonl"
    for path in (direct_path,memory_path):
        assert digest(path)==summary["input_sha256"][str(path)]
    assert digest(trace_path)==summary["trace_sha256"]
    trace=rows(trace_path)
    assert len(trace)==summary["replayed_ticks"]
    by_capture={}
    for r in trace:
        by_capture.setdefault(r["capture"],[]).append(r)
    direct={r["frame"]:r for r in rows(direct_path)}
    prior=None
    for remembered in rows(memory_path):
        frame=remembered["frame"]
        pred=prediction[name,frame]
        assert abs(remembered["t"]-pred["time"])<1e-6
        distance=pred["predicted_distance_m"]
        current_prediction=(frame,pred["time"],pred["predicted"],distance)
        stable=(distance is not None and prior is not None and frame==prior[0]+1
                and 0<pred["time"]-prior[1]<=.2 and pred["predicted"]==prior[2]
                and prior[3] is not None and distance<=prior[3]+.5)
        prior=current_prediction
        if frame not in by_capture:
            continue
        sections=[]
        for observed,carried in zip(direct[frame]["sections"],remembered["sections"],strict=True):
            partial=[k for k in ("left_y_m","right_y_m") if observed[k] is not None]
            use=(carried["source"]=="carried" and bool(partial)
                 and max(abs(observed[k]-carried[k]) for k in partial)<=agreement)
            sections.append(carried if use else observed)
        truth=("none","left","right")[pred["direction"]]
        truth_near=truth!="none" and pred["distance"]<=30.
        label=("false_near_alert" if not truth_near else "wrong_direction" if truth!=pred["predicted"]
               else "correct_within_5m" if distance is not None and abs(distance-pred["distance"])<=5.
               else "correct_direction_distance_error")
        for cap in caps:
            line,_=plan_local_line(sections,0.,speed_ceiling_mps=cap/3.6,**limits)
            assert line is not None
            bound=np.asarray(line["speed_bound_mps"])
            cap_with_preview=speed_bound_with_corner_preview(line,distance,
                corner_speed_mps=20/3.6,braking_mps2=4.)
            oracle_bound=speed_bound_with_corner_preview(line,pred["distance"] if truth!="none" else None,
                corner_speed_mps=20/3.6,braking_mps2=4.)
            records.append(dict(run=name,capture=frame,cap_kmh=cap,label=label,stable=stable,
                predicted_distance_m=distance,truth_distance_m=pred["distance"],
                entry_reduction_mps=float(bound[0]-cap_with_preview[0]),
                oracle_entry_reduction_mps=float(bound[0]-oracle_bound[0]),
                oracle_downstream_reduction_mps=float(np.max(bound-oracle_bound)),
                downstream_reduction_mps=float(np.max(bound-cap_with_preview)),
                observed_overspeed_ticks=int(sum(r["body_speed_mps"]>cap_with_preview[0]+.5
                                             and r["body_speed_mps"]<=bound[0]+.5 for r in by_capture[frame])),
                replay_ticks=len(by_capture[frame])))
summaries=[]
for cap in caps:
    all_rows=[r for r in records if r["cap_kmh"]==cap]
    for gate in (False,True):
        eligible=[r for r in all_rows if r["stable"] or not gate]
        entry=[r for r in eligible if r["entry_reduction_mps"]>1e-6]
        summaries.append(dict(cap_kmh=cap,stable_gate=gate,frames=len(all_rows),
            eligible=len(eligible),entry_n=len(entry),labels=dict(Counter(r["label"] for r in entry)),
            downstream_n=sum(r["downstream_reduction_mps"]>1e-6 for r in eligible),
            oracle_entry_n=sum(r["oracle_entry_reduction_mps"]>1e-6 for r in eligible),
            oracle_downstream_n=sum(r["oracle_downstream_reduction_mps"]>1e-6 for r in eligible),
            minimum_predicted_m=min((r["predicted_distance_m"] for r in eligible if r["predicted_distance_m"] is not None),default=None),
            minimum_truth_m=min((r["truth_distance_m"] for r in eligible if r["truth_distance_m"] is not None),default=None),
            induced_overspeed_ticks=sum(r["observed_overspeed_ticks"] for r in entry),
            max_entry_reduction_mps=max((r["entry_reduction_mps"] for r in entry),default=0.)))
jsonl(out/"authority-rows.jsonl",records)
write(out/"authority-result.json",dict(revision=receipt["revision"],seed=None,
    source_receipt_sha256=digest(replay/"result.json"),forecast_sha256=digest(forecast),
    program_sha256=digest(Path(__file__)),rows_sha256=digest(out/"authority-rows.jsonl"),
    summaries=summaries,scope="fixed archived paths and speed observations, counterfactual target envelopes",
    termination="all replayed capture groups compared"))
print(json.dumps(summaries))
