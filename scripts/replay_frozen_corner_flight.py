"""Run frozen archive maneuvers and same-episode body data through the flight circuit."""
import argparse
from collections import Counter
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter, ExperimentalBodyFeedbackCircuit
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder, decode_native_extension
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from plan_frozen_corner_archive import ArchivedMotion, MAX_BODY_AGE_S
from train_vision_teacher_line import digest, jsonl, read, write


def replay(archives, native, lines, output, native_only=False, interval_limit=None, device="cuda", command_source=None, run_limit=None, record_first_response=False, lag_one_frame=False):
    if interval_limit is not None and interval_limit < 1:
        raise ValueError("Interval limit must be positive")
    contract_file = Path("runs/thermal-driving-20261103/extension-data-01/contract.json")
    port_file = Path("runs/thermal-driving-20261103/native-fit-02/body-extension.pt")
    spec = Path("configs/flight-circuit-separated-v5-20261029.json")
    checkpoint = Path("runs/flight-separated-command-probe-20261029-t1081/checkpoint.pt")
    base_file = Path(os.environ["LOCALAPPDATA"])/"flybrain/runtime-runs/20261101-speed-wheel-turn-control-fit/speed-body-adapter.pt"
    model_receipt = Path("runs/vision-line-native-port-20261110-01/result.json")
    earlier = read(model_receipt)
    for path in (spec,checkpoint,base_file,port_file):
        if earlier["hashes"][str(path)] != digest(path):
            raise ValueError("Changed frozen flight circuit input")
    line_proof = read(lines.with_name("result.json"))
    if (digest(lines) != line_proof["rows_sha256"]
            or any(digest(p) not in line_proof["inputs_sha256"].values() for p in (archives,native))):
        raise ValueError("Maneuvers and source archives do not match")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    backend = "scipy_csr_compact_cpu" if device == "cpu" else "cached_csr_gpu"
    motor = PhysicalManeuverCircuit(spec,torch.load(checkpoint,map_location="cpu",weights_only=True),
                                    sparse_backend=backend).to(device)
    graph = ExperimentalBodyFeedbackCircuit(motor)
    graph.adapter = CenteredSpeedBodyFeedbackAdapter(len(motor.backbone.body_ports)).to(device)
    graph.adapter.load_state_dict(torch.load(base_file,map_location="cpu",weights_only=True)["model"])
    payload = torch.load(port_file,map_location="cpu",weights_only=True)
    assert digest(contract_file)==payload["extension_contract_sha256"]
    w = payload["body_port"]
    port = MultisensoryBodyFeedbackPort(w["reference"],w["scale"],len(motor.backbone.body_ports)).to(device)
    port.load_state_dict(w)
    graph.eval().requires_grad_(False)
    port.eval().requires_grad_(False)

    names = ("native","yaw_neutral","tilt_neutral","load_neutral","temperature_neutral")
    if native_only:
        names = names[:1]
    goals = torch.tensor(racing_goal(90,len(names)),device=device)
    kwargs = dict(command_contract=MANEUVER_CONTRACT,goal_contract=RACING_GOAL_CONTRACT)
    holder = {}
    hook = graph.adapter.register_forward_hook(lambda module,args,result: result+holder["delta"])
    folders = {name:Path(path) for name,path in read(archives).items()}
    episodes = {name:Path(path) for name,path in read(native).items()}
    commands = {(r["run"],r["frame"]):r for r in map(json.loads,lines.read_text(encoding="utf8").splitlines())}
    trace,counts = [],Counter()
    inputs = [archives,native,lines,model_receipt,spec,checkpoint,base_file,contract_file,port_file,Path(__file__)]
    try:
        with torch.no_grad():
            for run_index,(name,folder) in enumerate(folders.items()):
                if run_limit is not None and run_index >= run_limit:
                    break
                if interval_limit is not None and len(trace) >= interval_limit:
                    break
                motion = ArchivedMotion(folder,episodes[name],name)
                inputs.extend(motion.inputs)
                extension = NativeMultisensoryExtensionEncoder(contract_file,native_identity=motion.identity)
                times = motion.samples[:,0]
                encoded = []
                for observation,prefix in motion.native_records:
                    t = observation.original_received_monotonic_seconds
                    sample = decode_native_extension(prefix,t)
                    assert sample.packet_id==observation.packet_id
                    extra = extension.encode(sample,episode_id=name,decision_monotonic_seconds=t)
                    encoded.append((observation,extra))
                with CameraArchive(folder) as archive:
                    stream = list(map(json.loads,archive.read("stream.jsonl").splitlines()))
                state = None
                prior_goal = None
                for previous,item in zip(stream,stream[1:]):
                    if interval_limit is not None and len(trace) >= interval_limit:
                        break
                    frame,t = previous["capture_id"],previous["received"]
                    end = item["received"]
                    row = (command_source(name,frame) if command_source is not None
                           else commands.get((name,frame)))
                    goal_row = row
                    if lag_one_frame:
                        goal_row = prior_goal
                        prior_goal = row if (row and row.get("maneuver") is not None
                                             and previous["generation"] == item["generation"]
                                             and item["capture_id"] == frame+1 and 0 < end-t <= .18) else None
                        if goal_row is not None and goal_row["frame"] != frame-1:
                            goal_row = None
                    if (not row or row.get("maneuver") is None or goal_row is None
                            or previous["generation"] != item["generation"]
                            or item["capture_id"] != frame+1 or not 0 < end-t <= .18):
                        state = None
                        continue
                    clock = t
                    j = int(np.searchsorted(times,t,side="right")-1)
                    if j<0:
                        state = None
                        continue
                    if not np.isclose(row["body_speed_mps"],motion.samples[j,1],atol=1e-7):
                        raise ValueError("Plan and flight replay select different speed records")
                    started = time.perf_counter()
                    if state is None:
                        state = graph.initial_state(len(names))
                        counts["state_initializations"]+=1
                    command = torch.tensor(goal_row["maneuver"],device=device).repeat(len(names),1)
                    snapshots = {}
                    while clock < end-1e-10:
                        until = min(end,times[j+1]) if j+1<len(times) else end
                        if until-times[j] > MAX_BODY_AGE_S:
                            state = None
                            counts["stale_intervals"]+=1
                            break
                        observation,extra = encoded[j]
                        body = torch.tensor(observation.values,dtype=torch.float32,device=device).repeat(len(names),1)
                        valid = torch.tensor(observation.valid,device=device).repeat(len(names),1)
                        ext = torch.tensor(extra.values,dtype=torch.float32,device=device).repeat(len(names),1)
                        ext_valid = torch.tensor(extra.valid,device=device).repeat(len(names),1)
                        if not native_only:
                            body[1,4]=0.
                            body[2,:2]=0.
                            body[3,8:12]=.25
                            ext[3,7:11]=port.reference[7:11]
                            body[4,12:16]=port.reference[11:15]
                            ext[4,11:15]=port.reference[11:15]
                        holder["delta"]=port(ext,ext_valid)
                        intent,muscle,state=graph.advance(command,body,valid,goals,state,float(until-clock),**kwargs)
                        if record_first_response:
                            for key,offset in (("first",0.),("20ms",.02),("40ms",.04)):
                                if key not in snapshots and until-t>=offset:
                                    snapshots[key] = dict(intent=intent.cpu().tolist(),muscles=muscle.cpu().tolist(),
                                                          end=until,elapsed_s=until-t,
                                                          completed_s=time.perf_counter(),packet=observation.packet_id)
                        clock=until
                        if j+1<len(times) and clock==times[j+1] and clock<end-1e-10:
                            j+=1
                    if state is not None:
                        trace.append(dict(run=name,frame=frame,received=t,end=end,
                                          last_packet=encoded[j][0].packet_id,command=goal_row["maneuver"],
                                          goal_frame=goal_row["frame"],
                                          intent=intent.cpu().tolist(),muscles=muscle.cpu().tolist()))
                        if record_first_response:
                            trace[-1]["first_response"] = snapshots["first"]
                            trace[-1]["snapshots"] = snapshots
                        trace[-1]["compute_started_s"] = started
                        trace[-1]["compute_s"] = time.perf_counter()-started
    finally:
        hook.remove()
    if not trace:
        raise ValueError("No fresh flight replay intervals")
    values=np.array([r["intent"] for r in trace])
    effects={name:np.max(abs(values[:,i]-values[:,0]),axis=0).tolist() for i,name in enumerate(names)}
    profiled = trace[5:] if len(trace)>5 else trace
    elapsed = np.array([r["compute_s"] for r in profiled])
    budgets = np.array([r["end"]-r["received"] for r in profiled])
    timing = dict(samples=len(profiled), median_ms=float(np.median(elapsed)*1000),
                  p95_ms=float(np.quantile(elapsed,.95)*1000),
                  median_capture_interval_ms=float(np.median(budgets)*1000),
                  exceeds_capture_interval=int((elapsed>budgets).sum()))
    output.mkdir(parents=True,exist_ok=False)
    jsonl(output/"rows.jsonl",trace)
    write(output/"result.json",dict(revision=line_proof["revision"],seed=0,conditions=names,
        intervals=len(trace),counts=dict(counts),timing=timing,device=device,backend=backend,lag_one_frame=lag_one_frame,max_intent_delta=effects,actuator_messages=0,
        rows_sha256=digest(output/"rows.jsonl"),inputs_sha256={str(p):digest(p) for p in inputs},
        termination="same-episode causal flight circuit replay completed"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("archives","native","lines","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--lag-one-frame", action="store_true")
    parser.add_argument("--interval-limit", type=int)
    replay(**vars(parser.parse_args()))


if __name__=="__main__":
    main()
