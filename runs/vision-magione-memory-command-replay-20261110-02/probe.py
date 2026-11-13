from pathlib import Path
import os
import numpy as np
import torch

from fit_ac_mujoco_corner_control import response
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter, ExperimentalBodyFeedbackCircuit
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from train_vision_teacher_line import digest, read, write

out=Path("runs/vision-magione-memory-command-replay-20261110-02")
cal=Path("runs/mujoco-ac-corner-control-engine-20261107/calibration.json")
data=read(cal)
assert data["accepted"] and data["parameter_names"]==[
    "drive_m_s2","brake_m_s2","drag_m_s2_at_25ms","steer_gain","understeer","yaw_response_per_s"]
parameters=np.array(data["parameters"])
spec=Path("configs/flight-circuit-separated-v5-20261029.json")
cp=Path("runs/flight-separated-command-probe-20261029-t1081/checkpoint.pt")
base=Path(os.environ["LOCALAPPDATA"])/"flybrain/runtime-runs/20261101-speed-wheel-turn-control-fit/speed-body-adapter.pt"
reference=Path("runs/vision-line-flight-20261110-01/result.json")
previous=read(reference)
for p in (spec,cp,base):
    assert previous["hashes"][str(p)]==digest(p)
torch.set_num_threads(1)
torch.manual_seed(0)
device="cuda"
motor=PhysicalManeuverCircuit(spec,torch.load(cp,map_location="cpu",weights_only=True)).to(device)
graph=ExperimentalBodyFeedbackCircuit(motor)
graph.adapter=CenteredSpeedBodyFeedbackAdapter(len(motor.backbone.body_ports)).to(device)
graph.adapter.load_state_dict(torch.load(base,map_location="cpu",weights_only=True)["model"],strict=True)
graph.eval().requires_grad_(False)
dt=.02
kwargs=dict(command_contract=MANEUVER_CONTRACT,goal_contract=RACING_GOAL_CONTRACT)
def inputs(speeds):
    b=torch.zeros((len(speeds),21),device=device)
    b[:,16]=torch.as_tensor(speeds,device=device)
    b[:,8:12]=.25
    valid=torch.zeros_like(b,dtype=torch.bool)
    valid[:,[4,8,9,10,11,16]]=True
    c=torch.zeros((len(speeds),2),device=device)
    c[:,1]=b[:,16]
    goals=torch.as_tensor(racing_goal(90,len(speeds)),device=device)
    return c,b,valid,goals

def settle(c,b,valid,goals):
    state=graph.initial_state(len(b))
    for _ in range(100):
        intent,_,state=graph.advance(c,b,valid,goals,state,dt,**kwargs)
    return state,intent.cpu().numpy().copy()

from flybrain_drive.local_racing_line import plan_local_line,maneuver_from_line,speed_bound_with_corner_preview

speed_cal=Path("runs/vision-flight-speed-plant-20261110-01/calibration.json")
yaw_cal=Path("runs/vision-flight-yaw-calibration-20261110-01/result.json")
sensitivity=read(speed_cal)["sensitivity"]
yaw_gain=read(yaw_cal)["development_command_gain"]
limits=dict(clearance_m=1.1,max_curvature=.15,lateral_accel_mps2=4.,braking_mps2=4.,speed_ceiling_mps=25./3.6)
import base64
import hashlib
import json
from collections import Counter
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_multisensory_extension import NativeMultisensoryExtensionEncoder,decode_native_extension
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder,decode_navigation_body_prefix
from train_next_corner import BASE,MAGIONE
from train_vision_teacher_line import jsonl

pixel=Path("runs/vision-magione-pixel-boundary-20261110-02")
vision=read(pixel/"result.json")
memory_run=Path("runs/vision-magione-boundary-memory-20261110-01")
memory_receipt=read(memory_run/"result.json")
agreement_m=memory_receipt["current_edge_agreement_gate"]["gate_max_disagreement_m"]
assert agreement_m==.75
corner=Path("runs/vision-next-corner-receipt-20261110-01/flyvis-predictions.jsonl")
assert digest(corner)==read(pixel/"comparison.json")["forecast_sha256"]
thermal=Path("runs/thermal-driving-20261103")
port_file=thermal/"native-fit-02/body-extension.pt"
contract=thermal/"extension-data-01/contract.json"
assert digest(port_file)==read(port_file.parent/"result.json")["checkpoint_sha256"]
payload=torch.load(port_file,map_location="cpu",weights_only=True)
assert digest(contract)==payload["extension_contract_sha256"]
w=payload["body_port"]
port=MultisensoryBodyFeedbackPort(w["reference"],w["scale"],len(motor.backbone.body_ports)).to(device)
port.load_state_dict(w)
port.eval().requires_grad_(False)
holder={}
handle=graph.adapter.register_forward_hook(lambda module,args,output:output+holder["delta"])
goal=torch.tensor(racing_goal(90,2),device=device)
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
forecast={(a["run"],a["frame"]):a for a in rows(corner) if a["split"]=="external"}
summaries={}
for name,sub in MAGIONE:
    root=BASE/name/sub
    cert=read(root/"result.json")
    manifest=read(root/"manifest.json")["config"]
    identity={k:manifest["identity"][k] for k in ("car_model","ac_version","shared_memory_version")}
    assert manifest["identity"]["track"]=="magione"
    assert manifest["teacher_speed_targets_kmh"]==[20,25,30] and manifest["hard_speed_ceiling_kmh"]==40
    raw_file=root/"owner-native-copy.json"
    ticks_file=root/"owner-ticker-events.json"
    assert digest(raw_file)==cert["native_sha256"] and digest(ticks_file)==cert["ticker_events_sha256"]
    native={}
    for row in read(raw_file):
        a=row["authority"]
        raw=base64.b64decode(a["prefix_base64"],validate=True)
        assert hashlib.sha256(raw).hexdigest()==a["prefix_sha256"] and a["same_native_copy"]
        prefix=raw[:PHYSICS_PREFIX.size]
        assert hashlib.sha256(prefix).hexdigest()==a["body_prefix_sha256"]
        t=float(a["received_monotonic_seconds"])
        old=native.setdefault(a["packet_id"],(t,prefix))
        assert old[1]==prefix
    frame_file=pixel/f"{name}.jsonl"
    assert digest(frame_file)==vision["sources"][name]["output_sha256"]
    frames={a["frame"]:a for a in rows(frame_file)}
    lines={}
    memory_file=memory_run/f"{name}.jsonl"
    assert digest(memory_file)==memory_receipt["counts"][name]["output_sha256"]
    memories={a["frame"]:a for a in rows(memory_file)}
    assert memories.keys()==frames.keys()
    for f,frame in frames.items():
        assert (name,f) in forecast
        remembered=memories[f]
        assert abs(remembered["t"]-forecast[name,f]["time"])<1e-6
        sections=[]
        for direct,carried in zip(frame["sections"],remembered["sections"],strict=True):
            partial=[k for k in ("right_y_m","left_y_m") if direct[k] is not None]
            use=(carried["source"]=="carried" and bool(partial)
                 and max(abs(direct[k]-carried[k]) for k in partial)<=agreement_m)
            sections.append(carried if use else direct)
        frame["prior_path"]=frame["path_generated"]
        frame["sections"]=sections
        line,_=plan_local_line(sections,0.,**limits)
        frame["path_generated"]=line is not None
        if line is not None:
            frow=forecast[name,f]
            lines[f]=(line,speed_bound_with_corner_preview(line,frow["predicted_distance_m"],
                                      corner_speed_mps=20./3.6,braking_mps2=limits["braking_mps2"]))
    nav=NavigationBodyEncoder("configs/ac-virtual-navigation-body-v2.json",native_identity=identity)
    extension=NativeMultisensoryExtensionEncoder(contract,native_identity=identity)
    previous=None
    initial=None
    state=None
    outputs=[]
    abstentions=[]
    counts=Counter()
    joined=set()
    joined_path=set()
    with torch.no_grad():
        for event in read(ticks_file):
            if event["status"]!="shadow_output" or event.get("capture_id") not in frames:
                counts["outside_matched_frames"]+=1
                previous=None
                continue
            f=event["capture_id"]
            packet=event["body_packet"]
            if packet not in native:
                counts["missing_native_packet"]+=1
                previous=None
                continue
            t=float(event["decision_time"])
            receipt,prefix=native[packet]
            forecast_row=forecast[name,f]
            assert abs(forecast_row["time"]-event["camera_received"])<1e-6
            assert receipt==event["body_first_receipt"] and event["camera_received"]<=t and receipt<=t
            if t-receipt>.05:
                counts["stale_native"]+=1
                previous=None
                continue
            joined.add(f)
            if not frames[f]["path_generated"]:
                counts["no_visual_path"]+=1
                abstentions.append(dict(decision_time=t,capture=f,body_packet=packet,reason="no_visual_path",commands=None))
                previous=None
                continue
            joined_path.add(f)
            if f not in lines:
                line,_=plan_local_line(frames[f]["sections"],0.,**limits)
                assert line is not None
                bound=speed_bound_with_corner_preview(line,forecast_row["predicted_distance_m"],
                     corner_speed_mps=20./3.6,braking_mps2=limits["braking_mps2"])
                lines[f]=(line,bound)
            obs=nav.encode(decode_navigation_body_prefix(prefix,receipt),episode_id=name,decision_monotonic_seconds=t)
            ext=extension.encode(decode_native_extension(prefix,receipt),episode_id=name,decision_monotonic_seconds=t)
            assert obs.valid[[0,1,4,8,9,10,11,16]].all() and ext.valid[7:15].all()
            if previous is not None and (not 0<t-previous[0]<=.05 or t-previous[1]>.05):
                counts["interval_break"]+=1
                previous=None
            command=maneuver_from_line(lines[f][0],float(obs.values[16]),lookahead_m=12.,speed_bound=lines[f][1])
            yaw=command[0]
            if yaw:
                command[0]=np.clip(yaw*yaw_gain[str(float(np.sign(yaw)))],-2.,2.)
            if previous is None:
                counts["segments"]+=1
                state=graph.initial_state(2)
                initial=(obs.values.copy(),ext.values.copy(),ext.valid.copy())
                onset=t
                _,neutral_body,_,_=inputs(np.full(2,float(obs.values[16])))
                neutral_body[:]=torch.tensor(np.tile(obs.values,(2,1)),dtype=torch.float32,device=device)
                neutral_valid=torch.tensor(np.tile(obs.valid,(2,1)),device=device)
                neutral_cmd=torch.zeros((2,2),device=device)
                neutral_cmd[:,1]=neutral_body[:,16].clamp(max=25.)
                holder["delta"]=port(torch.tensor(np.tile(ext.values,(2,1)),dtype=torch.float32,device=device),
                                      torch.tensor(np.tile(ext.valid,(2,1)),device=device))
                state,neutral=settle(neutral_cmd,neutral_body,neutral_valid,goal)
            b=np.tile(obs.values,(2,1))
            b[0,[0,1,*range(8,16)]]=initial[0][[0,1,*range(8,16)]]
            x=np.tile(ext.values,(2,1))
            k=np.tile(ext.valid,(2,1))
            x[0,7:15]=initial[1][7:15]
            k[0,7:15]=initial[2][7:15]
            arguments=(torch.tensor(np.tile(command,(2,1)),device=device),
                       torch.tensor(b,dtype=torch.float32,device=device),
                       torch.tensor(np.tile(obs.valid,(2,1)),device=device))
            delta=port(torch.tensor(x,dtype=torch.float32,device=device),torch.tensor(k,device=device))
            if previous is not None:
                pt,_,pf,pp,pv,args,pdelta,pcommand=previous
                holder["delta"]=pdelta
                intent,_,state=graph.advance(*args,goal,state,t-pt,**kwargs)
                raw=intent.cpu().numpy()-neutral
                sg=parameters[3]*max(pv,1.)/(1+parameters[4]*(pv/25.)**2)
                steer=np.clip(raw[:,0]/(sg*.22),-1.,1.)
                ax=parameters[2]*(pv/25.)**2+.9*raw[:,1]/sensitivity
                command_out=np.c_[steer,np.clip(ax/parameters[0],0.,1.),np.clip(-ax/parameters[1],0.,1.)]
                assert np.isfinite(command_out).all()
                outputs.append(dict(decision_time=t,capture=pf,body_packet=pp,
                    body_speed_mps=pv,target_speed_mps=float(pcommand[1]),path_from_direct=frames[pf]["prior_path"],commands=command_out.tolist(),segment_age_s=t-onset))
            previous=(t,receipt,f,packet,float(obs.values[16]),arguments,delta,command)
    dest=out/f"{name}.jsonl"
    jsonl(dest,outputs)
    gap_file=out/f"{name}-abstentions.jsonl"
    jsonl(gap_file,abstentions)
    stable=[q for q in outputs if q["segment_age_s"]>=.30]
    result=dict(counts=dict(counts),path_frames_in_source=len([f for f,x in frames.items() if x["path_generated"]]),
       joined_frames=len(joined),joined_path_frames=len(joined_path),replayed_ticks=len(outputs),after_burnin_ticks=len(stable),
       input_sha256={str(p):digest(p) for p in (raw_file,ticks_file,frame_file,root/"manifest.json",memory_file)},trace_sha256=digest(dest),abstentions_sha256=digest(gap_file))
    if stable:
        x=np.array([q["commands"] for q in stable])
        result["mean_abs_delta_by_axis"]=np.abs(x[:,1]-x[:,0]).mean(0).tolist()
        result["max_abs_delta_by_axis"]=np.abs(x[:,1]-x[:,0]).max(0).tolist()
    summaries[name]=result
    print(name,result,flush=True)
handle.remove()
write(out/"result.json",dict(revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",seed=0,
    input_sha256={str(p):digest(p) for p in (Path(__file__),corner,port_file,contract,memory_run/"result.json",spec,cp,base,cal,speed_cal,yaw_cal)},
    summaries=summaries, speed_ceiling_kmh=25, preview_corner_speed_kmh=20, actuator_messages=0,
    scope="same-run native replay with current-edge-agreement boundary memory; abstentions explicit; engineering actuator conversion",
    termination="all archived Magione episodes replayed"))
