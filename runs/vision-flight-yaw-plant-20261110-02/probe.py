from pathlib import Path
import os
import numpy as np
import torch

from fit_ac_mujoco_corner_control import response
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter, ExperimentalBodyFeedbackCircuit
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from train_vision_teacher_line import digest, read, write

out=Path("runs/vision-flight-yaw-plant-20261110-02")
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
cases=[(v,g,d,k,kick) for v in (12.,20.) for g in (.55,.9) for d in (-1.,1.)
       for k in ("full","yaw_neutral") for kick in (False,True)]
n=len(cases)
speeds,grips,directions=np.asarray([c[:3] for c in cases]).T
active=np.array([c[3]=="full" for c in cases])
mask=torch.ones((n,21),dtype=torch.bool,device=device)
mask[:,[2,3,5,6,7,12,13,14,15,17,18,19,20]]=False
body=torch.zeros((n,21),device=device)
body[:,8:12]=.25
body[:,16]=torch.tensor(speeds,device=device)
goals=torch.tensor(racing_goal(90,n),device=device)
cmd=torch.zeros((n,2),device=device)
cmd[:,1]=body[:,16]
kwargs=dict(command_contract=MANEUVER_CONTRACT,goal_contract=RACING_GOAL_CONTRACT)
dt=.02
# Approximate steady calibration from the earlier independent +/- .30 rad/s step.
intent_per_yaw=.22
with torch.no_grad():
    state=graph.initial_state(n)
    for _ in range(75):
        intent,_,state=graph.advance(cmd,body,mask,goals,state,dt,**kwargs)
    neutral=intent[:,0].cpu().numpy().copy()
    plant=np.column_stack((speeds,np.zeros(n)))
    speed_hold=parameters[2]*(speeds/25.)**2/parameters[0]
    steer_gain=parameters[3]*speeds/(1+parameters[4]*(speeds/25.)**2)
    trace=[]
    for j in range(250):
        t=j*dt
        target=directions*(.30 if .5<=t<3. else 0.)
        # Fixed kick to the simulated body. Each branch gets the same kick.
        if j==90:
            plant[:,1]+=.18*np.array([c[4] for c in cases])
        body[:,4]=torch.as_tensor(np.where(active,plant[:,1],0.),device=device)
        cmd[:,0]=torch.as_tensor(target,device=device)
        intent,_,state=graph.advance(cmd,body,mask,goals,state,dt,**kwargs)
        raw=intent[:,0].cpu().numpy()
        steer=np.clip((raw-neutral)/(steer_gain*intent_per_yaw),-1.,1.)
        request=np.column_stack((steer,speed_hold,np.zeros(n)))
        delta=response(plant,request,parameters,grips)
        plant+=dt*delta
        trace.append(np.column_stack((target,plant,steer,raw)))
trace=np.stack(trace)
assert np.isfinite(trace).all()
assert np.allclose(trace[:,:,1],speeds)
summary=[]
for i,(v,g,d,k,kick) in enumerate(cases):
    target,_,yaw,steer,_=trace[:,i].T
    corner=slice(25,150)
    steady=slice(65,88)
    summary.append(dict(speed_mps=v,grip=g,direction=d,condition=k,kick=kick,
        corner_yaw_rmse=float(np.sqrt(np.mean((yaw[corner]-target[corner])**2))),
        signed_steady_yaw=float(d*np.mean(yaw[steady])),
        disturbance_rmse=float(np.sqrt(np.mean((yaw[90:115]-target[90:115])**2))),
        exit_yaw_rms=float(np.sqrt(np.mean(yaw[150:]**2))),
        peak_abs_steer=float(np.abs(steer).max())))
np.savez_compressed(out/"trace.npz",trace=trace,dt=dt,
    fields=["target_left_rad_s","speed_mps","plant_left_rad_s","ac_steer","neural_left_intent"])
stats={c:{m:float(np.mean([r[m] for r in summary if r["condition"]==c and r["kick"]])) for m in
          ("corner_yaw_rmse","signed_steady_yaw","disturbance_rmse","exit_yaw_rms")} for c in ("full","yaw_neutral")}
for c in stats:
    indexes=[i for i,k in enumerate(cases) if k[3]==c and not k[4]]
    paired=trace[90:115,np.array(indexes)+1,2]-trace[90:115,indexes,2]
    stats[c]["isolated_disturbance_yaw_rmse"]=float(np.sqrt(np.mean(paired**2)))
    stats[c]["undisturbed_tracking_yaw_rmse"]=float(np.mean([summary[i]["corner_yaw_rmse"] for i in indexes]))
write(out/"result.json",dict(revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",seed=0,
    sha256={str(p):digest(p) for p in (spec,cp,base,cal,reference,Path(__file__),out/"trace.npz")},
    intent_per_yaw=intent_per_yaw,cases=summary,mean_by_condition=stats,
    scope="synthetic prescribed yaw and fixed-speed planar AC-fit dynamics, body rotation feedback only",
    actuator_messages=0,termination="all paired simulated yaw trajectories completed"))
print(stats)
