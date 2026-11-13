from pathlib import Path
import os
import numpy as np
import torch

from fit_ac_mujoco_corner_control import response
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter, ExperimentalBodyFeedbackCircuit
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from train_vision_teacher_line import digest, read, write

out=Path("runs/vision-flight-speed-plant-20261110-01")
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

with torch.no_grad():
    dev_speed=np.repeat([12.,20.],3)
    offset=np.tile([-3.,0.,3.],2)
    cmd,body,valid,goals=inputs(dev_speed)
    state,neutral=settle(cmd,body,valid,goals)
    cmd[:,1]+=torch.as_tensor(offset,device=device)
    for _ in range(90):
        out_dev,_,state=graph.advance(cmd,body,valid,goals,state,dt,**kwargs)
    delta=(out_dev.cpu().numpy()[:,1]-neutral[:,1]).reshape(2,3)
    # Compensate any shared residual from the end of warmup.
    delta-=delta[:,1:2].copy()
    sensitivity=float(np.sum(delta*offset.reshape(2,3))/np.sum(offset**2))
    write(out/"calibration.json",dict(speeds=dev_speed.tolist(),offset=offset.tolist(),
                                      forward_intent_delta=delta.tolist(),sensitivity=sensitivity))
    assert np.all(delta[:,0]<0) and np.all(delta[:,2]>0) and sensitivity>1e-9
    yaw_source=Path("runs/vision-flight-yaw-calibration-20261110-01/result.json")
    yaw_gain=read(yaw_source)["development_command_gain"]
    cases=[(v,g,d,dv,closed,kick) for v in (11.,17.,22.) for g in (.65,.82) for d in (-1.,1.)
           for dv in (-2.,2.) for closed in (False,True) for kick in (False,True)]
    v0,grip,side,dv,closed,kick=np.array(cases).T
    cmd,body,valid,goals=inputs(v0)
    state,neutral=settle(cmd,body,valid,goals)
    plant=np.c_[v0,np.zeros(len(v0))]
    gain=np.array([yaw_gain[str(float(s))] for s in side])
    trace=[]
    for j in range(300):
        t=j*dt
        v_target=v0+dv*(.5<=t<4.5)
        yaw_target=side*.22*(.7<=t<4.)
        if j==110:
            plant[:,0]-=1.*kick
        v,r=plant.T
        body[:,4]=torch.as_tensor(r,device=device)
        body[:,16]=torch.as_tensor(np.where(closed,v,v0),device=device)
        cmd[:,0]=torch.as_tensor(yaw_target*gain,device=device)
        cmd[:,1]=torch.as_tensor(v_target,device=device)
        intent,_,state=graph.advance(cmd,body,valid,goals,state,dt,**kwargs)
        drive=intent.cpu().numpy()-neutral
        steer_gain=parameters[3]*v/(1+parameters[4]*(v/25.)**2)
        steer=np.clip(drive[:,0]/(steer_gain*.22),-1.,1.)
        # Nominal drag feedforward is held at initial speed in both arms.
        ax=parameters[2]*(v0/25.)**2+.9*drive[:,1]/sensitivity
        gas=np.clip(ax/parameters[0],0.,1.)
        brake=np.clip(-ax/parameters[1],0.,1.)
        plant+=dt*response(plant,np.c_[steer,gas,brake],parameters,grip)
        trace.append(np.c_[v_target,yaw_target,plant,steer,gas,brake])
trace=np.stack(trace)
assert np.isfinite(trace).all() and trace[:,:,2].min()>0
summary={}
for closed in (False,True):
    idx=np.flatnonzero(np.array([c[4]==closed and not c[5] for c in cases]))
    assert all(cases[i][:-1]==cases[i+1][:-1] and cases[i+1][-1] for i in idx)
    clean=trace[25:225,idx]
    kick_error=trace[110:160,idx+1,2]-trace[110:160,idx,2]
    summary[str(closed)]=dict(undisturbed_speed_rmse=float(np.sqrt(np.mean((clean[:,:,2]-clean[:,:,0])**2))),
      isolated_disturbance_speed_rms=float(np.sqrt(np.mean(kick_error**2))),
      undisturbed_yaw_rmse=float(np.sqrt(np.mean((clean[:,:,3]-clean[:,:,1])**2))),
      acceleration_ticks=int((trace[:,idx,5]>.05).sum()),brake_ticks=int((trace[:,idx,6]>.05).sum()))
np.savez_compressed(out/"trace.npz",trace=trace,cases=np.array(cases),dt=dt,
    fields=["target_speed","target_yaw","speed","yaw","steer","throttle","brake"])
write(out/"result.json",dict(revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",seed=0,
    inputs_sha256={str(p):digest(p) for p in (spec,cp,base,cal,yaw_source,reference,Path(__file__))},
    trace_sha256=digest(out/"trace.npz"),calibration_sha256=digest(out/"calibration.json"),
    cases=len(cases),summary=summary,actuator_messages=0,termination="paired synthetic combined yaw and speed closed loop completed"))
print(summary)
