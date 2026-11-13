import json
import os
from pathlib import Path
import numpy as np
import torch
from audit_corner_speed_preview import LIMITS, CORNER_SPEED_MPS
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter, ExperimentalBodyFeedbackCircuit
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.local_racing_line import maneuver_from_line, plan_local_line, speed_bound_with_corner_preview
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal
from train_vision_teacher_line import digest, read, write

out = Path("runs/vision-line-flight-20261110-01")
root = Path(os.environ["LOCALAPPDATA"])/"flybrain/runtime-runs"
fit = root/"20261101-speed-wheel-turn-control-fit"
weights = fit/"speed-body-adapter.pt"
spec = Path("configs/flight-circuit-separated-v5-20261029.json")
checkpoint = Path("runs/flight-separated-command-probe-20261029-t1081/checkpoint.pt")
forecast = Path("runs/vision-corner-speed-preview-20261110-01/advisories.jsonl")
boundaries = Path("runs/vision-registered-corridor-20261110-01/monza-predictions.jsonl")
identity = Path("runs/vision-road-corridor-20261110-02/monza-predictions.jsonl")
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]

torch.set_num_threads(1)
torch.manual_seed(0)
device = "cuda"
assert read(fit/"result.json")["checkpoint_sha256"] == digest(weights)
payload = torch.load(weights,map_location="cpu",weights_only=True)
assert payload["actor_qualified"] is False
motor = PhysicalManeuverCircuit(spec,torch.load(checkpoint,map_location="cpu",weights_only=True)).to(device)
graph = ExperimentalBodyFeedbackCircuit(motor)
graph.adapter = CenteredSpeedBodyFeedbackAdapter(len(motor.backbone.body_ports)).to(device)
graph.adapter.load_state_dict(payload["model"],strict=True)
graph.eval().requires_grad_(False)

forecasts = {r["frame"]:r for r in rows(forecast) if r["split"]=="heldout" and r["model"]=="linear_plus_flyvis"}
first, second = rows(boundaries), rows(identity)
assert len(first)==len(second)
cases = []
for row, meta in zip(first,second):
    assert row["episode_id"]==meta["episode_id"] and row["observation_time_s"]==meta["observation_time_s"]
    frame = meta["capture_id"]
    if frame not in forecasts:
        continue
    f = forecasts[frame]
    speed = row["source_speed_mps"]
    line, _ = plan_local_line(row["road_geometry"]["sections"],speed,**LIMITS)
    original = maneuver_from_line(line,speed,lookahead_m=12.)
    bound = speed_bound_with_corner_preview(line,f["predicted_distance_m"],corner_speed_mps=CORNER_SPEED_MPS,braking_mps2=LIMITS["braking_mps2"])
    preview = maneuver_from_line(line,speed,lookahead_m=12.,speed_bound=bound)
    assert abs(float(bound[0])-f["preview_entry_bound_mps"])<1e-7
    assert original[0]==preview[0]
    cases.append((frame,speed,original,preview))
turns = [min(cases,key=lambda r:r[2][0]),max(cases,key=lambda r:r[2][0]),max(cases,key=lambda r:r[2][1]-r[3][1])]
assert turns[0][2][0] < 0 < turns[1][2][0] and turns[2][2][1]>turns[2][3][1]
names = ("baseline","preview","yaw","tilt","load","tyre","wheel","missing_tyre")
groups = {"yaw":([4],[.3]),"tilt":([0,1],[.05,-.08]),"load":([8,9,10,11],[.32,.18,.29,.21]),
          "tyre":([12,13,14,15],[95.,100.,95.,100.]),"wheel":([17,18,19,20],[80.,87.,80.,87.])}
n = len(turns)*len(names)
body = torch.zeros((n,21),device=device)
mask = torch.ones_like(body,dtype=torch.bool)
mask[:,[2,3,5,6,7]] = False
commands = []
for i, (_,speed,original,preview) in enumerate(turns):
    for j,name in enumerate(names):
        k=i*len(names)+j
        commands.append(preview if name=="preview" else original)
        body[k,16] = speed
        body[k,8:12] = .25
        body[k,12:16] = 75.
        body[k,17:21] = 80.
        if name in groups:
            ids, values = groups[name]
            body[k,ids] = torch.tensor(values,device=device)
        if name=="missing_tyre":
            mask[k,12:16] = False
command = torch.tensor(np.asarray(commands),device=device)
goal = torch.tensor(racing_goal(90,n),device=device)
kwargs = dict(command_contract=MANEUVER_CONTRACT,goal_contract=RACING_GOAL_CONTRACT)
with torch.no_grad():
    state = graph.initial_state(n)
    baseline_rows = torch.arange(len(turns),device=device)*len(names)
    for _ in range(50):
        _, _, state = graph.advance(command[baseline_rows],body[baseline_rows],mask[baseline_rows],
            goal[baseline_rows],state[:len(turns)],.01,**kwargs)
    state = state.repeat_interleave(len(names),dim=0)
    traces, muscles = [], []
    for _ in range(30):
        intent, muscle, state = graph.advance(command,body,mask,goal,state,.01,**kwargs)
        traces.append(intent.cpu().numpy())
        muscles.append(muscle.cpu().numpy())
intents = np.stack(traces).reshape(30,len(turns),len(names),-1)
muscles = np.stack(muscles).reshape(30,len(turns),len(names),-1)
assert np.isfinite(intents).all() and np.isfinite(muscles).all()
effects = {}
for i,name in enumerate(names[1:],1):
    effects[name] = dict(max_intent_delta=np.max(np.abs(intents[:,:,i]-intents[:,:,0]),axis=(0,1)).tolist(),
        max_muscle_delta=float(np.max(np.abs(muscles[:,:,i]-muscles[:,:,0]))))
np.savez_compressed(out/"responses.npz",intent=intents,muscles=muscles,commands=np.asarray(commands),
                   body=body.cpu().numpy(),body_valid=mask.cpu().numpy())
write(out/"result.json",dict(revision="9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda",seed=0,
    hashes={str(p):digest(p) for p in (spec,checkpoint,weights,forecast,boundaries,identity,Path(__file__),
      Path("src/flybrain_drive/local_racing_line.py"),Path("src/flybrain_drive/flight_maneuver.py"),out/"responses.npz")},
    conditions=names,frames=[r[0] for r in turns],commands=np.asarray(commands).reshape(len(turns),len(names),2)[:,:2].tolist(),
    effects=effects,scope="stored visual paths and speed with synthetic held body surrogates",
    actuator_messages=0,termination="recurrent paired physical flight graph probe completed"))
print(effects)
