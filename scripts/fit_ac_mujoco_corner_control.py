"""Fit a reduced-order MuJoCo plant from recorded AC requests and train on synthetic corners."""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import hashlib
from itertools import product
import os
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.moving_stop_calibration import native_track_fields


SPEED_RANGE = (8., 42.)
G = 9.81


def load_trace(folder):
    events = folder/'events.jsonl'
    proof = json.loads((folder/'result.json').read_bytes())
    digest = sha256(events)
    if proof['events_sha256'] != digest:
        raise ValueError('AC events digest differs')
    body, commands = [], []
    for line in events.open(encoding='utf-8'):
        row = json.loads(line)
        if row['kind'] == 'raw_pair':
            native = row['native']
            raw = base64.b64decode(native['prefix_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest()!=native['prefix_sha256']:
                raise ValueError('Original native prefix differs')
            sample = decode_physics_prefix(raw,native['received_monotonic_seconds'])
            if sample.packet_id != native['packet_id']:
                raise ValueError('Original native packet id differs')
            body.append([sample.received_monotonic_seconds,sample.fields['speedKmh']/3.6,
                         sample.fields['localAngularVel'][1],native_track_fields(sample)[0]])
        elif row['kind']=='teacher_request' and row['request']['driver_accepted']:
            r=row['request']
            commands.append([r['driver_call_finished_perf_counter_seconds'],
                             *[r['requested_control'][k] for k in ('steering','throttle','brake')]])
    body, commands = np.array(body), np.array(commands)
    if not (np.all(np.diff(body[:,0])>0) and np.all(np.diff(commands[:,0])>0)):
        raise ValueError('Nonadvancing AC source')
    return body, commands, digest


def response(state, command, p, grip):
    v,r = np.moveaxis(state,-1,0)
    steering,throttle,brake = np.moveaxis(command,-1,0)
    drive, brake_gain, drag, steer_gain, understeer, rate = p
    ax=drive*throttle-brake_gain*brake-drag*(v/25.)**2
    # Only the calibrated nominal grip is identified from AC. Other grip values are perturbations.
    ax=np.clip(ax,-grip*G,grip*G)
    ay_limit=np.sqrt(np.maximum((grip*G)**2-ax**2,0))
    target=steer_gain*steering*v/(1+understeer*(v/25.)**2)
    target=np.clip(target,-ay_limit/np.maximum(v,1.),ay_limit/np.maximum(v,1.))
    return np.stack([ax,rate*(target-r)],axis=-1)


def windows(body, command, n=8):
    span=np.arange(n+1)+np.arange(0,len(body)-n,2)[:,None]
    rows=body[span]
    dt=np.diff(rows[:,:,0],axis=1)
    # Use only the last command already issued when the original interval began.
    ids=np.searchsorted(command[:,0],rows[:,:-1,0],side='right')-1
    age=rows[:,:-1,0]-command[ids,0]
    keep=((dt>.018)&(dt<.080)&(ids>=0)&(age>=0)&(age<.075)).all(1)
    keep&=((rows[:,:,1]>=SPEED_RANGE[0])&(rows[:,:,1]<=SPEED_RANGE[1])&(rows[:,:,3]==0)).all(1)
    return rows[keep,:,1:3], command[ids[keep],1:], dt[keep]


def simulate_windows(w,p):
    y,u,dt=w
    pred=y[:,0].copy()
    steps=[]
    for i in range(dt.shape[1]):
        pred=pred+dt[:,i,None]*response(pred,u[:,i],p,1.15)
        steps.append(pred.copy())
    return np.stack(steps,axis=1)


def simulate_mujoco_windows(w, p):
    init_worker()
    y, u, dt = w
    out = np.empty_like(y[:, 1:])
    for i, initial in enumerate(y[:, 0]):
        mj.mj_resetData(model, data)
        data.qvel[:] = [initial[0], 0., initial[1]]
        for j, interval in enumerate(dt[i]):
            advance(u[i, j], p, 1.15, interval)
            out[i, j] = np.hypot(*data.qvel[:2]), data.qvel[2]
    return out


def calibration_metrics(w, prediction):
    y,u,_=w
    result={}
    for name,mask in {
        'all': np.ones(len(y),bool),
        'steering': np.max(abs(u[:,:,0]),axis=1)>.05,
        'braking': np.max(u[:,:,2],axis=1)>.25,
        'combined': ((abs(u[:,:,0])>.05)&(u[:,:,2]>.25)).any(1),
    }.items():
        if not mask.any():
            result[name]=dict(windows=0)
            continue
        target=y[mask,-1]
        rmse=lambda q: np.sqrt(np.mean((q-target)**2,axis=0)).tolist()
        result[name]=dict(windows=int(mask.sum()), fitted_rmse=rmse(prediction[mask,-1]),
                          persistence_rmse=rmse(y[mask,0]))
    return result


def fit(train,test):
    baseline=np.array([4.,8.,1.,-.24,1.,7.])
    bound=([.5,2.,0.,-1.,0.,1.],[12.,18.,6.,-.01,5.,22.])
    def residual(p):
        err=(simulate_windows(train,p)-train[0][:,1:])*[1.,12.]
        return err.ravel()
    sol=least_squares(residual,baseline,bounds=bound,loss='soft_l1',max_nfev=100)
    metrics=calibration_metrics(test,simulate_mujoco_windows(test,sol.x))
    prior=calibration_metrics(test,simulate_mujoco_windows(test,baseline))
    accepted=sol.success and all(metrics[group]['windows']>=20 and
            metrics[group]['fitted_rmse'][j] < min(metrics[group]['persistence_rmse'][j],
                                                  prior[group]['fitted_rmse'][j])
            for group,j in [('steering',1),('braking',0)])
    return sol.x,dict(accepted=bool(accepted),optimizer_success=bool(sol.success),
                     train_windows=len(train[0]),heldout=metrics,unfitted=prior,
                     parameter_names=['drive_m_s2','brake_m_s2','drag_m_s2_at_25ms',
                                      'steer_gain','understeer','yaw_response_per_s'])


XML = """<mujoco><option timestep=".04" gravity="0 0 0" integrator="Euler"/>
<worldbody><body><joint name="x" type="slide" axis="1 0 0"/>
<joint name="y" type="slide" axis="0 1 0"/><joint name="yaw" type="hinge" axis="0 0 1"/>
<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
</body></worldbody></mujoco>"""


def init_worker():
    global mj, model, data
    import mujoco as mj
    model = mj.MjModel.from_xml_string(XML)
    data = mj.MjData(model)


def advance(request, p, grip, interval):
    # Integrate the same force model in validation and policy training.
    count = max(1, int(np.ceil(interval/.02)))
    model.opt.timestep = interval/count
    for _ in range(count):
        angle = data.qpos[2]
        vx, vy, yaw_rate = data.qvel
        v = max(np.hypot(vx, vy), .01)
        ax, ar = response(np.array([v, yaw_rate]), np.array(request), p, grip)
        forward = np.array([np.cos(angle), np.sin(angle)])
        lateral = np.array([-forward[1], forward[0]])
        ay = np.clip(v*yaw_rate-5*(np.array([vx, vy])@lateral), -grip*G, grip*G)
        scale = min(1., grip*G/max(np.hypot(ax, ay), 1.e-8))
        data.qfrc_applied[:2] = scale*(ax*forward+ay*lateral)
        data.qfrc_applied[2] = ar
        mj.mj_step(model,data)


def corner(job):
    gain,p,seed,grip,radius,side=job
    rng=np.random.default_rng(seed)
    mj.mj_resetData(model,data)
    data.qpos[:]=[0.,rng.uniform(-.35,.35),rng.uniform(-.035,.035)]
    data.qvel[0]=rng.uniform(15,20)
    # Reference is privileged in this plant/controller experiment.
    s=np.arange(0.,105.,.5)
    heading=side*np.clip((s-22.)/radius,0.,.95)
    ref=np.vstack([[0.,0.],np.cumsum(np.c_[np.cos(heading[:-1]),np.sin(heading[:-1])]*.5,axis=0)])
    track_error=[]
    speeds=[]
    reason='time_limit'
    best=0
    for i in range(420):
        x,y,angle=data.qpos
        vx,vy,yaw_rate=data.qvel
        v=max(np.hypot(vx,vy),.01)
        speeds.append(v)
        idx=int(np.argmin(np.sum((ref-[x,y])**2,axis=1)))
        best=max(best,idx)
        tangent=heading[idx]
        ey=float(np.array([-np.sin(tangent),np.cos(tangent)]) @ (np.array([x,y])-ref[idx]))
        error=(tangent-angle+np.pi)%(2*np.pi)-np.pi
        track_error.append(ey)
        if abs(ey)>2.2:
            reason='track_exit'
            break
        if idx>=len(s)-4:
            reason='completion'
            break
        ahead=min(idx+int(gain[0]),len(s)-1)
        upcoming=(heading[ahead]-tangent)/max((ahead-idx)*.5,1.)
        wanted=gain[1]*error-gain[2]*ey+gain[3]*upcoming
        target_speed=np.clip(gain[4]/np.sqrt(abs(upcoming)*35+1),9.,24.)
        power=np.clip(gain[5]*(target_speed-v),-1.,1.)
        # The policy emits the entire request. No post-policy pedal override.
        request=[np.clip(wanted/p[3],-1.,1.),max(power,0),max(-power,0)]
        advance(request, p, grip, .04)
    return dict(reason=reason,steps=i+1,progress_m=float(s[best]),
                rms_m=float(np.sqrt(np.mean(np.square(track_error)))),grip=grip,seed=seed,radius=radius,side=side,
                speed_range_m_s=[min(speeds),max(speeds)], worker_pid=os.getpid())


def jobs(gain,p,seeds,grips,radii):
    return [(gain,p,seed,grip,radius,side)
            for seed,grip,radius,side in product(seeds,grips,radii,(-1,1))]


def train_policy(p,workers,seed):
    rng=np.random.default_rng(seed)
    mean=np.array([18.,1.5,.14,7.,21.,.25])
    sigma=np.array([5.,.5,.06,3.,3.,.1])
    lo=[6.,.4,.02,0.,13.,.06]
    hi=[32.,3.,.45,17.,25.,.65]
    initial=mean.copy()
    history=[]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        best_score=-np.inf
        for _ in range(5):
            variants=np.clip(rng.normal(mean,sigma,(10,len(mean))),lo,hi)
            variants[0]=mean
            batches=[jobs(v,p,range(2),(.65,.9,1.15),(27.,36.,52.)) for v in variants]
            trajectories=list(pool.map(corner,[j for batch in batches for j in batch],chunksize=3))
            count=len(batches[0])
            values=np.array([np.mean([r['progress_m']-6*r['rms_m']-.01*r['steps']+
                    (10 if r['reason']=='completion' else -15)
                    for r in trajectories[i*count:(i+1)*count]]) for i in range(10)])
            rank=np.argsort(values)[-3:]
            if values[rank[-1]]>best_score:
                best_score=values[rank[-1]]
                best=variants[rank[-1]].copy()
            mean=variants[rank].mean(0)
            sigma=np.maximum(sigma*.75,variants[rank].std(0))
            history.append(dict(best=float(values.max()),completed=sum(r['reason']=='completion' for r in trajectories),
                                episodes=len(trajectories),worker_pids=sorted({r['worker_pid'] for r in trajectories}),reasons={k:sum(r['reason']==k for r in trajectories)
                                for k in sorted({r['reason'] for r in trajectories})}))
        args=lambda gain:jobs(gain,p,range(100,103),(.72,1.02,1.21),(31.,45.,56.))
        baseline=list(pool.map(corner,args(initial),chunksize=2))
        learned=list(pool.map(corner,args(best),chunksize=2))
    return dict(workers=workers,seed=seed,elapsed_s=time.perf_counter()-start,
                initial=initial.tolist(),learned=best.tolist(),training=history,
                baseline_heldout=baseline,learned_heldout=learned,
                scope='synthetic corner with privileged reference and grip perturbations')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--test',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=35)
    args=parser.parse_args()
    if args.workers<1 or args.train.resolve()==args.test.resolve():
        raise ValueError('Positive worker count and distinct AC recordings required')
    args.output.mkdir(parents=True,exist_ok=False)
    t,u,train_sha=load_trace(args.train)
    q,v,test_sha=load_trace(args.test)
    if train_sha==test_sha:
        raise ValueError('Identical recording bytes')
    params,result=fit(windows(t,u),windows(q,v))
    result.update(parameters=params.tolist(),speed_range_m_s=SPEED_RANGE,program_sha256=sha256(__file__),
                  train_sha256=train_sha,test_sha256=test_sha,
                  train_source=args.train.as_posix(),test_source=args.test.as_posix(),
                  validation_backend="MuJoCo planar force integration")
    write_json(args.output/'calibration.json',result)
    if not result['accepted']:
        raise RuntimeError('AC response gate failed; corner learning did not start')
    out=train_policy(params,args.workers,args.seed)
    write_json(args.output/'result.json',dict(status='completed',calibration_sha256=sha256(args.output/'calibration.json'),
               **out))


if __name__=='__main__':
    main()
