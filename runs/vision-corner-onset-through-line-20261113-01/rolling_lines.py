"""Plan curvature-limited rolling lines from archived body-frame road sections."""
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from numpy.polynomial import Polynomial

from dense_onset import ARCHIVE, CLI, EVAL, OUT, VAL, frames, load_observer, old
from flybrain_drive.local_racing_line import plan_local_line
from replay_corner_boundary_states import alt, sensor_rows, sequence
from plan_frozen_corner_archive import LIMITS

STEP = .25
MAX_GAP = 4.01
MIN_PREVIEW = 8.
MAX_NEAR = 8.
MIN_OVERLAP = 5.
MAX_SEPARATION = 1.


def corridor(stations, edges):
    width = edges[:,1]-edges[:,0]
    valid = np.isfinite(edges).all(1)&(width>2*LIMITS["clearance_m"])&(width<20.)
    index = np.flatnonzero(valid)
    if not len(index):
        return None
    end = np.flatnonzero(np.diff(stations[index])>MAX_GAP)
    if len(end):
        index=index[:end[0]+1]
    if len(index)<4 or stations[index[0]]>MAX_NEAR or np.ptp(stations[index])<MIN_PREVIEW:
        return None
    return stations[index],edges[index]


def valid_spline(curve, x, lower, upper):
    for i, dx in enumerate(np.diff(x)):
        segment = Polynomial(curve.c[:,i][::-1])
        first, second, third = (segment.deriv(k) for k in (1,2,3))
        extreme = third*(1.+first*first)-3.*first*second*second
        def stations(poly):
            return [0., dx, *(z.real for z in poly.roots() if abs(z.imag)<1.e-8 and 0.<z.real<dx)]
        for bound, sign in ((lower,1.),(upper,-1.)):
            gap = sign*(segment-Polynomial([bound[i],(bound[i+1]-bound[i])/dx]))
            if np.min(gap(stations(gap.deriv()))) < -1.e-8:
                return False
        t = np.asarray(stations(extreme))
        if np.max(abs(second(t)/(1.+first(t)**2)**1.5))>LIMITS["max_curvature"]:
            return False
    return True


def rolling_line(section, previous, forecast, baseline):
    """Generate smooth feasible samples using curvature and agreement with the preceding plan."""
    if section is None:
        return None
    x,edges=section
    right,left=edges.T
    margin=LIMITS["clearance_m"]
    lower,upper=right+margin,left-margin
    if np.any(upper-lower<=2.e-6):
        return baseline
    middle=(lower+upper)*.5
    span=(upper-lower)*.5
    grid=np.unique(np.r_[np.arange(x[0],x[-1],STEP),x])
    lower_grid,upper_grid=np.interp(grid,x,lower),np.interp(grid,x,upper)
    targets=[middle, middle-.2*span,middle+.2*span]
    bend=forecast["state"]
    onset=forecast["distance_m"]
    if bend in ("left","right") and x[0]<=onset<=x[-1]-4.:
        # Arc and forward distance diverge at large lateral offsets; map the
        # predicted arc distance onto the observed centerline.
        arc=np.r_[0.,np.cumsum(np.hypot(np.diff(np.r_[0.,x]),np.diff(np.r_[0.,middle])))]
        start=np.interp(onset,arc,np.r_[0.,x])
        spline=CubicSpline(x,middle)
        slope=spline(x,1)
        k=spline(x,2)/(1+slope**2)**1.5
        direction=1 if bend=="left" else -1
        usable=(x>=start)&(direction*k>=.002)
        if np.count_nonzero(usable)>=2:
            apex=x[np.argmax(np.where(usable,direction*k,-np.inf))]
            if apex>start and apex<x[-1]:
                pattern=np.interp(x,[x[0],start,apex,x[-1]],[0.,-.45,.45,-.30])
                targets.append(middle+direction*span*pattern)

    shared=np.zeros(len(grid),bool)
    carried=None
    if previous is not None and len(previous)>1 and np.all(np.diff(previous[:,0])>0):
        shared=(grid>=previous[0,0])&(grid<=previous[-1,0])
        if shared.any() and np.ptp(grid[shared])>=MIN_OVERLAP:
            carried=np.interp(grid,previous[:,0],previous[:,1])
            target=np.interp(x,previous[:,0],previous[:,1])
            taper=np.minimum(1.,np.maximum(0.,previous[-1,0]-x)/4.)
            taper*=x>=previous[0,0]
            for amount in (.5,.8,1.):
                targets.append(np.clip(middle+amount*taper*(target-middle),lower,upper))
    # Fit a smooth center trajectory jointly across the corridor; a previous
    # trajectory is only a preference inside the currently supported interval.
    regularizer = np.diff(np.eye(len(x)),n=2,axis=0)
    weights = np.ones(len(x))
    desired = middle.copy()
    if carried is not None:
        supported = (x>=previous[0,0]) & (x<=previous[-1,0])
        weights[supported] = 3.
        desired[supported] = (middle[supported]+2*np.interp(x[supported],previous[:,0],previous[:,1]))/3
    design = np.vstack((np.diag(weights),4.*regularizer))
    response = np.r_[weights*desired,np.zeros(len(regularizer))]
    optimum = lsq_linear(design,response,bounds=(lower+1.e-6,upper-1.e-6),method="bvls")
    if optimum.success:
        targets.append(optimum.x)
    lines=[]
    if baseline is not None:
        bx, by, bk = (np.asarray(baseline[name]) for name in ("x_m","y_m","curvature_inv_m"))
        y = np.interp(grid,bx,by)
        deviation=float(np.max(abs(y[shared]-carried[shared]))) if carried is not None else 0.
        cost=float(np.mean(np.interp(grid,bx,bk)**2))
        lines.append(((deviation>MAX_SEPARATION,cost+.002*deviation**2),
                      dict(baseline,source="ego_cubic",overlap_m=float(np.ptp(grid[shared])) if carried is not None else None,
                           overlap_max_delta_m=deviation if carried is not None else None)))
    for target in targets:
        curve=CubicSpline(x,target,bc_type="natural")
        y,slope,second=curve(grid),curve(grid,1),curve(grid,2)
        curvature=second/(1+slope**2)**1.5
        if (np.any(y<lower_grid-1.e-8) or np.any(y>upper_grid+1.e-8)
                or np.max(abs(curvature))>LIMITS["max_curvature"]):
            continue
        arc=np.r_[0.,np.cumsum(np.hypot(np.diff(grid),np.diff(y)))]
        deviation=float(np.max(abs(y[shared]-carried[shared]))) if carried is not None else 0.
        smooth=float(np.trapezoid(curvature**2,arc)/arc[-1])
        lines.append(((deviation>MAX_SEPARATION,smooth+.002*deviation**2),
                      dict(x_m=grid.tolist(),y_m=y.tolist(),arc_m=arc.tolist(),
                           curvature_inv_m=curvature.tolist(),first_constrained_x_m=float(x[0]),source="corridor_spline",
                           overlap_m=float(np.ptp(grid[shared])) if carried is not None else None,
                           overlap_max_delta_m=deviation if carried is not None else None)))
    for _, candidate in sorted(lines, key=lambda v:v[0]):
        if candidate["source"]=="ego_cubic":
            return candidate
        sampled = np.asarray(candidate["y_m"])
        curve = CubicSpline(x,sampled[np.searchsorted(grid,x)],bc_type="natural")
        if valid_spline(curve,x,lower,upper):
            return candidate
    return None


def classic(section,speed):
    if section is None:
        return None
    x,edges=section
    sections=[dict(forward_x_m=s,right_y_m=r,left_y_m=l,valid=True)
              for s,(r,l) in zip(x,edges,strict=True)]
    return plan_local_line(sections,speed,**LIMITS)[0]


def run():
    _,settings,sources=load_observer()
    onset=old.read(OUT/"onset-result.json")
    output={}
    for track,name in EVAL:
        root=CLI/("vallelunga-retry" if track==VAL else name)
        path=root/"recovered-sensors.npz"
        assert old.digest(path)==old.read(root/"result.json")["recovered_input_sha256"]
        with np.load(path,allow_pickle=False) as data:
            rows=sensor_rows(data,settings)
            stations,edges,body=[data[k] for k in ("stations","sections","motion")]
        trace=OUT/f"{track}-onset.jsonl"
        turns=frames(trace,onset["tracks"][track]["forecasts_sha256"])
        assert [(a["run"],a["frame"]) for a in turns]==[(a["run"],a["frame"]) for a in rows]
        continuous=[False,*(alt.adjacent(a,b) for a,b in zip(rows,rows[1:]))]
        aligned=sequence(stations,continuous,edges,body)
        output_rows=[]
        previous={"geometry":None,"preview":None}
        for i,row in enumerate(rows):
            moving=np.isfinite(body[i]).all() and body[i,1]>=old.MIN_SPEED_MPS
            section=corridor(stations,aligned[i]) if moving else None
            step=alt.transform(body[i],body[i-1]) if i and moving and continuous[i] and np.isfinite(body[i-1]).all() else None
            forecasts=turns[i]["forecasts"]
            variants={"classic":classic(section,body[i,1]),}
            for method in previous:
                prior=alt.advance(previous[method],*step) if step is not None and previous[method] is not None else None
                forecast=forecasts["t4t5_confirm"] if method=="preview" else dict(state="abstain",distance_m=None)
                line=rolling_line(section,prior,forecast,variants["classic"])
                variants[method]=line
                previous[method]=np.column_stack((line["x_m"],line["y_m"])) if line is not None else None
            output_rows.append(dict(**row,forecast=forecasts["t4t5_confirm"],lines=variants))
        witness=OUT/f"{track}-lines.jsonl"
        old.jsonl(witness,output_rows)
        count={m:sum(r["lines"][m] is not None for r in output_rows) for m in ("classic","geometry","preview")}
        print(track,count,flush=True)
        output[track]=dict(frames=len(rows),coverage=count,lines_sha256=old.digest(witness),
                           source_sha256={str(p):old.digest(p) for p in (path,trace)})
    sources.update((Path(__file__),Path(plan_local_line.__code__.co_filename),Path(sequence.__code__.co_filename),
                    Path(alt.transform.__code__.co_filename),OUT/"onset-result.json"))
    old.write(OUT/"line-result.json",dict(tracks=output,limits=LIMITS,sample_step_m=STEP,maximum_gap_m=MAX_GAP,
              minimum_preview_m=MIN_PREVIEW,source_revision=old.revision(),seed=None,
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="archive_end"))


if __name__=="__main__":
    run()
