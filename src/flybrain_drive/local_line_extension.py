"""Extend a 20 m local line through continuously observed far sections."""
import numpy as np
from numpy.polynomial import Polynomial


def extend_line(line, sections, limits):
    if line is None or line["x_m"][-1] != 20.:
        return None
    usable = []
    anchor_width = previous_width = None
    for section in sections:
        station = section["forward_x_m"]
        if station < 20.:
            continue
        if not section["valid"]:
            break
        width = section["left_y_m"] - section["right_y_m"]
        if previous_width is None:
            anchor_width = width
        elif width > previous_width + 1.5 or width > anchor_width + 2.5:
            break
        usable.append(section)
        previous_width = width
    if len(usable)<2 or usable[0]["forward_x_m"]!=20.:
        return None
    x,right,left=np.asarray([[s["forward_x_m"],s["right_y_m"],s["left_y_m"]] for s in usable]).T
    clearance=limits["clearance_m"]
    lower,upper=right+clearance,left-clearance
    if np.any(lower>upper):
        return None
    length=x[-1]-20.
    stations=(x-20.)/length
    old_x,old_y=map(np.asarray,(line["x_m"],line["y_m"]))
    curve=Polynomial.fit(old_x/20.,old_y,4,domain=[0.,1.]).convert()
    start=Polynomial([old_y[-1],curve.deriv()(1.)*length/20.,
                      curve.deriv(2)(1.)*(length/20.)**2/2.])
    center=.5*(left+right)
    final_slope=(center[-1]-center[-2])/(x[-1]-x[-2])
    u=np.unique(np.r_[np.linspace(0.,1.,int(np.ceil(length/.25))+1),stations])
    choices=[]
    for end in np.linspace(lower[-1],upper[-1],5):
        for slope in final_slope+np.array([-.1,0.,.1]):
            a,b=np.linalg.solve([[1.,1.],[3.,4.]],[end-start(1.),length*slope-start.deriv()(1.)])
            candidate=start+Polynomial([0.,0.,0.,a,b])
            bounded=True
            for j,(lo,hi) in enumerate(zip(stations,stations[1:])):
                for edge,sign in ((lower,1.),(upper,-1.)):
                    gradient=(edge[j+1]-edge[j])/(hi-lo)
                    gap=sign*(candidate-Polynomial([edge[j]-gradient*lo,gradient]))
                    roots=gap.deriv().roots()
                    points=np.array([lo,hi,*(z.real for z in roots if abs(z.imag)<1e-9 and lo<z.real<hi)])
                    bounded &= bool(np.min(gap(points))>=-1e-9)
            if not bounded:
                continue
            d1,d2,d3=(candidate.deriv(k)/length**k for k in (1,2,3))
            critical=(d3*(1+d1*d1)-3*d1*d2*d2).roots()
            points=np.array([0.,1.,*(z.real for z in critical if abs(z.imag)<1e-9 and 0<z.real<1)])
            kappa=lambda t:d2(t)/(1+d1(t)**2)**1.5
            peak=float(np.max(abs(kappa(points))))
            if peak>limits["max_curvature"]:
                continue
            choices.append((peak,abs(end-center[-1]),candidate(u[1:]),kappa(u[1:])))
    if not choices:
        return None
    _,_,extension,kappa=min(choices,key=lambda entry:entry[:2])
    new_x=np.r_[old_x,20.+length*u[1:]]
    new_y=np.r_[old_y,extension]
    curvature=np.r_[line["curvature_inv_m"],kappa]
    arc=np.r_[0.,np.cumsum(np.hypot(np.diff(new_x),np.diff(new_y)))]
    braking=limits["braking_mps2"]
    cap=np.minimum(limits["speed_ceiling_mps"]**2,limits["lateral_accel_mps2"]/np.maximum(abs(curvature),1e-12))
    cap=np.minimum.accumulate((cap+2*braking*arc)[::-1])[::-1]-2*braking*arc
    speed=np.sqrt(cap)
    time=float(np.sum(2*np.diff(arc)/(speed[1:]+speed[:-1])))
    return dict(line,x_m=new_x.tolist(),y_m=new_y.tolist(),arc_m=arc.tolist(),
                curvature_inv_m=curvature.tolist(),speed_bound_mps=speed.tolist(),envelope_time_score_s=time)
