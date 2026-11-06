"""Diagnostic sustained body-turn windows; not actor features or map truth."""
import numpy as np

def steering_yaw_alignment(training):
    agreements=[]
    for d in training:
        m=d['fit_mask']&(abs(d['body'][:,4])>.15)&(abs(d['labels'][:,0])>.03)
        agreements.extend((np.sign(d['body'][m,4])*np.sign(d['labels'][m,0])).tolist())
    v=np.asarray(agreements)
    if len(v)<100:raise ValueError('Insufficient TRAIN sign evidence')
    sign=int(np.sign(v.mean()));agreement=float(np.mean(v==sign))
    if agreement<.9:raise ValueError('Ambiguous empirical steering/yaw sign')
    return sign,dict(training_only=True,count=len(v),agreement=agreement,sign=sign,
        qualification='Empirical diagnostic sign mapping, not independent actuator calibration')

def sustained_onsets(d,alignment):
    """Detect transitions independently of whether an opposite donor exists."""
    ids=np.flatnonzero(d['fit_mask']);t=d['decision_time'];yaw=d['body'][:,4];last=-1e30;events=[]
    for k,j in enumerate(ids):
        if k<25 or k>=len(ids)-15 or t[j]-last<2 or abs(yaw[j])<.15:continue
        prior=ids[(t[ids]>=t[j]-.6)&(t[ids]<t[j])]
        after=ids[(t[ids]>=t[j])&(t[ids]<t[j]+.6)]
        if len(prior)<10 or len(after)<10 or np.mean(abs(yaw[prior]))>=.07:continue
        median=np.median(yaw[after]);direction=int(np.sign(median))
        if abs(median)<.15 or np.mean(np.sign(yaw[after])==direction)<.8:continue
        events.append(dict(event=int(j),start=int(np.searchsorted(t,t[j]-1.2)),time=float(t[j]),
            direction=float(direction*alignment),post_yaw_median=float(median)))
        last=t[j]
    return events


def paired_sustained_onsets(d,alignment):
    events=sustained_onsets(d,alignment)
    out=[]
    for w in events:
        candidates=[z for z in events if z['direction']!=w['direction']]
        if not candidates:continue
        z=min(candidates,key=lambda z:abs(d['body'][z['start'],16]-d['body'][w['start'],16]))
        out.append(dict(**w,donor_event=z['event'],donor_time=z['time']))
    return out
