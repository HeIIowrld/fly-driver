"""Offline comparison against past accepted teacher commands, never actor input."""
import bisect,math
import numpy as np


def accepted_teacher_commands(events,steering_scale):
    if not math.isfinite(steering_scale) or not 0<steering_scale<=1:raise ValueError('Explicit positive canonical steering scale required')
    commands=[]
    for i,event in enumerate(events):
        if event.get('kind')!='teacher_request':continue
        request=event['request'];control=request['requested_control']
        start=request['driver_call_started_perf_counter_seconds'];finished=request['driver_call_finished_perf_counter_seconds']
        decision=event['decision_started_host_time']
        target=[control['steering']/steering_scale,control['throttle'],control['brake']]
        if (request['driver_accepted'] is not True or not all(math.isfinite(v) for v in [decision,start,finished,*target])
                or not decision<=start<=finished or abs(target[0])>1+1e-8 or any(not 0<=v<=1 for v in target[1:])
                or commands and finished<=commands[-1]['finished']):raise ValueError('Ordered finite accepted canonical teacher commands required')
        commands.append(dict(event_index=i,finished=finished,decision_time=decision,target=target,stage=event['stage'],game_applied_verified=request.get('game_applied') is True))
    if not commands:raise ValueError('No accepted normal curriculum commands')
    return commands


def pair_past_commands(decision_times,commands,*,maximum_age_seconds):
    times=np.asarray(decision_times,dtype=float)
    if (times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or not math.isfinite(maximum_age_seconds) or not 0<maximum_age_seconds<=.06):raise ValueError('Original increasing decisions and bounded command age required')
    ends=[c['finished'] for c in commands];pairs=[]
    for t in times:
        index=bisect.bisect_right(ends,float(t))-1
        if index<0:pairs.append(dict(eligible=False,reason='no_past_accepted_command'));continue
        age=float(t)-ends[index]
        pairs.append(dict(eligible=age<=maximum_age_seconds,reason=None if age<=maximum_age_seconds else 'past_command_stale',command_index=index,age_seconds=age))
    return pairs


def discrepancy(predictions,targets,command_indices):
    p,y=np.asarray(predictions,float),np.asarray(targets,float);ids=np.asarray(command_indices)
    if p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=3 or not len(p) or ids.shape!=(len(p),) or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Finite aligned canonical comparison required')
    error=p-y;_,inverse,counts=np.unique(ids,return_inverse=True,return_counts=True);weights=1/counts[inverse];weights/=weights.sum()
    directional=(np.abs(y[:,0])>=.05)&(np.abs(p[:,0])>=.05)
    return dict(events=len(p),unique_teacher_commands=len(counts),rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        command_balanced_rmse=np.sqrt(np.sum(weights[:,None]*error**2,axis=0)).tolist(),mae=np.mean(abs(error),axis=0).tolist(),
        material_steering_events=int(directional.sum()),opposite_steering_events=int(((p[:,0]*y[:,0]<0)&directional).sum()),
        target_mean=y.mean(axis=0).tolist(),prediction_mean=p.mean(axis=0).tolist(),
        simultaneous_pedals_above_point1=int(((p[:,1]>.1)&(p[:,2]>.1)).sum()))
