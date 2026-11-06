"""Privileged finite teacher stabilization/stop; never an actor sensor or fly circuit."""
import math
import time
import numpy as np
from .ac_geometric_teacher import GeometricRecoveryTeacher


def lateral_request(requested, state, geometric):
    if (not isinstance(geometric,GeometricRecoveryTeacher)
            or not 0<=state['speed']*3.6<=40 or state['numberOfTyresOut']!=0):
        raise ValueError('Teacher-only lateral stabilization requires speed<=40 and zero tyre exits')
    result=np.asarray(requested,dtype=float).copy()
    if result.shape!=(3,) or not np.isfinite(result).all():raise ValueError('Original finite SAC request required')
    result[0]=geometric.request(state)[0]
    return result


def run_terminal_stop(session,current,receive,geometric,guard,source_guard,emit,*,
                      steering_request_scale,state_guard,maximum_seconds=3.,clock=time.perf_counter,sleep=time.sleep):
    if not (0<steering_request_scale<=1 and maximum_seconds==3.):
        raise ValueError('Explicit calibrated steering and fixed three-second stop budget required')
    started=clock();deadline=started;requests=0;stopped=0;last_packet=None;feedback=[]
    while clock()-started<maximum_seconds:
        guard()
        state_guard(current);state=current.state
        if (not np.isfinite([state['speed'],state['gap']]).all() or not 0<=state['speed']*3.6<=40
                or abs(state['gap'])>3 or state['numberOfTyresOut']!=0 or state['actualGear']<1):
            raise ValueError('Teacher stopping envelope exceeded')
        steering=float(geometric.request(state)[0]*steering_request_scale)
        guard();age=source_guard()
        if not isinstance(age,(float,int)) or not math.isfinite(age) or not 0<=age<=.05:raise TimeoutError('Original source capture exceeds50ms at stopping request')
        request=session.submit(steering,0.,1.);requests+=1
        emit('teacher_stop_request',request=request,source_capture_age_before_actuation_ms=age*1000,
             measured_speed_before_kmh=state['speed']*3.6,teacher_only=True,
             lateral_diagnostics=geometric.last_diagnostics)
        deadline+=.04;remaining=deadline-clock()
        if remaining<=0:raise TimeoutError('Teacher stopping request exceeded40ms budget')
        sleep(remaining);guard();current=receive();state_guard(current);state=current.state
        if state['numberOfTyresOut']!=0:raise ValueError('Native/SDK tyre exit during stopping feedback')
        if current.native_packet_id==last_packet:raise ValueError('Distinct stopping feedback packet required')
        last_packet=current.native_packet_id
        if clock()-started>maximum_seconds:raise TimeoutError('Teacher stopping feedback exceeded3s boundary')
        row=dict(packet=last_packet,speed_kmh=state['speed']*3.6,brake=state['brakeStatus'],tyres_out=state['numberOfTyresOut'])
        feedback.append(row);emit('teacher_stop_feedback',**row)
        stopped=stopped+1 if row['speed_kmh']<=.5 and row['brake']>=.98 else 0
        if stopped>=3:
            return dict(status='completed_teacher_feedback_stop',requests=requests,feedback=feedback,
                elapsed_seconds=clock()-started,distinct_stationary_feedback=stopped,
                maximum_seconds=maximum_seconds,student_actuation=False,actor_input=False)
    raise TimeoutError('Teacher did not reach verified stationary brake hold within3s')
