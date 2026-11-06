from types import SimpleNamespace
import numpy as np
import pytest
from flybrain_drive.ac_geometric_teacher import GeometricRecoveryTeacher
from flybrain_drive.teacher_terminal_stop import lateral_request,run_terminal_stop

LINE=np.array([[-100.,0.],[0.,0.],[100.,0.],[100.,100.],[0.,100.],[-100.,100.]])

def reading(packet=1,speed=0.,out=0):
    return SimpleNamespace(native_packet_id=packet,state=dict(world_position_x=0.,world_position_y=-1.,yaw=0.,gap=1.,speed=speed/3.6,numberOfTyresOut=out,actualGear=2,brakeStatus=1.))

class Session:
    def __init__(self):self.requests=[]
    def submit(self,*values):self.requests.append(values);return dict(requested_control=dict(zip(['steering','throttle','brake'],values)))

def run(rows,source_age=.01):
    clock=[100.];session=Session();events=[];teacher=GeometricRecoveryTeacher(LINE,-.1)
    def sleep(v):clock[0]+=v
    iterator=iter(rows)
    result=run_terminal_stop(session,reading(speed=20),lambda:next(iterator),teacher,lambda:None,lambda:source_age,
        lambda kind,**kw:events.append(dict(kind=kind,**kw)),steering_request_scale=.6,state_guard=lambda row:None,clock=lambda:clock[0],sleep=sleep)
    return session,events,result

def test_teacher_lateral_retains_sac_pedals_and_has_explicit_privileged_correction():
    original=np.array([.1,-.8,-1.]);changed=lateral_request(original,reading().state,GeometricRecoveryTeacher(LINE,-.1))
    assert changed[0]<0 and np.array_equal(changed[1:],original[1:])
    np.testing.assert_array_equal(original,[.1,-.8,-1.])

def test_stop_requires_three_distinct_stationary_brake_feedbacks_before_release():
    session,events,result=run([reading(1,10),reading(2,.2),reading(3,.1),reading(4,.1)])
    assert result['requests']==4 and result['distinct_stationary_feedback']==3
    assert all(t==0 and b==1 for s,t,b in session.requests)
    assert all(abs(s)<=.21 for s,t,b in session.requests)
    assert len([e for e in events if e['kind']=='teacher_stop_feedback'])==4

@pytest.mark.parametrize('age',[-.001,.050001,float('nan')])
def test_source_age_is_never_relaxed_for_terminal_braking(age):
    with pytest.raises(TimeoutError):run([],age)

def test_track_exit_and_repeated_packet_reject_terminal_completion():
    with pytest.raises(ValueError,match='exit'):run([reading(out=1)])
    with pytest.raises(ValueError,match='Distinct'):run([reading(1),reading(1)])

def test_stationary_count_resets_after_speed_rises_and_three_new_packets_required():
    session,events,result=run([reading(1,.1),reading(2,1.),reading(3,.1),reading(4,.1),reading(5,.1)])
    assert result['requests']==5

def test_no_stationary_feedback_rejects_at_finite_three_second_budget():
    with pytest.raises(TimeoutError,match='3s'):run([reading(i,20) for i in range(1,100)])
