import pytest
from flybrain_drive.physical_timing_intervals import TimeBound,compare_physical_timing

def bound(a,b=None):return TimeBound(a,a if b is None else b,'host_qpc_seconds','original-evidence')
def compare(**kwargs):
    args=dict(decision=bound(10.,10.002),views=(bound(9.97,9.971),)*4,body=bound(9.99,9.991),action=bound(10.04,10.041),maximum_view_age=.15,maximum_body_age=.05,minimum_action_delay=.015,maximum_action_delay=.065)
    args.update(kwargs);return compare_physical_timing(**args)

def test_bounds_can_be_consistent_without_granting_training():
    result=compare();assert result['status']=='consistent_bounded_timing_only';assert result['training_admissible'] is False and result['evidence_authenticated'] is False

def test_unknown_physical_sample_is_not_replaced_with_host_receipt():
    result=compare(body=None,views=(None,bound(9.97),bound(9.97),bound(9.97)))
    assert result['status']=='unresolved_missing_physical_bounds' and result['missing_physical_bounds']==['front','body']

def test_reject_possible_future_sensor_or_unbounded_target():
    assert compare(body=bound(9.99,10.001))['status']=='rejected_sensor_interval'
    assert compare(action=bound(10.00,10.05))['status']=='rejected_action_interval'
    assert compare(views=(bound(9.80),)*4)['status']=='rejected_sensor_interval'

def test_reject_wrong_clock_and_boolean_time():
    with pytest.raises(ValueError):TimeBound(1.,2.,'producer_clock','x')
    with pytest.raises(ValueError):TimeBound(True,2.,'host_qpc_seconds','x')
