"""Pure/native-byte fixtures; no live readers, clock calls, vehicle or model."""

import copy
import json
import struct

import pytest

from scripts.transit_to_train_start_v2 import (
    check_child,
    check_idle,
    coordinate_link,
    decode_observer,
    execute_stages,
    stable_native_copy,
    transit_decision,
    validate_pair,
)


def plan():
    return dict(generation='fixture', epoch=0, engine_pid=24200,
                expected_initial_spline=.09588946402072906)


def idle():
    return dict(generation='fixture', epoch=0, window_handle='733364ULL',
        car_id='bmw_z4_gt3', track_id='monza', paused=True, ai_controlled=False,
        is_online=False, speed_kmh=.01, gas=0., brake=1., script_clock=12.)


def child():
    return dict(status='completed_diagnostic_transport_collection', role='diagnostic',
        fit_allowed=False, heartbeat_evidence=dict(status='complete'),
        transport_state='closed', transport_error=None, owned_threads_still_live=[],
        lease_retained=False, handoff=dict(same_writer_joined=True, lease_removed=True,
                                         original_source_clock=20.))


def test_normal_status_without_route_is_the_actual_supported_idle_schema():
    assert 'diagnostic_route' not in idle()
    assert check_idle(idle(), .1, plan(), 733364)
    assert not check_idle(idle(), .1, plan(), 733364, after_source=12.)
    assert check_idle(idle(), .1, plan(), 733364, after_source=11.9)


@pytest.mark.parametrize(('field','value'), [
    ('generation','foreign'),('epoch',1),('paused',False),('ai_controlled',True),
    ('is_online',True),('camera_active',True),('window_handle','foreign'),
    ('speed_kmh',float('nan')),('speed_kmh',.1),('gas',.02),('brake',.98),
    ('script_clock',float('nan')),
])
def test_unknown_or_nonidle_csp_state_fails_without_route_fallback(field,value):
    value_state = idle()
    value_state[field] = value
    with pytest.raises(ValueError):
        check_idle(value_state, .1, plan(), 733364)


def graphics(*, packet=20, state=3, laps=0, spline=.3):
    raw = bytearray(264)
    struct.pack_into('<iii', raw, 0, packet, state, 0)
    struct.pack_into('<i', raw, 132, laps)
    struct.pack_into('<f3f', raw, 248, spline, 1.,2.,3.)
    return bytes(raw)


def test_native_observer_uses_exact264_offset248_and_separate252_coordinates():
    value = decode_observer(graphics(), 100.)
    assert abs(value['spline_position']-.3) < 2e-8
    assert value['coordinates_native_xyz_m'] == [1.,2.,3.]
    assert value['graphics_packet'] == 20 and value['graphics_status'] == 'pause'
    assert value['received_perf_counter_seconds'] == 100.
    assert not value['actor_input'] and not value['exact_physical_sample_time_verified']


@pytest.mark.parametrize('raw', [graphics(state=2),graphics(laps=1),
    graphics(spline=float('nan')),graphics(spline=1.),graphics()[:260]])
def test_native_observer_rejects_other_states_or_invalid_raw(raw):
    with pytest.raises(ValueError):
        decode_observer(raw, 100.)


def test_pair_requires_graphics_advance_and_preserves_frozen_physics_receipt_truth():
    one = dict(observer=decode_observer(graphics(),100.),physics420_sha256='a',native_packet=7,
               status=idle())
    two = copy.deepcopy(one)
    two['observer'] = decode_observer(graphics(packet=21),100.08)
    validate_pair(one,two)
    for field,value in [('physics420_sha256','changed'),('native_packet',8)]:
        bad = copy.deepcopy(two)
        bad[field] = value
        with pytest.raises(ValueError):
            validate_pair(one,bad)
    bad = copy.deepcopy(two)
    bad['observer'] = copy.deepcopy(one['observer'])  # Duplicate retains original receipt.
    with pytest.raises(ValueError):
        validate_pair(one,bad)


def test_repeated_copy_is_bounded_and_does_not_claim_atomicity():
    class Mapping:
        def __init__(self,stable):
            self.reads,self.stable = 0,stable
        def read(self):
            self.reads += 1
            return b'same' if self.stable else bytes([self.reads%2])
    good = Mapping(True)
    assert stable_native_copy(good) == (b'same',1) and good.reads == 2
    changing = Mapping(False)
    with pytest.raises(RuntimeError):
        stable_native_copy(changing)
    assert changing.reads == 16


def test_exact_three_children_stop_at_first_eligible_between_complete_tasks():
    p,result,calls = plan(),dict(attempts=[]),[]
    points = [.09588946402072906,.18,.26,.33]
    def observe(after,index):
        calls.append(('observe',index,after))
        return dict(observer=dict(spline_position=points[index],actor_input=False))
    def run(index):
        calls.append(('child',index))
        return child(),'a'*64
    outcome = execute_stages(p,result,observe=observe,child=run,
        verify=lambda:calls.append(('verify',)),checkpoint=lambda:None)
    assert outcome['status'] == 'completed_diagnostic_transit_reached_registered_start'
    assert len(result['attempts']) == 3 and all(r['role'] == 'transit' and not r['fit_allowed']
                                               for r in result['attempts'])
    assert calls == [('observe',0,None),('verify',),('child',0),('observe',1,20.),
        ('verify',),('child',1),('observe',2,20.),('verify',),('child',2),('observe',3,20.)]


def test_first_eligible_baseline_stops_without_child_or_further_owner():
    p,result = plan(),dict(attempts=[])
    p['expected_initial_spline'] = .3
    value = execute_stages(p,result,
        observe=lambda *args:dict(observer=dict(spline_position=.3)),
        child=lambda *args:pytest.fail('No activation'),
        verify=lambda:pytest.fail('No activation verify needed'),checkpoint=lambda:None)
    assert value['final_spline'] == .3 and not result['attempts']


def test_failed_or_uncertain_child_preserves_started_attempt_and_never_observes_next_owner():
    result,calls = dict(attempts=[]),[]
    def observe(after,index):
        calls.append(index)
        return dict(observer=dict(spline_position=.09588946402072906))
    broken = child()
    broken['lease_retained'] = True
    with pytest.raises(ValueError):
        execute_stages(plan(),result,observe=observe,child=lambda i:(broken,'b'*64),
                       verify=lambda:None,checkpoint=lambda:None)
    assert calls == [0] and len(result['attempts']) == 1
    assert result['attempts'][0]['status'] == 'started'


@pytest.mark.parametrize(('field','value'), [('status','failed'),('role','train'),
    ('owned_threads_still_live',['writer']),('transport_state','failed'),('fit_allowed',True)])
def test_only_original_complete_diagnostic_child_can_authorize_next_observer(field,value):
    actual = child()
    actual[field] = value
    with pytest.raises(ValueError):
        check_child(actual)


def test_overshoot_at_last_task_fails_without_another_lap_or_role_change():
    result,points = dict(attempts=[]),[.09588946402072906,.18,.26,.45]
    with pytest.raises(ValueError,match='Target passed'):
        execute_stages(plan(),result,
            observe=lambda after,i:dict(observer=dict(spline_position=points[i])),
            child=lambda i:(child(),'c'*64),verify=lambda:None,checkpoint=lambda:None)
    assert len(result['attempts']) == 3 and all(not r['fit_allowed'] for r in result['attempts'])


def test_source_failure_stops_before_any_child_and_exhausted_cap_is_not_success():
    result = dict(attempts=[])
    def fail():
        raise ValueError('changed')
    with pytest.raises(ValueError,match='changed'):
        execute_stages(plan(),result,
            observe=lambda *args:dict(observer=dict(spline_position=.09588946402072906)),
            child=lambda *args:pytest.fail('No activation'),verify=fail,checkpoint=lambda:None)
    assert not result['attempts']
    points = [.09588946402072906,.13,.17,.2]
    outcome = execute_stages(plan(),result,
        observe=lambda after,i:dict(observer=dict(spline_position=points[i])),
        child=lambda i:(child(),'d'*64),verify=lambda:None,checkpoint=lambda:None)
    assert outcome['status'] == 'completed_finite_transit_start_not_reached'
    assert len(result['attempts']) == 3


def test_saved_coordinate_link_never_reads_normal_status_or_creates_actor_route(tmp_path):
    csp,native = tmp_path/'saved-csp.json',tmp_path/'saved-guard.json'
    p = plan()
    csp.write_text(json.dumps(dict(after=dict(generation='fixture',epoch=0,paused=True,
        diagnostic_route=dict(spline_position=.095889464020729,lap_count=0,actor_input=False)))))
    native.write_text(json.dumps(dict(current_engine_pid=24200,current_generation='fixture',
        route_for_diagnostic_staging_only=dict(spline_position=.09588946402072906,lap_count=0,
                                               graphics_advance_observed=True))))
    p['coordinate_link'] = dict(CSP_receipt=str(csp),native_guard_summary=str(native),
        no_motion_between_saved_observers_root_attestation=True)
    value = coordinate_link(p)
    assert value['absolute_difference'] < 1e-15 and not value['actor_input']
    assert not value['continuous_geometry_verified']
    p['coordinate_link']['no_motion_between_saved_observers_root_attestation'] = False
    with pytest.raises(ValueError):
        coordinate_link(p)


@pytest.mark.parametrize('point',[.441,.99,float('nan'),-1.,1.])
def test_impossible_or_passed_region_never_starts_another_task(point):
    with pytest.raises(ValueError):
        transit_decision(point)
