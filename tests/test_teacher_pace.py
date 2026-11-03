import json
import os
import time

import pytest
from test_ac_lua_session_receipts import (
    fast_motion_service,
    handled_luajit_errors_are_not_python_fatal_signals,  # noqa: F401
    set_owned_camera,
)

from flybrain_drive.teacher_pace import PACE_COMMAND, goal_matches, validate_pace_receipt
from scripts.control_ac_session import validate_probe_receipt
from scripts.record_ac_engine_ai_probe import expert_command, read_ai_phase


def pace_service():
    lua, _ = fast_motion_service()
    set_owned_camera(lua)
    lua.execute('''
      request.command='expert-pace60-100-stop30'; request.hold_seconds=30
      request.speed_limit_kmh=100; request.pace_contract='flybrain.external_speed_task.v1'
      request.pace_boost_after_seconds=5; request.pace_stop_after_seconds=15
      request.pace_restart_after_seconds=23; request.pace_stop_seconds=8
      pulses={}
      physics.setAIStopCounter=function(index,duration)
        assert(index==0);table.insert(pulses,duration)
        if paceThrow and duration>0 then error('uncertain pace stop outcome') end
      end
    ''')
    return lua


def test_pace_keeps_external_stop_goal_separate_from_ai_speed_hint_and_cleanup():
    lua = pace_service()
    lua.execute('''
      script.update(0)
      now=6.05;camera.last_capture_clock=now;car.position.x=20
      car.speedKmh=60;car.gas=.3;car.brake=0;script.update(0)
      now=16.05;camera.last_capture_clock=now;car.position.x=150;car.speedKmh=100
      script.update(0)
      now=16.10;camera.last_capture_clock=now;car.position.x=153
      car.speedKmh=85;car.brake=.7;car.gas=0;script.update(0)
      assert(files['mock/flybrain/csp-session/status.json'].external_task.desired_speed_m_s==0)
      now=20.05;camera.last_capture_clock=now;car.speedKmh=.2;car.brake=1
      car.position.x=300;script.update(0)
      now=24.05;camera.last_capture_clock=now;car.speedKmh=0;script.update(0)
      now=31.05;camera.last_capture_clock=now;car.speedKmh=55;car.gas=.2;car.brake=0
      car.position.x=400;script.update(0)
      now=31.15;car.speedKmh=0;car.gas=0;car.brake=1;script.update(0)
      now=31.30;script.update(0)
    ''')
    table = lua.eval('files[terminal]')
    receipt = dict(table.items())
    for key in ('pace_transitions', 'telemetry'):
        receipt[key] = [dict(table[key][i].items()) for i in range(1, len(table[key])+1)]
    assert receipt['status'] == 'api_returned_true'
    assert receipt['pace_task_passed'] is True
    assert receipt.get('label_speed_hint_kmh') is None
    assert lua.eval('#pulses') == 2
    assert lua.eval('pulses[1]') == 8 and lua.eval('pulses[2]') == 0
    assert any(s.get('external_task_phase') == 'stop0'
               and s['teacher_speed_hint_kmh'] == 100 for s in receipt['telemetry'])
    assert validate_pace_receipt(receipt)['moving_brake_samples'] >= 1
    validate_probe_receipt(receipt, motion=True, collection=True, fast=True, pace=True)
    with pytest.raises(ValueError):
        validate_probe_receipt(receipt, motion=True, collection=True, fast=True, speed60=True)


@pytest.mark.parametrize('mutation', [
    'request.speed_limit_kmh=60', 'request.pace_stop_seconds=.4',
    'request.pace_restart_after_seconds=22', 'request.pace_boost_after_seconds=4',
    'request.pace_contract=nil', 'physics.setAIStopCounter=nil',
])
def test_changed_pace_contract_is_rejected_before_any_setter(mutation):
    lua = pace_service()
    lua.execute(mutation+';script.update(0)')
    assert lua.eval('#setters') == 0 and lua.eval('calls') == 0
    assert lua.eval('files[accepted].status') == 'rejected'


def test_unknown_pace_stop_is_never_repeated_and_does_not_pass():
    lua = pace_service()
    lua.execute('''
      paceThrow=true;script.update(0)
      now=6.05;camera.last_capture_clock=now;car.position.x=20
      car.speedKmh=60;car.gas=.2;car.brake=0;script.update(0)
      now=16.05;camera.last_capture_clock=now;car.position.x=100;car.speedKmh=90
      script.update(0);now=16.1;camera.last_capture_clock=now;script.update(0)
      now=16.2;car.speedKmh=0;car.gas=0;car.brake=1;script.update(0)
      now=16.3;script.update(0)
    ''')
    assert lua.eval('#pulses') == 2
    assert lua.eval('pulses[1]') == 8 and lua.eval('pulses[2]') == 0
    assert lua.eval('files[terminal].status') == 'api_outcome_unknown'
    assert lua.eval('files[terminal].pace_task_passed') is False


def test_pace_requires_real_moving_brake_and_stop_before_restart():
    lua = pace_service()
    lua.execute('''
      script.update(0)
      now=6.05;camera.last_capture_clock=now;car.position.x=20
      car.speedKmh=60;car.gas=.2;car.brake=0;script.update(0)
      now=16.05;camera.last_capture_clock=now;car.position.x=100;car.speedKmh=90
      script.update(0)
      now=24.05;camera.last_capture_clock=now;car.speedKmh=50;script.update(0)
      now=24.1;car.speedKmh=0;car.gas=0;car.brake=1;script.update(0)
      now=24.2;script.update(0)
    ''')
    assert lua.eval('files[terminal].status') == 'api_returned_false'
    assert lua.eval('files[terminal].pace_task_passed') is False


def test_stopped_brake_without_moving_brake_does_not_release_stop_or_restart():
    lua = pace_service()
    lua.execute('''
      script.update(0)
      now=6.05;camera.last_capture_clock=now;car.position.x=20
      car.speedKmh=60;car.gas=.2;car.brake=0;script.update(0)
      now=16.05;camera.last_capture_clock=now;car.position.x=100;car.speedKmh=90
      script.update(0)
      now=20.05;camera.last_capture_clock=now;car.speedKmh=0;car.gas=0;car.brake=1
      script.update(0)
      now=24.05;camera.last_capture_clock=now;script.update(0)
      now=24.1;script.update(0);now=24.2;script.update(0)
    ''')
    assert lua.eval('files[terminal].pace_task_passed') is False
    assert lua.eval('files[terminal].pace_moving_brake_seen') is False
    assert lua.eval("table.concat(setters,','):find('limit:60',1,true)") is not None
    assert lua.eval('#files[terminal].pace_transitions') == 3  # No restart transition.


def test_pace_requires_explicit_recorder_preset_and_task_is_a_held_instruction(tmp_path):
    assert expert_command(30, 100, pace=True) == PACE_COMMAND
    for duration, speed, pulse in ((3, 100, False), (30, 60, False), (30, 100, True)):
        with pytest.raises(ValueError):
            expert_command(duration, speed, brake_pulse=pulse, pace=True)
    with pytest.raises(ValueError):
        expert_command(30, 100)
    path = tmp_path/'status.json'
    task = dict(schema='flybrain.external_speed_task.v1', phase='stop0',
                desired_speed_m_s=0, actor_input=True)
    path.write_text(json.dumps(dict(generation='g', script_clock=10, external_task=task,
                                   ai_controlled=True, paused=False)))
    modified = time.time()-.05  # Avoid Windows filesystem timestamp rounding into the future.
    os.utime(path, (modified, modified))
    sampled = read_ai_phase(path, 'g', task=True, clock=lambda: 123.)
    assert sampled['desired_speed_m_s'] == 0 and sampled['source_script_clock'] == 10
    assert sampled['observed_at_host_seconds'] == 123.
    assert sampled['held_external_instruction'] is True


def test_actual_csp_json_rounded_goal_is_retained_and_meaningful_deviation_rejected(tmp_path):
    path = tmp_path/'status.json'
    task = dict(schema='flybrain.external_speed_task.v1', phase='accelerate60',
                desired_speed_m_s=16.666666666667, actor_input=True)
    value = dict(generation='g', script_clock=669.4804709, external_task=task,
                 ai_controlled=True, paused=False)
    path.write_text(json.dumps(value))
    modified = time.time()-.05
    os.utime(path, (modified, modified))
    sampled = read_ai_phase(path, 'g', task=True, clock=lambda: 123.)
    assert sampled['desired_speed_m_s'] == task['desired_speed_m_s']
    assert sampled['desired_speed_m_s'] != 60/3.6  # Never canonicalize the wire value.
    assert goal_matches(27.777777777778, 100/3.6)
    for bad in (False, '16.666666666667', float('nan'), float('inf'),
                60/3.6+1e-9, 60/3.6+1e-7):
        assert not goal_matches(bad, 60/3.6)
    value['external_task']['desired_speed_m_s'] = 60/3.6+1e-7
    path.write_text(json.dumps(value))
    os.utime(path, (modified, modified))
    with pytest.raises(ValueError, match='externally commanded'):
        read_ai_phase(path, 'g', task=True, clock=lambda: 123.)
