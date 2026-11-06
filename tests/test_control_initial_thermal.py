import json
import time
import pytest
from pathlib import Path
from types import SimpleNamespace
from scripts import control_ac_session,control_initial_thermal
from flybrain_drive.evidence import write_json
from flybrain_drive.initial_thermal_contract import THERMAL_COMMAND


@pytest.mark.parametrize('resume,pause,live,wake,execute,wake_status',[(False,False,False,False,True,'native_wake_returned_readback_required'),(True,False,False,False,True,'native_wake_returned_readback_required'),(True,True,False,False,True,'native_wake_returned_readback_required'),(False,True,True,False,True,'native_wake_returned_readback_required'),(False,True,True,True,True,'native_wake_returned_readback_required'),(False,False,True,True,False,'native_wake_returned_readback_required'),(False,False,True,True,True,'native_wake_unknown')])
def test_host_journals_once_and_accepts_readback_from_consumed_epoch(tmp_path,monkeypatch,resume,pause,live,wake,execute,wake_status,resume_inside=False,post_settle=False):
    m=control_initial_thermal
    state=dict(schema='flybrain.csp.session.status.v1',generation='g',epoch=1,window_handle='7',
        script_clock=10,is_online=False,in_menu=False,paused=True,live=False,
        ai_controlled=False,camera_active=False,expert_probe_active=False,brake_settle_active=False,
        physics_api_allowed=True,car_id='bmw_z4_gt3',track_id='monza',speed_kmh=0,gas=0,brake=1,
        capabilities=dict(initial_tyre_temperature_v1=True))
    if live:
        state.update(paused=False,live=True);state['capabilities']['initial_tyre_temperature_live_v1']=True
        state['capabilities']['native_thermal_wake_v1']=True
    if resume_inside:state.update(paused=True,live=False)
    session_root=tmp_path/'flybrain/csp-session';session_root.mkdir(parents=True)
    output=tmp_path/'output';axis=tmp_path/'axis.json'
    axis.write_text(json.dumps(dict(dll_path='mock',dll_sha256='mock',device_id=1)),encoding='utf-8')
    window=SimpleNamespace(hwnd=7,executable='C:/mock/acs.exe')
    api=SimpleNamespace(discover=lambda:[window],probe=lambda *a,**k:window,
        user=SimpleNamespace(GetForegroundWindow=lambda:8))
    monkeypatch.setenv('LOCALAPPDATA',str(tmp_path))
    monkeypatch.setattr(m,'WindowsACWindow',lambda:api)
    monkeypatch.setattr(m,'load_vjoy_sdk',lambda *a:SimpleNamespace(GetVJDStatus=lambda device:1))
    monkeypatch.setattr(m,'sha256',lambda p:'pinned')
    def begin(path,config):
        Path(path).mkdir();return Path(path)
    monkeypatch.setattr(m,'begin_run',begin)
    reader=SimpleNamespace(mapping=None,close=lambda:None)
    monkeypatch.setattr(m,'NativeTeacherPhysicsReader',lambda **kw:reader)
    packet=[10];published=[];post_count=[0]
    def native(*a,**k):
        if not resume or state['paused'] is False:packet[0]+=1
        if any(r['command']==THERMAL_COMMAND for r in published):post_count[0]+=1
        speed=.169 if post_settle and post_count[0]==1 else 0.
        return SimpleNamespace(physics=SimpleNamespace(packet_id=packet[0],received_monotonic_seconds=time.perf_counter(),
            fields=dict(speedKmh=speed,gas=0,brake=1,tyreCoreTemperature=[40 if any(r['command']==THERMAL_COMMAND for r in published) else 80]*4)),
            builtin_ai_controlled=False,authority_record=lambda:dict(packet_id=packet[0],same_native_copy=True))
    monkeypatch.setattr(m,'repeated_teacher_snapshot',native)
    monkeypatch.setattr(control_ac_session,'read_session_heartbeat',lambda p:(dict(state),time.time(),0))
    def write(path,value):
        write_json(path,value)
        if Path(path)==session_root/'request.json':
            published.append(value)
            if value['command']==THERMAL_COMMAND:state['tyres_initialization_epoch']=1
            elif value['command']=='wake-thermal-native-initial':state['tyres_native_wake_epoch']=1
            else:
                assert value['command'] in ('resume','pause')
                state['paused']=value['command']=='pause';state['live']=not state['paused']
    monkeypatch.setattr(m,'write_json',write)
    def terminal(root,request):
        status={'wake-thermal-native-initial':wake_status,THERMAL_COMMAND:'thermal_calls_returned_readback_required'}.get(request['command'],'api_returned_true')
        return dict(status=status,id=request['id'],command=request['command'])
    monkeypatch.setattr(control_ac_session,'terminal_receipt',terminal)
    args=SimpleNamespace(output=str(output),command=THERMAL_COMMAND,expected_lua_sha256='pinned',
        settle_initial_thermal_readback=post_settle,resume_before_live_initialization=resume_inside,wake_native_before_live_initialization=wake,live_initialization=live,temperature_c=40,axis_config=str(axis),initialization_run=None,execute=execute,resume_for_readback=resume,pause_after_readback=pause)
    if wake and execute and wake_status!='native_wake_returned_readback_required':
        with pytest.raises(SystemExit):m.run_initial_thermal(args)
        assert len(published)==1 and published[0]['command']=='wake-thermal-native-initial'
        result=json.loads((output/'result.json').read_text())
        assert result['status']=='failed' and result['lifecycle_requests_sent']==1
        assert not (output/'request.json').exists()
        return
    m.run_initial_thermal(args)
    result=json.loads((output/'result.json').read_text())
    if not execute:
        assert result['status']=='read_only' and published==[]
        assert result['temperature_applied_verified'] is False
        assert result['lifecycle_requests_sent']==0
        return
    assert result['temperature_applied_verified'] is True
    if wake:
        assert published[0]['command']=='wake-thermal-native-initial'
        wake_packet=json.loads((output/'native-wake-publication.json').read_text())['initial_packet']
        assert all(row['packet_id']>wake_packet for row in json.loads((output/'live-before-native.json').read_text()))
    if live:
        thermal=[v for v in published if v['command']==THERMAL_COMMAND][0]
        assert thermal['phase']=='episode_initialization_live'
        assert len(thermal['native_stationary_packets'])==3
        assert len(set(thermal['native_stationary_packets']))==3
        assert (output/'live-before-native.json').exists()
    assert result['lifecycle_requests_sent']==1+int(resume)+int(pause)+int(wake)+int(resume_inside) and len(published)==1+int(resume)+int(pause)+int(wake)+int(resume_inside)
    if pause:
        assert state['paused'] is True and result['paused_after_readback_verified'] is True
        assert (output/'frozen-native.json').exists()
    for request in published:
        intent=json.loads((session_root/'intent-journal'/f"{request['id']}.json").read_text())
        assert intent['request']==request
    assert result['current_lap_excluded'] is True
    assert result['training_admissible'] is False


def test_owned_pre_setting_resume_keeps_reader_open_and_sets_temperature_once(tmp_path,monkeypatch):
    test_host_journals_once_and_accepts_readback_from_consumed_epoch(tmp_path,monkeypatch,
        False,True,True,False,True,'native_wake_returned_readback_required',resume_inside=True)
    import json
    out=tmp_path/'output'
    result=json.loads((out/'result.json').read_text())
    assert result['initial_resume_stationary_native_verified'] is True
    rows=json.loads((out/'initial-resume-observations.json').read_text())
    assert len(rows)>=3 and not any(row['actor_admitted'] for row in rows)
    assert (out/'initial-resume-api-receipt.json').exists()


def test_owned_pre_setting_resume_is_also_read_only_without_execute(tmp_path,monkeypatch):
    test_host_journals_once_and_accepts_readback_from_consumed_epoch(tmp_path,monkeypatch,
        False,False,True,False,False,'native_wake_returned_readback_required',resume_inside=True)
    assert not (tmp_path/'output/initial-resume-request.json').exists()


def test_post_setter_motion_is_preserved_before_strict_three_packet_readback(tmp_path,monkeypatch):
    test_host_journals_once_and_accepts_readback_from_consumed_epoch(tmp_path,monkeypatch,
        False,True,True,False,True,'native_wake_returned_readback_required',resume_inside=True,post_settle=True)
    import json
    output=tmp_path/'output'
    rows=json.loads((output/'post-thermal-observations.json').read_text())
    assert rows[0]['stationary_native'] is False
    assert rows[0]['rejection']=='nonstationary_initialization_sample'
    assert sum(r['stationary_native'] for r in rows)>=3
    assert not any(r['actor_admitted'] for r in rows)
