from pathlib import Path
import pytest

lua_module=pytest.importorskip('lupa.luajit21')
SOURCE=Path('integrations/ac_csp/initial_tyre_temperature.lua')


def harness():
    lua=lua_module.LuaRuntime(unpack_returned_tuples=True)
    module=lua.execute(SOURCE.read_text(encoding='utf-8'))
    state=lua.table_from(dict(generation='g1',epoch=1,car_id='bmw_z4_gt3',track_id='monza',
        paused=True,in_menu=False,is_online=False,ai_controlled=False,camera_active=False,
        expert_probe_active=False,brake_settle_active=False,physics_api_allowed=True,
        speed_kmh=0,gas=0,brake=1))
    request=lua.table_from(dict(schema='flybrain.csp.session.request.v1',id='a'*32,
        command='set-tyres-temperature-initial',generation='g1',epoch=1,expires_clock=12,
        phase='episode_initialization',destination_flags=15,remove_blankets=True,
        released_controller_verified=True,temperature_c=40,
        expected_car_id='bmw_z4_gt3',expected_track_id='monza'))
    api=lua.execute('''local a={calls={}};
      function a.setTyresBlankets(car,wheels,active)
        a.calls[#a.calls+1]={name='blankets',car=car,wheels=wheels,active=active}
        if a.fail_blankets then error('blanket failure') end
      end
      function a.setTyresTemperature(car,wheels,temp,flags)
        a.calls[#a.calls+1]={name='temperature',car=car,temp=temp,flags=flags}
        if a.fail_temperature then error('temperature failure') end
      end
      return a''')
    return lua,module,state,request,api


def test_void_setters_require_readback_and_same_epoch_never_replays():
    _,m,s,r,a=harness()
    assert m.validate(r,s,10,a)
    result=m.apply(r,s,10,a)
    assert result.status=='thermal_calls_returned_readback_required'
    assert result.temperature_applied_verified is False
    assert result.current_lap_excluded is True
    assert len(a.calls)==2 and a.calls[1].active is False
    assert a.calls[2].temp==40 and a.calls[2].flags==15
    r.id='b'*32
    assert not m.validate(r,s,10,a)
    assert m.apply(r,s,10,a).status=='thermal_guard_rejected'
    assert len(a.calls)==2
    r.epoch=2;s.epoch=2
    assert m.validate(r,s,10,a)


@pytest.mark.parametrize('which,count',[('fail_blankets',1),('fail_temperature',2)])
def test_partial_failure_consumes_epoch_and_never_retries(which,count):
    _,m,s,r,a=harness();a[which]=True
    result=m.apply(r,s,10,a)
    assert result.status=='thermal_partial_or_unknown' and len(a.calls)==count
    assert not m.validate(r,s,10,a)
    assert result.temperature_applied_verified is False


@pytest.mark.parametrize('key,value',[
    ('paused',False),('in_menu',True),('is_online',True),('ai_controlled',True),
    ('camera_active',True),('expert_probe_active',True),('brake_settle_active',True),
    ('physics_api_allowed',False),('error','bad'),('pending_receipt_id','other'),
    ('speed_kmh',.1),('gas',.02),('brake',.98),('brake',1.1),
    ('speed_kmh',float('nan')),('car_id','other'),('track_id','other'),
    ('epoch',2),('generation','other'),('paused',None),('ai_controlled',None)])
def test_csp_state_guards_prevent_any_setter(key,value):
    _,m,s,r,a=harness();s[key]=value
    assert not m.validate(r,s,10,a)
    assert m.apply(r,s,10,a).status=='thermal_guard_rejected'
    assert len(a.calls)==0


@pytest.mark.parametrize('key,value',[
    ('temperature_c',19.9),('temperature_c',120.1),('temperature_c',float('nan')),
    ('expires_clock',9),('expires_clock',16),('phase','measured_lap'),
    ('destination_flags',8),('remove_blankets',False),('released_controller_verified',False),
    ('expected_car_id','other'),('expected_track_id','other'),('id','invalid')])
def test_request_guards_prevent_any_setter(key,value):
    _,m,s,r,a=harness();r[key]=value
    assert not m.validate(r,s,10,a)
    assert len(a.calls)==0


def test_builder_preserves_prior_source_and_compiles_full_installed_base():
    from scripts.build_initial_thermal_csp import build
    installed=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua')
    if not installed.exists():pytest.skip('Full AC deployed source unavailable')
    base=installed.read_text(encoding='utf-8')
    # After deployment, use preserved exact base to make the same compilation test.
    if 'local initialTyres =' in base:
        base=Path('runs/thermal-initial-20261103/build-01/deployed-base.lua').read_text(encoding='utf-8')
    module=SOURCE.read_text(encoding='utf-8')
    candidate=build(base,module)
    lua=lua_module.LuaRuntime(unpack_returned_tuples=True)
    ok,error=lua.eval('function(s) local f,e=loadstring(s); return f~=nil,e end')(candidate)
    assert ok,error
    assert 'source_body_recording_1600_v1 = true' in candidate
    assert candidate.count('initialTyres.apply(')==1
    with pytest.raises(ValueError):build(candidate,module)
    with pytest.raises(ValueError):build(base+'\nlocal function sessionState()',module)


def test_explicit_live_initialization_uses_same_one_shot_epoch_and_strict_native_witness():
    lua,m,s,r,a=harness();r.phase='episode_initialization_live';s.paused=False;s.live=True
    r.native_stationary_verified=True;r.native_stationary_packets=lua.table_from([10,11,12])
    assert m.validate(r,s,10,a)
    assert m.apply(r,s,10,a).status=='thermal_calls_returned_readback_required'
    assert len(a.calls)==2
    s.paused=True;r.phase='episode_initialization'
    assert not m.validate(r,s,10,a)

@pytest.mark.parametrize('problem',['no_live','no_proof','duplicate','reverse','no_packets'])
def test_live_initialization_rejects_incomplete_or_fabricated_witness(problem):
    lua,m,s,r,a=harness();r.phase='episode_initialization_live';s.paused=False;s.live=True
    r.native_stationary_verified=True;r.native_stationary_packets=lua.table_from([10,11,12])
    if problem=='no_live':s.live=False
    if problem=='no_proof':r.native_stationary_verified=False
    if problem=='duplicate':r.native_stationary_packets=lua.table_from([10,10,12])
    if problem=='reverse':r.native_stationary_packets=lua.table_from([12,11,10])
    if problem=='no_packets':r.native_stationary_packets=None
    assert not m.validate(r,s,10,a) and len(a.calls)==0


def wake_harness():
    lua,m,s,r,a=harness();s.paused=False;s.live=True
    r.phase='episode_initialization_live';r.command='wake-thermal-native-initial'
    a.awakeCar=lua.eval("function(car) assert(car==0); return nil end")
    return lua,m,s,r,a


def test_wake_is_one_shot_without_consuming_temperature_and_preserves_proof():
    lua,m,s,r,a=wake_harness()
    result=m.prepareNative(r,s,10,a)
    assert result.status=='native_wake_returned_readback_required'
    assert result.temperature_setter_called is False
    assert result.native_readiness_verified is False
    assert result.current_lap_excluded is True and len(a.calls)==0
    assert m.lastWakeEpoch()==1 and m.lastEpoch() is None
    r.id='b'*32
    assert not m.validateWake(r,s,10,a)
    assert m.prepareNative(r,s,10,a).status=='native_wake_guard_rejected'
    r.command='set-tyres-temperature-initial'
    assert not m.validate(r,s,10,a)
    r.native_stationary_verified=True;r.native_stationary_packets=lua.table_from([20,21,22])
    assert m.validate(r,s,10,a)
    assert m.apply(r,s,10,a).status=='thermal_calls_returned_readback_required'
    assert len(a.calls)==2


@pytest.mark.parametrize('mode',['error','false'])
def test_unknown_wake_never_retries_and_never_calls_temperature(mode):
    lua,m,s,r,a=wake_harness()
    a.awakeCar=lua.eval("function(car) error('unknown wake') end" if mode=='error' else "function(car) return false end")
    result=m.prepareNative(r,s,10,a)
    assert result.status=='native_wake_unknown'
    assert m.lastWakeEpoch()==1 and m.lastEpoch() is None and len(a.calls)==0
    r.id='c'*32
    assert not m.validateWake(r,s,10,a)


@pytest.mark.parametrize('key,value',[('paused',True),('live',False),('camera_active',True),('ai_controlled',True),('speed_kmh',.1),('brake',.98)])
def test_wake_rejects_nonquiescent_state(key,value):
    _,m,s,r,a=wake_harness();s[key]=value
    assert not m.validateWake(r,s,10,a)
    assert m.prepareNative(r,s,10,a).status=='native_wake_guard_rejected'
    assert m.lastWakeEpoch() is None and len(a.calls)==0


def test_thermal_consumption_prevents_later_wake():
    lua,m,s,r,a=wake_harness();r.command='set-tyres-temperature-initial'
    r.native_stationary_verified=True;r.native_stationary_packets=lua.table_from([20,21,22])
    assert m.apply(r,s,10,a).status=='thermal_calls_returned_readback_required'
    r.command='wake-thermal-native-initial'
    assert not m.validateWake(r,s,10,a)


def test_live_builder_extends_exact_v2_source_and_compiles_entire_candidate():
    from scripts.build_live_thermal_csp import build
    base=Path('runs/thermal-apply-diagnostic-20261104/live-build-02/FlyBrainCamera.lua').read_text(encoding='utf-8')
    old=Path('runs/thermal-apply-diagnostic-20261104/live-build-02/thermal-module.lua').read_text(encoding='utf-8')
    candidate=build(base,old,SOURCE.read_text(encoding='utf-8'))
    lua=lua_module.LuaRuntime(unpack_returned_tuples=True)
    ok,error=lua.eval('function(s) local f,e=loadstring(s); return f~=nil,e end')(candidate)
    assert ok,error
    assert candidate.count("if request.command == 'wake-thermal-native-initial' then")==1
    assert candidate.count('local function thermalSDKSnapshot()')==1
    assert candidate.count('tyres_native_wake_epoch =')==1
    assert 'source_body_recording_1600_v1 = true' in candidate
