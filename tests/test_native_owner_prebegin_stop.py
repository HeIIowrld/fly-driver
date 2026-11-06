import json
import pytest
from flybrain_drive import native_flight_body_owner_process as owner


def test_prepared_owner_can_close_before_sensory_episode_without_starting_sampler(tmp_path,monkeypatch):
    class Runtime:
        def __init__(self,*args):self.model=object();self.episode=None;self.closed=False
        def close(self):self.closed=True
    class Pipe:
        def __init__(self):self.sent=[];self.closed=False
        def recv(self):return {'kind':'stop'}
        def send(self,value):self.sent.append(value)
        def close(self):self.closed=True
    def forbidden_sampler(*args,**kwargs):raise AssertionError('No sampling before explicit begin')
    monkeypatch.setattr(owner,'NativeFlightDecisionRuntime',Runtime)
    monkeypatch.setattr(owner,'prepare_fixed_cpu_backend',lambda model:{'fixture':True})
    monkeypatch.setattr(owner,'NativeProcessRowsSampler',forbidden_sampler)
    pipe=Pipe();owner._serve_owner(pipe,(),{},tmp_path)
    terminal=json.loads((tmp_path/'owner-terminal.json').read_text())
    assert pipe.closed and pipe.sent[0]['status']=='ready' and pipe.sent[-1]['status']=='stopped'
    assert terminal['sampler_started'] is False and terminal['sampler_stopped'] is True
    assert terminal['native_producer_pid'] is None and terminal['outputs']==0
    for name in ['owner-native-copy.json','owner-rejected-native-copy.json','owner-retired-native-copy.json','owner-ticker-events.json']:
        assert json.loads((tmp_path/name).read_text())==[]


def test_larger_wait_is_explicit_startup_only_and_does_not_allow_second_begin():
    process=owner.NativeFlightBodyOwnerProcess.__new__(owner.NativeFlightBodyOwnerProcess)
    process.episode=None;process.poisoned=False;sent=[];waited=[]
    class Pipe:
        def send(self,value):sent.append(value)
    process.pipe=Pipe()
    def receive(timeout):waited.append(timeout);return {'status':'begun','episode':'new'}
    process._receive=receive
    with pytest.raises(ValueError):process.begin_episode('new','generation',startup_timeout=31)
    assert sent==[]
    process.begin_episode('new','generation',startup_timeout=30)
    assert waited==[30] and len(sent)==1
    with pytest.raises(RuntimeError):process.begin_episode('again','generation',startup_timeout=30)


def test_motion_owner_is_explicit_and_can_close_without_starting_native_source(tmp_path,monkeypatch):
    from flybrain_drive import motion_spatial_runtime
    marker=object();received=[]
    class Runtime:
        def __init__(self,*args,motion_port):
            received.append(motion_port);self.model=object();self.episode=None
        def close(self):pass
    class Pipe:
        def __init__(self):self.sent=[]
        def recv(self):return {'kind':'stop'}
        def send(self,value):self.sent.append(value)
        def close(self):pass
    monkeypatch.setattr(motion_spatial_runtime,'MotionSpatialDecisionRuntime',Runtime)
    monkeypatch.setattr(owner,'prepare_fixed_cpu_backend',lambda model:{})
    def forbidden(*args,**kwargs):raise AssertionError('Source must not start before begin')
    monkeypatch.setattr(owner,'NativeProcessRowsSampler',forbidden)
    pipe=Pipe();owner._serve_owner(pipe,(),{},tmp_path,motion_port=marker)
    assert received==[marker] and pipe.sent[0]['motion_features_required'] is True
    assert pipe.sent[-1]['summary']['sampler_started'] is False


@pytest.mark.parametrize('deferred',[False,True])
def test_owner_gc_scope_surrounds_source_and_restores_after_source_stopped(tmp_path,monkeypatch,deferred):
    import gc
    from types import SimpleNamespace
    previous=gc.isenabled();observed=[]
    class Runtime:
        def __init__(self,*args):self.model=object();self.episode=None
        def begin_episode(self,episode):self.episode=episode
        def close(self):observed.append(('runtime_close',gc.isenabled()))
    class Sampler:
        def __init__(self,**kw):
            self.rows=[];self.source=SimpleNamespace(retired_rows=[],rejected_rows=[],process=SimpleNamespace(pid=456))
        def start(self):observed.append(('source_start',gc.isenabled()))
        def stop(self):observed.append(('source_stop',gc.isenabled()))
    class Ticker:
        def __init__(self,*args,**kwargs):self.thread=SimpleNamespace(ident=1);self.events=[];self.wait_events=[];self.error=None
        def start(self):pass
        def stop(self):observed.append(('ticker_stop',gc.isenabled()))
    class Pipe:
        messages=iter([{'kind':'begin','episode':'e','generation':'g'},{'kind':'stop'}])
        def recv(self):return next(self.messages)
        def send(self,value):pass
        def close(self):pass
    monkeypatch.setattr(owner,'NativeFlightDecisionRuntime',Runtime)
    monkeypatch.setattr(owner,'prepare_fixed_cpu_backend',lambda model:{})
    monkeypatch.setattr(owner,'NativeProcessRowsSampler',Sampler)
    monkeypatch.setattr(owner,'NativeFlightBodyTicker',Ticker)
    monkeypatch.setattr(owner,'OnlineBodyShadow',lambda **kwargs:None)
    owner._serve_owner(Pipe(),(),{},tmp_path,defer_cyclic_gc=deferred)
    assert gc.isenabled()==previous
    assert all(enabled==(False if deferred else previous) for _,enabled in observed)
    terminal=json.loads((tmp_path/'owner-terminal.json').read_text())
    assert terminal['sampler_stopped'] and terminal['outputs']==0
    if deferred:assert terminal['gc_scope']['previous_state_restored']
