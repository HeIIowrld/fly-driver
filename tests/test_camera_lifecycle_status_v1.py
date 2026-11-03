"""Three focused policy groups; CPU Lua mocks only, no device, model or live call."""

import ast
import copy
import faulthandler
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from flybrain_drive.camera_status_policy_v1 import (
    ECHO_KEY,
    ECHO_SCHEMA,
    POLICY,
    from_record,
    profile_record,
    validate_producer_echo,
    validate_requested_policy,
)
from scripts import prepare_camera_lifecycle_status_v1 as builder

lua_module = pytest.importorskip('lupa.luajit21')
LUA_PATH = Path('integrations/ac_csp/candidates/FlyBrainCamera-mmf-lifecycle-status-v1.lua')
HOST_PATH = Path('src/flybrain_drive/ac_csp_vision_lifecycle_status_v1.py')


@pytest.fixture(autouse=True)
def handled_lua_exceptions_are_not_fatal_python_signals():
    restore = sys.platform == 'win32' and faulthandler.is_enabled()
    if restore:
        faulthandler.disable()
    try:
        yield
    finally:
        if restore:
            faulthandler.enable()


def lua():
    runtime = lua_module.LuaRuntime(unpack_returned_tuples=True)
    text = LUA_PATH.read_text(encoding='utf8')
    status = text[text.index('local function status(state, reason)'):
                  text.index('-- BEGIN explicit MMF camera status lifecycle policy v1.')]
    runtime.execute('''
      generation, lastID, statusPath, writeCalls, writeFails = 'g', 'r', 'mock/status', 0, false
      ac={getPatchVersion=function()return 'mock'end, getPatchVersionCode=function()return 1 end,
          getSim=function()return{frame=1,windowHandle=1}end}
      os.preciseClock=function()return 2 end
      function write(path,value)
        writeCalls=writeCalls+1
        if writeFails then error('immutable write failed') end
        lastPath,lastValue=path,value
      end
    '''+status+builder.LUA+'''
      policyAPI={accept=acceptCameraStatusPolicy,echo=cameraStatusPolicyEcho,
                 terminal=writeCameraTerminal,status=status}
    ''')
    return runtime


def host_method(name):
    tree = ast.parse(HOST_PATH.read_text(encoding='utf8'))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == 'CSPVisionRecorder')
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_case1_exact_optin_rejects_bad_types_nonMMF_and_profile_boolean_aliases_before_setup():
    profile = profile_record()
    for key, value in profile.items():
        if type(value) is bool:
            with pytest.raises(ValueError):
                from_record(dict(profile, **{key: int(value)}))
    for value in (True, False, 0, 1, {}, 'unknown'):
        with pytest.raises(ValueError):
            validate_requested_policy(value, mmf_metadata=True, shared_transport=True)
    for mmf, shared in ((False, True), (True, False), (1, True), (True, 1)):
        with pytest.raises(ValueError):
            validate_requested_policy(POLICY, mmf_metadata=mmf, shared_transport=shared)
    init = host_method('__init__')
    boundary = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == 'validate_camera_budget')
    prefix = ast.FunctionDef(name='guard', args=init.args,
        body=init.body[:boundary]+[ast.Raise(exc=ast.Call(func=ast.Name(
            id='AssertionError', ctx=ast.Load()), args=[ast.Constant('resource boundary')],
            keywords=[]))], decorator_list=[])
    namespace = dict(from_record=from_record, validate_requested_policy=validate_requested_policy)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[])),
                 '<actual-new-constructor-prefix>', 'exec'), namespace)
    for mmf, shared in ((False, True), (True, False)):
        with pytest.raises(ValueError):
            namespace['guard'](SimpleNamespace(), 0, {}, camera_status_profile=profile,
                               mmf_metadata=mmf, shared_transport=shared)
    runtime = lua()
    accept = runtime.globals().policyAPI.accept
    for value in (True, False, 0, 1, 'unknown'):
        ok, _ = runtime.eval('pcall')(accept, runtime.table_from(dict(
            camera_status_policy=value, mmf_metadata=True, shared_transport=True)))
        assert ok is False
    for mmf, shared in ((False, True), (True, False), (1, True), (True, 1)):
        ok, _ = runtime.eval('pcall')(accept, runtime.table_from(dict(
            camera_status_policy=POLICY, mmf_metadata=mmf, shared_transport=shared)))
        assert ok is False
    source = LUA_PATH.read_text(encoding='utf8')
    begin = source[source.index('local function begin(request, now)'):]
    assert begin.index('acceptCameraStatusPolicy(request)') < begin.index('lastID = request.id')
    assert begin.index('acceptCameraStatusPolicy(request)') < begin.index('io.createDir(directory)')
    assert runtime.globals().writeCalls == 0


def test_case2_default_inverse_periodic_only_suppression_and_exact_accepted_echo(
        monkeypatch, tmp_path):
    source = LUA_PATH.read_text(encoding='utf8')
    assert callable(lua().eval('loadstring')(source))  # Compile whole candidate, never execute it.
    original = next(iter(builder.BASES))
    assert builder.inverse_lua(source).encode() == Path(original).read_bytes()
    for path, raw in builder.build().items():
        assert path.read_bytes() == raw
    start = source.index('    if now >= nextStatus then\n')
    block = source[start:source.index('\n  end)', start)]
    for active, policy, expected in ((True, None, 1), (True, 'unknown', 1),
                                     (True, POLICY, 0), (False, POLICY, 1)):
        runtime = lua()
        runtime.execute("now=2;nextStatus=1;calls=0;terminalState='stopped';"
                        "function status()calls=calls+1 end")
        runtime.globals().CAMERA_STATUS_POLICY = POLICY
        runtime.globals().active = (
            runtime.table_from({'camera_status_policy': policy}) if active else None)
        runtime.execute(block)
        assert runtime.globals().calls == expected and runtime.globals().nextStatus == 3
    runtime = lua()
    api = runtime.globals().policyAPI
    request = runtime.table_from(dict(id='r', camera_status_policy=POLICY,
                                     mmf_metadata=True, shared_transport=True))
    owner = runtime.table_from(dict(request=request, camera_status_policy=api.accept(request)))
    echo = dict(api.echo(owner).items())
    validate_producer_echo(echo, policy=POLICY, request_id='r', generation='g')
    # 807 can retain the unknown request string but has no accepted-policy state.
    ignored_owner = runtime.table_from(dict(request=request))
    assert api.echo(ignored_owner) is None
    with pytest.raises(ValueError, match='807'):
        validate_producer_echo(None, policy=POLICY, request_id='r', generation='g')
    value = runtime.table_from(dict(status='stopped', frames=1, request_id='r'))
    api.terminal(ignored_owner, 'mock/result', value)
    assert dict(value.items()) == dict(status='stopped', frames=1, request_id='r')
    foreign = tmp_path/'existing.lua'
    foreign.write_bytes(b'original')
    monkeypatch.setattr(builder, 'build', lambda: {foreign: b'different'})
    monkeypatch.setattr(sys, 'argv', ['builder', '--write-new-sources'])
    with pytest.raises(ValueError, match='no overwrite'):
        builder.main()
    assert foreign.read_bytes() == b'original'


def test_case3_initial_terminal_write_failures_and_echo_failure_do_not_weaken_cleanup():
    runtime = lua()
    api = runtime.globals().policyAPI
    runtime.globals().writeFails = True
    assert runtime.eval('pcall')(api.status, 'capturing', None)[0] is False
    owner = runtime.table_from(dict(request=runtime.table_from(dict(id='r')),
                                    camera_status_policy=POLICY))
    assert runtime.eval('pcall')(api.terminal, owner, 'mock/result',
        runtime.table_from(dict(status='stopped', frames=1, request_id='r')))[0] is False
    source = LUA_PATH.read_text(encoding='utf8')
    assert source.count("status('capturing')") == 1
    assert source.count('writeCameraTerminal(active, active.directory') == 4
    assert source.count('writeCameraTerminal(owner, owner.directory') == 1
    assert 'now - (active.last_capture_clock or -1e9) <= .15' in source
    close = host_method('close')
    policy = next(n for n in close.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Attribute)
                  and n.test.left.attr == 'camera_status_policy')
    host = HOST_PATH.read_text(encoding='utf8')
    assert host.index('        if self.camera_status_policy is not None:',
                      host.index('    def close(self):')) > host.index('        self.final_clock =')
    for primary in (None, OSError('original cleanup/data error')):
        cleanup = {'producer_terminal_known': True, 'camera_thread_joined': True}
        original = copy.deepcopy(cleanup)
        obj = SimpleNamespace(camera_status_policy=POLICY, camera_status_profile=profile_record(),
            camera_status_first_echo_verified=True, triggered=True, terminal={},
            request_id='r', clock=SimpleNamespace(generation='g'), error=primary,
            owned_cleanup=cleanup)
        result = dict(status='stopped')
        namespace = dict(self=obj, result=result, ECHO_KEY=ECHO_KEY,
                         validate_producer_echo=validate_producer_echo)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[policy], type_ignores=[])),
                     '<actual-end-of-close-policy-validation>', 'exec'), namespace)
        assert result['status'] == 'failed' and cleanup == original
        if primary is not None:
            assert obj.error is primary
        assert result['camera_status_policy_validation']['terminal_echo_verified'] is False
    validate_producer_echo(dict(schema=ECHO_SCHEMA, policy=POLICY, request_id='r', generation='g'),
                           policy=POLICY, request_id='r', generation='g')
