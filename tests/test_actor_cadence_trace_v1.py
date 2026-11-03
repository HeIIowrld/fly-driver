"""UNSEALED prototype CPU fixtures; no original recorder main/device/model calls."""

import ast
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.actor_cadence_trace_v1 import (
    ITERATIONS,
    PHASES,
    ROW,
    SLOTS,
    ActorCadenceProfile,
    ActorTrace,
    ActorTraceSink,
    verify_actor_trace,
)
from scripts.prepare_actor_cadence_trace_v1 import BASES, build


def cleanup():
    return dict(schema='flybrain.collection.owned_cleanup.v4', complete=True,
        control_process_terminal=True, camera={k: True for k in (
            'camera_thread_joined', 'producer_terminal_known', 'persistence_joined',
            'precompute_joined', 'shared_reader_closed')})


def trace():
    counter = [0]

    def clock():
        counter[0] += 100
        return counter[0]

    return ActorTrace(wall=clock, cpu=clock)


def isolated_loop(source, *, new, idle=False, native_timeout=False):
    """Compile ONLY the saved recorder's actual actor with-block with CPU fakes."""
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.With)
                and isinstance(n.items[0].context_expr, ast.Call)
                and isinstance(n.items[0].context_expr.func, ast.Attribute)
                and isinstance(n.items[0].context_expr.func.value, ast.Name)
                and n.items[0].context_expr.func.value.id == 'events_path')
    fn = ast.FunctionDef(name='isolated_actor', args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[node], decorator_list=[])
    code = compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                   '<actual-source-actor-loop-only>', 'exec')
    calls, sleeps, state = [], [], dict(wall=0., polls=0, native=0, decisions=0)

    class Clock:
        def perf_counter(self):
            return state['wall']

        monotonic = perf_counter

        def sleep(self, seconds):
            calls.append('sleep')
            sleeps.append(seconds)
            state['wall'] += seconds

    def poll():
        calls.append('poll')
        state['polls'] += 1
        return None if state['polls'] <= 2 else 0

    def phase(*args, **kwargs):
        calls.append('status')
        state['wall'] += .009
        if idle:
            return None
        return dict(phase='accelerate60', desired_speed_m_s=60/3.6,
                    observed_at_host_seconds=99., source_script_clock=1.)

    class Vision:
        clock = SimpleNamespace(generation='fixture')

        def wait_for_candidate(self, *, timeout):
            assert timeout == .04
            calls.append('candidate_wait')
            state['wall'] += .001

        def candidate_before(self, original):
            calls.append('candidate_select')
            return dict(capture_id=state['native']-1, generation='fixture')

        def host_clock(self):
            calls.append('decision')
            value = 100.+state['decisions']*.05
            state['decisions'] += 1
            return value

        def samples_for(self, candidate):
            calls.append('samples_for')
            return {}

    def native(*args, **kwargs):
        calls.append('native_receive')
        state['wall'] += .001
        state['native'] += 1
        if native_timeout:
            raise TimeoutError('fixture paused native source')
        return SimpleNamespace(fields=dict(gas=.3), received_monotonic_seconds=99.999), None

    def pack(*args, **kwargs):
        calls.append('retinal_pack')
        state['wall'] += .002
        return dict(view_valid=np.ones(4, bool))

    class Stream(io.StringIO):
        def write(self, value):
            calls.append('write')
            state['wall'] += .0005
            return super().write(value)

        def flush(self):
            calls.append('flush')
            state['wall'] += .0005

        def __exit__(self, *args):
            return False  # Keep RAM fixture bytes available, no file operation.

    stream = Stream()
    t = trace()
    namespace = dict(child=SimpleNamespace(poll=poll), time=Clock(), deadline=100.,
        read_ai_phase=phase, ai_is_running=lambda: False, session_status='fixture',
        args=SimpleNamespace(pace_task=True, same_copy_native_authority=True),
        vision=Vision(), reader=object(), receive_recording_native=native,
        retina=SimpleNamespace(pack=pack, required_native_fields={'gas'}, max_age=.15),
        observations=[], times=[], camera_gaps=[], NoFreshCSPFrame=RuntimeError,
        events_path=SimpleNamespace(open=lambda *a, **k: stream),
        native=SimpleNamespace(user=SimpleNamespace(GetForegroundWindow=lambda: 0)),
        windows=[SimpleNamespace(hwnd=1)], physics_record=lambda body: dict(packet_id=7),
        shadow=None, authority_records=0, authority_eligible_records=0,
        json=json, sys=sys, actor_trace=t, actor_profile=ActorCadenceProfile())
    exec(code, namespace)
    namespace['isolated_actor']()
    t.seal()
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    return namespace, calls, sleeps, stream.getvalue(), t, events, state


def test_new_sources_exact_derivatives_and_all_old_bases_immutable():
    import hashlib

    for path, expected in BASES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    for path, raw in build().items():
        assert path.read_bytes() == raw
    text = Path('scripts/record_ac_engine_ai_probe_actor_trace_v1.py').read_text()
    assert text.index('started = time.perf_counter()', text.index('while True:')) < (
        text.index('held_task = read_ai_phase(', text.index('while True:')))
    assert 'period=.04 if args.shared_transport else .03' in text
    assert 'vision.wait_for_candidate(timeout=.04)' in text


def test_actual_loop_call_footprint_and_original_event_bytes_unchanged():
    old = Path('scripts/record_ac_engine_ai_probe_host_v4.py').read_text()
    new = Path('scripts/record_ac_engine_ai_probe_actor_trace_v1.py').read_text()
    a, ac, sleeps_old, raw_old, _, _, old_state = isolated_loop(old, new=False)
    b, bc, sleeps_new, raw_new, t, events, new_state = isolated_loop(new, new=True)
    assert ac == bc and raw_old == raw_new
    assert a['times'] == b['times'] == [100., 100.05]
    assert len(a['observations']) == len(b['observations']) == 2
    assert all(s > n for s, n in zip(sleeps_old, sleeps_new, strict=True))
    assert old_state['wall'] == pytest.approx(.098)
    assert new_state['wall'] == pytest.approx(.06)
    assert t.iterations == 3 and t.decisions == len(events) == 2
    assert t.per_iteration[:3].tolist() == [24, 24, 5]


def test_idle_and_native_timeout_paths_keep_calls_and_original_no_event_behavior():
    old = Path('scripts/record_ac_engine_ai_probe_host_v4.py').read_text()
    new = Path('scripts/record_ac_engine_ai_probe_actor_trace_v1.py').read_text()
    for kwargs in (dict(idle=True), dict(native_timeout=True)):
        a, ac, _, ar, _, _, _ = isolated_loop(old, new=False, **kwargs)
        b, bc, _, br, t, events, _ = isolated_loop(new, new=True, **kwargs)
        assert ac == bc and ar == br == '' and a['times'] == b['times'] == []
        assert t.closed and not events


def saved_trace(tmp_path):
    source = Path('scripts/record_ac_engine_ai_probe_actor_trace_v1.py').read_text()
    _, _, _, _, t, events, _ = isolated_loop(source, new=True)
    sink = ActorTraceSink()
    sink.accept(t, cleanup())
    target = tmp_path/'actor-trace'
    assert sink.persist(target, context=None)['status'] == 'complete'
    return target, events, sink


def test_saved_trace_exact_original_decision_linkage_and_duplicate_refusal(tmp_path):
    target, events, sink = saved_trace(tmp_path)
    result = verify_actor_trace(target, events)
    assert result['decisions'] == 2 and result['records'] == 53
    assert result['CPU_less_than_wall_assumed'] is False
    with pytest.raises(RuntimeError, match='repeated'):
        sink.persist(target, context=None)
    events[1]['decision_monotonic_seconds'] += .001
    with pytest.raises(ValueError, match='event/clock/count'):
        verify_actor_trace(target, events)


def test_trace_cannot_be_substituted_for_another_owned_cleanup_request(tmp_path):
    target, events, _ = saved_trace(tmp_path)
    foreign = cleanup()
    foreign['request_id'] = 'foreign-request'
    with pytest.raises(ValueError, match='same recorder cleanup/request'):
        verify_actor_trace(target, events, expected_cleanup=foreign)


@pytest.mark.parametrize('field,value', [('iterations', True), ('records_per_iteration', [1, 1, 1]),
    ('original_budget_clocks', []), ('binary_bytes', 1000000000)])
def test_independent_trace_verifier_rejects_forged_metadata(tmp_path, field, value):
    target, events, _ = saved_trace(tmp_path)
    path = target/'index.json'
    index = json.loads(path.read_bytes())
    index[field] = value
    path.write_text(json.dumps(index), encoding='utf8')
    with pytest.raises(ValueError):
        verify_actor_trace(target, events)


def test_binary_stage_order_forgery_rejected_even_with_new_sha(tmp_path):
    import hashlib

    target, events, _ = saved_trace(tmp_path)
    path = target/'trace.bin'
    raw = bytearray(path.read_bytes())
    iteration, _, flags, wall, cpu = ROW.unpack_from(raw, ROW.size)
    ROW.pack_into(raw, ROW.size, iteration, PHASES.index('loop_enter'), flags, wall, cpu)
    path.write_bytes(raw)
    index = json.loads((target/'index.json').read_bytes())
    index['binary_sha256'] = hashlib.sha256(raw).hexdigest()
    (target/'index.json').write_text(json.dumps(index), encoding='utf8')
    with pytest.raises(ValueError, match='stage/iteration order'):
        verify_actor_trace(target, events)


def test_unsafe_cleanup_or_live_worker_defers_without_file_or_new_worker(tmp_path):
    source = Path('scripts/record_ac_engine_ai_probe_actor_trace_v1.py').read_text()
    _, _, _, _, t, _, _ = isolated_loop(source, new=True)
    sink = ActorTraceSink()
    bad = cleanup()
    bad['camera']['precompute_joined'] = 1
    sink.accept(t, bad)
    assert sink.persist(tmp_path/'unsafe', context=None)['status'].startswith('deferred')
    assert not (tmp_path/'unsafe').exists()
    sink.cleanup = cleanup()
    context = SimpleNamespace(poll_thread=SimpleNamespace(is_alive=lambda: True), write_thread=None)
    assert sink.persist(tmp_path/'live', context=context)['status'].startswith('deferred')
    assert not (tmp_path/'live').exists()
    context.poll_thread = SimpleNamespace(is_alive=lambda: False)
    assert sink.persist(tmp_path/'safe', context=context)['status'] == 'complete'


def test_fixed_allocation_reserved_before_next_acquisition_no_trace_overwrite():
    t = trace()
    size = len(t.data)
    t.begin_iteration()
    for _ in range(SLOTS-1):
        t.mark('budget_started')
    before = bytes(t.data[:SLOTS*ROW.size])
    t.mark('budget_started')  # Latch failure; do not overwrite previous records.
    with pytest.raises(RuntimeError, match='retained'):
        t.finish_iteration()
    assert bytes(t.data[:SLOTS*ROW.size]) == before and len(t.data) == size
    with pytest.raises(RuntimeError, match='before another acquisition'):
        t.begin_iteration()
    fresh = trace()
    fresh.iterations = ITERATIONS
    with pytest.raises(RuntimeError, match='before another acquisition'):
        fresh.begin_iteration()
    assert fresh.records == 0


def test_profile_cannot_relax_original_bounds_or_change_nominal_silently():
    profile = ActorCadenceProfile().record()
    for key, wrong in [('maximum_actor_interval_seconds', .081),
                       ('maximum_camera_age_seconds', .151),
                       ('maximum_heartbeat_age_seconds', 2.01),
                       ('nominal_actor_seconds', .04)]:
        with pytest.raises(ValueError):
            ActorCadenceProfile.from_record(dict(profile, **{key: wrong}))
    with pytest.raises(ValueError):
        ActorCadenceProfile(.04).record()


def test_cpu_quantization_does_not_require_cpu_delta_less_than_wall():
    t = ActorTrace(wall=lambda: 100, cpu=iter([0, 15625000, 15625000]).__next__)
    t.begin_iteration()
    t.budget_started(1.)
    t.finish_iteration()
    assert t.failure is None and t.records == 3
    t.seal()
    assert ROW.unpack_from(t.data, ROW.size)[4] > ROW.unpack_from(t.data, ROW.size)[3]
