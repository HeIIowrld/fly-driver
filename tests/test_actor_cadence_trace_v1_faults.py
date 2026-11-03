"""1873 regressions: diagnostic exceptions and foreign CPU-thread mutations only."""

import struct
import sys
import threading

import pytest

import flybrain_drive.actor_cadence_trace_v1 as trace_module
from flybrain_drive.actor_cadence_trace_v1 import ActorTrace


def faulty_trace(mode, at, monkeypatch):
    wall_calls, cpu_calls, packs = [0], [0], [0]
    real_row = trace_module.ROW

    def wall():
        wall_calls[0] += 1
        if wall_calls[0] >= at:
            if mode == 'wall_throw':
                raise RuntimeError('secondary diagnostic clock failure')
            if mode == 'wall_uint64':
                return 1 << 65
        return wall_calls[0]

    def cpu():
        cpu_calls[0] += 1
        if cpu_calls[0] >= at:
            if mode == 'cpu_throw':
                raise OSError('secondary diagnostic CPU sampler failure')
            if mode == 'cpu_uint64':
                return 1 << 65
        return cpu_calls[0]

    class FaultRow:
        size, format = real_row.size, real_row.format

        @staticmethod
        def pack_into(data, offset, *values):
            packs[0] += 1
            if mode == 'pack_throw' and packs[0] >= at:
                # A packer can leave an unpublished partial slot. No retained
                # prefix may be overwritten or included as a successful row.
                data[offset:offset+real_row.size] = b'X'*real_row.size
                raise struct.error('secondary row packing failure')
            return real_row.pack_into(data, offset, *values)

    monkeypatch.setattr(trace_module, 'ROW', FaultRow)
    return ActorTrace(wall=wall, cpu=cpu)


@pytest.mark.parametrize('mode', ['wall_throw', 'cpu_throw', 'wall_uint64',
                                 'cpu_uint64', 'pack_throw'])
@pytest.mark.parametrize('where', ['span_exit', 'finisher'])
def test_thrown_and_overflow_diagnostics_preserve_primary_error_and_retained_prefix(
        monkeypatch, mode, where):
    t = faulty_trace(mode, 3, monkeypatch)
    t.begin_iteration()
    if where == 'finisher':
        t.budget_started(1.)
    retained = None
    with pytest.raises(OSError, match='^primary event flush failure$'):
        try:
            if where == 'span_exit':
                with t.span('event_flush'):
                    retained = bytes(t.data[:t.records*24])
                    raise OSError('primary event flush failure')
            else:
                retained = bytes(t.data[:t.records*24])
                raise OSError('primary event flush failure')
        finally:
            t.finish_iteration(original_error_active=sys.exc_info()[0] is not None)
    assert t.active is None and t.failure is not None and t.records == 2
    assert bytes(t.data[:t.records*24]) == retained
    t.seal()
    assert t.index()['complete'] is False
    with pytest.raises(RuntimeError, match='before another acquisition'):
        t.begin_iteration()


@pytest.mark.parametrize('mode', ['wall_throw', 'cpu_throw', 'wall_uint64',
                                 'cpu_uint64', 'pack_throw'])
def test_begin_fault_rejects_before_acquisition_and_can_seal_partial_trace(monkeypatch, mode):
    t = faulty_trace(mode, 1, monkeypatch)
    with pytest.raises(RuntimeError, match='no next acquisition'):
        t.begin_iteration()
    assert t.active is None and t.records == t.decisions == 0
    t.seal()
    assert t.index()['complete'] is False


def test_unexpected_mark_fault_in_exit_and_finisher_also_preserves_primary(monkeypatch):
    t = ActorTrace(wall=lambda: 1, cpu=lambda: 1)
    t.begin_iteration()
    original_mark = t.mark
    calls = [0]

    def broken_mark(*args, **kwargs):
        calls[0] += 1
        if calls[0] >= 2:
            raise ArithmeticError('secondary unexpected mark fault')
        return original_mark(*args, **kwargs)

    monkeypatch.setattr(t, 'mark', broken_mark)
    with pytest.raises(OSError, match='^primary event flush failure$'):
        try:
            with t.span('event_flush'):
                raise OSError('primary event flush failure')
        finally:
            t.finish_iteration(original_error_active=True)
    assert t.active is None and t.failure is not None
    t.seal()


def test_initial_unexpected_mark_exception_clears_owner_iteration(monkeypatch):
    t = ActorTrace(wall=lambda: 1, cpu=lambda: 1)

    def broken_mark(*args, **kwargs):
        raise ArithmeticError('secondary initial mark fault')

    monkeypatch.setattr(t, 'mark', broken_mark)
    with pytest.raises(RuntimeError, match='no next acquisition'):
        t.begin_iteration()
    assert t.active is None and t.records == t.decisions == 0
    t.seal()


def state(t):
    return (bytes(t.data[:72]), t.budget_clocks[0], t.decision_clocks[0],
            t.decisions, t.sleep_requests[0], t.active, t.closed, t.failure,
            t.iterations, t.records, t.per_iteration[0])


@pytest.mark.parametrize('operation', ['budget_started', 'decision', 'sleep_requested',
                                     'finish_iteration', 'seal', 'begin_iteration', 'mark'])
def test_actual_foreign_cpu_thread_cannot_mutate_any_owned_trace_state(operation):
    t = ActorTrace(wall=lambda: 1, cpu=lambda: 1)
    t.begin_iteration()
    before = state(t)
    errors = []

    def foreign():
        try:
            if operation == 'budget_started':
                t.budget_started(2.5)
            elif operation == 'decision':
                t.decision(0, 3.5)
            elif operation == 'sleep_requested':
                t.sleep_requested(.005)
            elif operation == 'finish_iteration':
                t.finish_iteration(original_error_active=True)
            elif operation == 'mark':
                t.mark('budget_started')
            else:
                getattr(t, operation)()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join(timeout=2.)
    assert not thread.is_alive() and len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert state(t) == before
    t.finish_iteration()
    t.seal()


@pytest.mark.parametrize('operation', ['budget_started', 'decision', 'sleep_requested'])
def test_closed_mutators_reject_before_changing_arrays_or_failure_flags(operation):
    t = ActorTrace(wall=lambda: 1, cpu=lambda: 1)
    t.begin_iteration()
    t.finish_iteration()
    t.seal()
    before = state(t)
    with pytest.raises(RuntimeError, match='Open same-thread'):
        if operation == 'budget_started':
            t.budget_started(1.)
        elif operation == 'decision':
            t.decision(0, 2.)
        else:
            t.sleep_requested(.01)
    assert state(t) == before
