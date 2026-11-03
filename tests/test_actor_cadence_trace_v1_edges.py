"""Independent trace-admission edge cases; only CPU fakes and pytest local files."""

import json

import pytest
from test_actor_cadence_trace_v1 import saved_trace, trace

from flybrain_drive.actor_cadence_trace_v1 import ActorTrace, verify_actor_trace


@pytest.mark.parametrize('field,value', [
    ('iterations', 0), ('iterations', 8193), ('records', True),
    ('complete', 1), ('failure', 'retained original trace failure'),
    ('binary_file', '../foreign.bin'), ('decisions', 1),
    ('serialized_after_owned_workers_terminal', 1),
    ('original_decision_clocks', [100., 100.051]),
    ('requested_sleep_seconds', [-1., -1., -1.]),
    ('phases', ['unknown_stage']),
])
def test_strict_admission_rejects_cap_type_path_clock_and_handoff_forgeries(
        tmp_path, field, value):
    directory, events, _ = saved_trace(tmp_path)
    path = directory/'index.json'
    header = json.loads(path.read_bytes())
    header[field] = value
    path.write_text(json.dumps(header), encoding='utf8')
    with pytest.raises(ValueError):
        verify_actor_trace(directory, events)


@pytest.mark.parametrize('which', ['wall', 'cpu'])
def test_decreasing_diagnostic_clock_latches_without_overwriting_old_records(which):
    clocks = iter([100, 99])
    t = ActorTrace(wall=clocks.__next__ if which == 'wall' else lambda: 100,
                   cpu=clocks.__next__ if which == 'cpu' else lambda: 100)
    t.begin_iteration()
    original = bytes(t.data[:24])
    t.budget_started(1.)
    assert t.failure and t.records == 1 and bytes(t.data[:24]) == original
    with pytest.raises(RuntimeError, match='retained'):
        t.finish_iteration()
    assert t.active is None


def test_sealed_or_foreign_thread_trace_cannot_accept_more_rows():
    t = trace()
    t.begin_iteration()
    original_owner = t.owner
    t.owner += 1
    with pytest.raises(RuntimeError, match='same-thread'):
        t.mark('budget_started')
    t.owner = original_owner
    t.finish_iteration()
    t.seal()
    with pytest.raises(RuntimeError, match='closed'):
        t.begin_iteration()


def test_secondary_trace_failure_preserves_primary_source_exception():
    t = trace()
    t.begin_iteration()
    t._latch('secondary diagnostic failure')
    with pytest.raises(OSError, match='primary event flush failure'):
        try:
            raise OSError('primary event flush failure')
        finally:
            t.finish_iteration(original_error_active=True)
    assert t.active is None and t.failure == 'secondary diagnostic failure'
