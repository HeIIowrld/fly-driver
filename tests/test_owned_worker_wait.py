"""Synthetic ownership/wait protocol, no native process termination in these tests."""

import json
from types import SimpleNamespace

import pytest

from flybrain_drive.owned_worker_wait import MOVING, safe_recorded_failure, wait_owned_worker


@pytest.mark.parametrize('pending_exit, released', [(False, True), (True, False), (True, True)])
def test_cleanup_requires_both_explicit_safe_failure_and_pending_exit(
        tmp_path, pending_exit, released):
    result_file, receipt_file = tmp_path/'result.json', tmp_path/'wait.json'
    result_file.write_text(json.dumps(dict(status='failed', actuator_messages_sent=5,
        released_neutral_and_pause_verified=released, timed_worker_alive=False,
        cleanup_errors=[], vision=dict(worker_alive=False))), encoding='utf8')
    observed = dict(samples=0, terminations=0, closed=False)

    def snapshot():
        observed['samples'] += 1
        finished = observed['terminations'] > 0 or observed['samples'] >= 3
        return dict(pid=17, active=not finished, exit_requested=pending_exit and not finished,
                    exit_code=1 if finished else None, termination_signaled=finished)

    def terminate():
        observed['terminations'] += 1

    child = SimpleNamespace(pid=17, terminate=terminate, wait=lambda: 1)
    receipt = SimpleNamespace(snapshot=snapshot,
                              close=lambda: observed.update(closed=True))
    code = wait_owned_worker(child, module=MOVING, result_file=result_file,
        receipt_file=receipt_file, receipt_factory=lambda pid: receipt, poll_seconds=0)
    assert code == 1 and observed['closed']
    assert observed['terminations'] == int(pending_exit and released)
    recorded = json.loads(receipt_file.read_text(encoding='utf8'))
    assert recorded['process']['termination_signaled'] and not recorded['worker_restarted']


def test_unreleased_or_unknown_lifecycle_worker_never_qualifies_for_forced_cleanup():
    base = dict(status='failed', released_neutral_and_pause_verified=True,
                timed_worker_alive=False, cleanup_errors=[], vision=dict(worker_alive=False))
    assert safe_recorded_failure(MOVING, base)
    assert not safe_recorded_failure(MOVING, dict(base, live_lifecycle_client_pid=123))
    assert not safe_recorded_failure(MOVING, dict(base, cleanup_errors=['pause unknown']))
    assert not safe_recorded_failure(MOVING, dict(base, vision=dict(worker_alive=True)))
    assert not safe_recorded_failure('scripts.other_user_work',
                                     dict(base, actuator_messages_sent=0))
