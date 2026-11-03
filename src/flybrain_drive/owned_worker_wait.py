"""Wait for actual termination; reclaim only explicitly failed, safely released owned children."""

import json
import time
from pathlib import Path

from .evidence import write_json
from .graph import sha256
from .win_process_receipt import ProcessReceipt

OFFLINE = {'scripts.verify_flight_yaw_trial', 'scripts.audit_flight_physical_yaw_trial',
           'scripts.verify_flight_fast_stream_trial'}
MOVING = 'scripts.probe_flight_moving_body'


def safe_recorded_failure(module, source):
    if source.get('status') != 'failed':
        return False
    if module in OFFLINE:
        return source.get('actuator_messages_sent') == 0
    if module == MOVING:
        return (source.get('released_neutral_and_pause_verified') is True
                and source.get('timed_worker_alive') is False
                and not source.get('cleanup_errors')
                and not source.get('live_lifecycle_client_pid')
                and (source.get('vision') or {}).get('worker_alive') is False)
    return False


def wait_owned_worker(child, *, module, result_file, receipt_file,
                      receipt_factory=ProcessReceipt, poll_seconds=.05):
    """No runtime timeout or replacement. A failure plus pending teardown can be reclaimed once."""
    if module not in OFFLINE | {MOVING}:
        raise ValueError('Explicit reviewed worker scope required')
    receipt = receipt_factory(child.pid)
    cleanup = None
    try:
        while True:
            state = receipt.snapshot()
            if not state['active']:
                code = child.wait()
                write_json(receipt_file, dict(status='observed_worker_termination',
                    process=state, child_returncode=code, failure_cleanup=cleanup,
                    worker_restarted=False))
                return code
            if state.get('exit_requested') and cleanup is None:
                try:
                    failure = json.loads(Path(result_file).read_text(encoding='utf8'))
                except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                    failure = None
                if failure is not None and safe_recorded_failure(module, failure):
                    cleanup = dict(reason='Explicit failed receipt and pending process teardown',
                        before=state, failure_result_sha256=sha256(result_file), module=module)
                    child.terminate()  # Exact original Popen handle, never PID-based broad cleanup.
                    deadline = time.perf_counter()+2
                    while time.perf_counter() < deadline and receipt.snapshot()['active']:
                        time.sleep(.01)
                    after = receipt.snapshot()
                    cleanup['after'] = after
                    if after['active']:
                        write_json(receipt_file, dict(status='failure_cleanup_outcome_unknown',
                            process=after, failure_cleanup=cleanup, worker_restarted=False))
                        raise RuntimeError('Same failed worker teardown remains unobserved')
            time.sleep(poll_seconds)
    finally:
        receipt.close()
