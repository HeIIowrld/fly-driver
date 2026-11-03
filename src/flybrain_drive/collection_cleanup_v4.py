"""Fail-closed data/transport outcome separation; no live work on import."""

import json
from pathlib import Path

SCHEMA = 'flybrain.collection.owned_cleanup.v4'


def recorder_cleanup(child, vision):
    """Called only after camera.close/archive returned in the owned recorder."""
    control_terminal = child is None or child.poll() is not None
    camera = dict(vision.owned_cleanup) if vision is not None else dict(
        camera_thread_joined=True, producer_terminal_known=True,
        persistence_joined=True, precompute_joined=True, shared_reader_closed=True)
    return dict(schema=SCHEMA, control_process_created=child is not None,
        control_process_pid=child.pid if child is not None else None,
        control_process_terminal=control_terminal,
        control_process_returncode=child.returncode if child is not None else None,
        request_id=vision.request_id if vision is not None else None,
        camera=camera, complete=control_terminal and all(camera.get(k) is True for k in (
            'camera_thread_joined','producer_terminal_known','persistence_joined',
            'precompute_joined','shared_reader_closed')))


def healthy_failed_data_eligible(context, receipt, *, now_qpc_ns):
    """No inferred absence-of-error proof, no restart, no deadline refresh."""
    if not isinstance(receipt, dict):
        return False
    camera = receipt.get('camera')
    if not isinstance(camera, dict):
        return False
    created = receipt.get('control_process_created')
    if (type(created) is not bool
            or created and (type(receipt.get('control_process_pid')) is not int
                or receipt['control_process_pid'] <= 0
                or type(receipt.get('control_process_returncode')) is not int)
            or not created and (receipt.get('control_process_pid') is not None
                or receipt.get('control_process_returncode') is not None)):
        return False
    if (receipt.get('schema') != SCHEMA
            or receipt.get('complete') is not True
            or receipt.get('control_process_terminal') is not True
            or any(camera.get(k) is not True for k in (
                'camera_thread_joined','producer_terminal_known','persistence_joined',
                'precompute_joined','shared_reader_closed'))
            or context is None or context.state != 'running' or context.error is not None
            or context.stop_event.is_set() or context.lease is None
            or context.writer is None
            or any(t is None or not t.is_alive() for t in (
                context.poll_thread, context.write_thread))
            or context.lease.registration['deadline_qpc_ns']-now_qpc_ns < 8_000_000_000):
        return False
    return True


def finish_owned_transport(context, evidence, fresh_idle, wait_owned):
    """Same standard retire/final/join/SYNC/release; callable only on the same context."""
    def idle_after_cleanup():
        try:
            return fresh_idle()
        except ValueError:
            return None

    wait_owned(context, idle_after_cleanup, 1., 'No verified idle after owned camera cleanup')
    context.request_retire()
    wait_owned(context, lambda: context.last_ack and context.last_ack[1]['phase'] == 'final'
        and context.writer.last_committed_sequence == context.last_ack[1].get('final_sequence'),
        2., 'Final quiescent original sequence not committed')
    context.fence_writer(join_timeout=2.)

    def sync_return():
        from .session_heartbeat_terminal_v2 import observe_sync_return

        return observe_sync_return(context)

    wait_owned(context, sync_return, 2., 'No exact synchronous return acknowledgement')
    evidence.capture_sync(context)
    handoff = context.complete_sync_return()
    evidence.bind_handoff(handoff)
    fresh_idle()
    return handoff


def retire_clean_failed_data(context, evidence, fresh_idle, wait_owned, result,
                            recording_result, *, now_qpc_ns):
    """Data status always remains failed. A stopped context is never resumed."""
    result['status'] = 'failed'
    result['data_outcome'] = ('failed' if isinstance(recording_result, dict)
        and recording_result.get('status') == 'failed' else 'not_admitted_due_to_parent_failure')
    result['fit_allowed'] = False
    result['transport_outcome'] = 'failure_fence_retained'
    if not isinstance(recording_result, dict) or recording_result.get('status') != 'failed':
        return False
    receipt = recording_result.get('owned_cleanup')
    if not healthy_failed_data_eligible(context, receipt, now_qpc_ns=now_qpc_ns):
        return False
    try:
        result['handoff'] = finish_owned_transport(context, evidence, fresh_idle, wait_owned)
        result['transport_outcome'] = 'normally_retired_after_clean_failed_data'
        result['owned_cleanup'] = receipt
        return True
    except BaseException as error:
        result['transport_retirement_error'] = dict(error=type(error).__name__, reason=str(error))
        result['transport_outcome'] = ('normal_release_completed_postcheck_failed'
            if context.state == 'closed' else 'retirement_failed_fence_retained')
        return False  # Existing outer finally fences/joins the same workers and retains the lease.


def load_owned_recording_failure(path):
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, ValueError):
        return None
    return value
