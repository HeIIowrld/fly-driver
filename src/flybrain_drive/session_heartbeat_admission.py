"""Fail-closed admission for legacy direct readers; no status JSON is opened."""

import os
import time
from pathlib import Path

from .session_heartbeat_provenance import (
    LEASE_SCHEMA,
    QPC_DOMAIN,
    capture_lease,
    lease_path,
    qpc_clock_metadata,
)


class UnmigratedHeartbeatConsumerError(RuntimeError):
    pass


def refuse_leased_transport(operation, *, status_path=None):
    """A partial, expired, corrupt or orphan lease also refuses admission.

    No source-age inference, stale-lock deletion or automatic fallback. Retain
    original no-lease behavior. Call before direct metadata reads/acquisitions.
    """
    target = (Path(status_path) if status_path is not None else
              Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json')
    try:
        lease_path(target).lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise UnmigratedHeartbeatConsumerError(
            operation+': cannot establish absence of heartbeat transport lease') from error
    raise UnmigratedHeartbeatConsumerError(
        operation+': direct session reader is unadmitted while heartbeat lease exists')


def require_leased_collection_budget(status_path, generation, seconds, *, now_qpc_ns=None):
    """Final pre-activation guard, after camera and original native priming.

    No lease is the exact original path. A leased3s/30s activation must leave
    its full duration plus13s original cleanup and2s handoff on the same clock.
    """
    captured = capture_lease(status_path)
    if captured is None:
        return None
    _, lease = captured
    now = time.perf_counter_ns() if now_qpc_ns is None else now_qpc_ns
    deadline, registered = lease.get('deadline_qpc_ns'), lease.get('registered_qpc_ns')
    if (seconds not in (3, 30) or lease.get('schema') != LEASE_SCHEMA
            or lease.get('generation') != generation or lease.get('clock_domain') != QPC_DOMAIN
            or lease.get('clock_metadata') != qpc_clock_metadata()
            or type(now) is not int or type(deadline) is not int or type(registered) is not int
            or not registered <= now <= deadline or not 0 < deadline-registered <= 60_000_000_000):
        raise ValueError('Same-generation original finite QPC lease required before activation')
    remaining = (deadline-now)/1e9
    if remaining < seconds+13+2:
        raise TimeoutError('Insufficient original lease at activation; never extend or retry')
    return dict(owner_id=lease['owner_id'], generation=generation,
        checked_qpc_ns=now, deadline_qpc_ns=deadline, remaining_seconds=remaining,
        collection_seconds=seconds, cleanup_seconds=13, handoff_seconds=2)
