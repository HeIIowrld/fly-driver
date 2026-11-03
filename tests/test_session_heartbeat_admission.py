"""Excluded entrypoints refuse before config/status reads or device acquisition."""

import importlib
import sys
from pathlib import Path

import pytest

from flybrain_drive.session_heartbeat_admission import (
    UnmigratedHeartbeatConsumerError,
    refuse_leased_transport,
    require_leased_collection_budget,
)


@pytest.mark.parametrize('name,args', [
    ('deploy_ac_camera_service', ['--expected-installed-sha256', 'a'*64]),
    ('probe_ac_async_readback', []), ('probe_ac_shared_texture', []),
    ('probe_flight_stationary_steering', ['--plan', 'must-not-open.json']),
    ('qualify_exterior_camera_stream', ['--config', 'must-not-open.json']),
])
def test_active_even_partial_lease_refuses_entry_before_metadata_or_devices(
        tmp_path, monkeypatch, name, args):
    module = importlib.import_module('scripts.'+name)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    root = tmp_path/'flybrain/csp-session'
    root.mkdir(parents=True)
    (root/'status.json.flybrain-heartbeat-owner').write_bytes(b'{partial')
    monkeypatch.setattr(sys, 'argv', [name, *args, '--output', str(tmp_path/'out')])

    def forbidden(*args, **kwargs):
        pytest.fail('Unadmitted consumer read metadata or acquired a device before refusal')

    monkeypatch.setattr(Path, 'open', forbidden)
    monkeypatch.setattr(Path, 'read_text', forbidden)
    if hasattr(module, 'WindowsACWindow'):
        monkeypatch.setattr(module, 'WindowsACWindow', forbidden)
    with pytest.raises(UnmigratedHeartbeatConsumerError, match='unadmitted'):
        module.main()


def test_absent_lease_keeps_default_and_orphan_entry_never_bypasses(tmp_path):
    target = tmp_path/'status.json'
    assert refuse_leased_transport('private fixture', status_path=target) is None
    lock = tmp_path/'status.json.flybrain-heartbeat-owner'
    lock.write_bytes(b'{}')
    with pytest.raises(UnmigratedHeartbeatConsumerError):
        refuse_leased_transport('private fixture', status_path=target)


def test_activation_budget_uses_original_deadline_after_setup_without_refresh(
        tmp_path, monkeypatch):
    from flybrain_drive import session_heartbeat_admission as module
    from flybrain_drive.session_heartbeat_provenance import (
        ProvenanceWriterLease,
        qpc_clock_metadata,
    )

    target = tmp_path/'status.json'
    assert require_leased_collection_budget(target, 'synthetic', 30) is None
    meta = qpc_clock_metadata()
    lease = ProvenanceWriterLease(target, owner_id='a'*32, generation='synthetic',
        calibration_sha256='b'*64, registered_qpc_ns=100_000_000_000,
        deadline_qpc_ns=160_000_000_000, clock_metadata=meta).acquire()
    monkeypatch.setattr(module, 'qpc_clock_metadata', lambda: meta)
    good = require_leased_collection_budget(target, 'synthetic', 30,
                                            now_qpc_ns=115_000_000_000)
    assert good['remaining_seconds'] == 45
    with pytest.raises(TimeoutError, match='never extend'):
        require_leased_collection_budget(target, 'synthetic', 30, now_qpc_ns=115_000_000_001)
    assert lease.registration['deadline_qpc_ns'] == 160_000_000_000
    lease.release()
