from pathlib import Path
import json

import pytest

from flybrain_drive.ac_published_state import read_published_json

@pytest.mark.parametrize('created,clock,sequence,valid',[(100,7.,1,True),(98,7.,1,False),(102,7.,1,False),(100,8.,1,False),(100,7.,0,False),(100.5,7.,1,False)])
def test_async_original_creation_age_cannot_be_refreshed_by_late_disk_write(tmp_path,monkeypatch,created,clock,sequence,valid):
    monkeypatch.setattr('flybrain_drive.ac_published_state.time.time',lambda:101.5)
    path=tmp_path/'status.json'
    value=dict(script_clock=7.,status_publication=dict(schema='flybrain.csp.async_periodic_status.v1',
        source_unix_seconds=created,source_script_clock=clock,sequence=sequence,policy='single_inflight_no_pending_queue'))
    path.write_text(json.dumps(value))
    if valid:assert read_published_json(path)[0]==value
    else:
        with pytest.raises(ValueError,match='Original asynchronous'):read_published_json(path)


def test_only_transient_windows_read_sharing_is_retried_and_same_handle_mtime_returned(
        tmp_path, monkeypatch):
    path = tmp_path/'state.json'
    path.write_text('{"paused": true}')
    original, calls = Path.open, []

    def once(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 1:
            error = PermissionError(13, 'sharing conflict')
            error.winerror = 32
            raise error
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', once)
    value, modified = read_published_json(path)
    assert value == dict(paused=True) and modified == path.stat().st_mtime and len(calls) == 2


def test_malformed_bytes_and_persistent_denial_cannot_become_cached_success(tmp_path, monkeypatch):
    path = tmp_path/'bad.json'
    path.write_text('broken')
    with pytest.raises(ValueError):
        read_published_json(path)

    def denied(self, *args, **kwargs):
        error = PermissionError(13, 'persistent denial')
        error.winerror = 32
        raise error

    monkeypatch.setattr(Path, 'open', denied)
    with pytest.raises(PermissionError, match='persistent denial'):
        read_published_json(path)


def test_atomic_replace_gap_can_recover_but_missing_publisher_cannot(tmp_path, monkeypatch):
    path = tmp_path/'state.json'
    path.write_text('{"paused": false}')
    original, calls = Path.open, []

    def gap(self, *args, **kwargs):
        calls.append(self)
        if len(calls) == 1:
            raise FileNotFoundError('temporary replace gap')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', gap)
    assert read_published_json(path)[0] == dict(paused=False)
    with pytest.raises(FileNotFoundError):
        read_published_json(tmp_path/'missing.json')
