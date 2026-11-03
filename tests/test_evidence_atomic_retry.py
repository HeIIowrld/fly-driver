"""Owned run output may retry transient Windows sharing, never ignore permanent failure."""

import json
from pathlib import Path

import pytest

from flybrain_drive.evidence import write_json


def test_atomic_replace_retries_transient_windows_read_share(tmp_path, monkeypatch):
    real = Path.replace
    attempts = []

    def replace(path, target):
        attempts.append(path)
        if len(attempts) < 3:
            error = PermissionError('transient share conflict')
            error.winerror = 32
            raise error
        return real(path, target)

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr('flybrain_drive.evidence.time.sleep', lambda _: None)
    output = tmp_path / 'heartbeat.json'
    write_json(output, dict(epoch=1))
    assert json.loads(output.read_text(encoding='utf8')) == dict(epoch=1)
    assert len(attempts) == 3 and not output.with_suffix('.json.partial').exists()


@pytest.mark.parametrize('winerror,attempts', [(5, 7), (None, 1)])
def test_permanent_denial_is_bounded_and_preserves_pending_content(tmp_path, monkeypatch,
                                                                 winerror, attempts):
    called = []

    def replace(path, target):
        called.append(path)
        error = PermissionError('permanent denial')
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr('flybrain_drive.evidence.time.sleep', lambda _: None)
    output = tmp_path / 'heartbeat.json'
    with pytest.raises(PermissionError):
        write_json(output, dict(epoch=1))
    assert len(called) == attempts and not output.exists()
    pending = output.with_suffix('.json.partial')
    assert json.loads(pending.read_text(encoding='utf8')) == dict(epoch=1)
