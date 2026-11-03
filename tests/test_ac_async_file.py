"""A blocked disk must not stall SDK callbacks or silently redate queued measurements."""

import importlib.util
import threading
from pathlib import Path

import pytest


def load_app():
    path = Path(__file__).resolve().parents[1]/'integrations/ac/flybrain_telemetry/flybrain_telemetry.py'
    spec = importlib.util.spec_from_file_location('async_ac_fixture', path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def test_blocked_writer_keeps_callback_free_and_one_latest_immutable_packet(tmp_path, monkeypatch):
    app = load_app()
    publisher = app.LatestFilePublisher(str(tmp_path))
    entered, release, callbacks_done, final_written = (threading.Event() for _ in range(4))
    rows = []

    def write(name, row):
        rows.append(row)
        if len(rows) == 1:
            entered.set()
            assert release.wait(2)
        if row['steps'] == 3:
            final_written.set()

    monkeypatch.setattr(publisher.file, '_write', write)
    data = {'body': {'load': 12.}}
    publisher.publish(data)
    assert entered.wait(1)
    data['body']['load'] = 99.

    def callbacks():
        publisher.publish({'body': {'load': 13.}})
        publisher.publish({'body': {'load': 14.}})
        callbacks_done.set()

    worker = threading.Thread(target=callbacks)
    worker.start()
    try:
        assert callbacks_done.wait(.5), 'Callback waited for blocked disk I/O'
        assert publisher.pending['steps'] == 3
        assert publisher.pending['publisher_dropped_before_write'] == 1
        assert rows[0]['body']['load'] == 12.
        timestamp = publisher.pending['timestamp_ac']
        release.set()
        assert final_written.wait(1)
        assert [r['steps'] for r in rows] == [1, 3]
        assert rows[1]['timestamp_ac'] == timestamp
        assert rows[1]['body']['load'] == 14.
    finally:
        release.set()
        worker.join(1)
        publisher.close()
    assert not publisher.thread.is_alive()
    with pytest.raises(ValueError, match='closed'):
        publisher.publish({})
