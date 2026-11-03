import json
import threading

import pytest

from flybrain_drive.async_event_archive import AsyncEventArchive


class BlockedDisk:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.rows = []

    def write(self, row):
        self.started.set()
        if not self.release.wait(2):
            raise TimeoutError('Test disk was not released')
        self.rows.append(row)

    def flush(self):
        pass

    def close(self):
        pass


def test_blocked_disk_cannot_block_control_enqueue_and_queue_saturation_is_explicit():
    disk = BlockedDisk()
    archive = AsyncEventArchive(None, capacity=2, stream=disk)
    original = dict(host_time=1., values=[2.])
    try:
        archive.emit(original)
        assert disk.started.wait(1)
        original['values'][0] = 9.
        archive.emit(dict(host_time=2.))
        archive.emit(dict(host_time=3.))
        with pytest.raises(RuntimeError, match='queue is full'):
            archive.emit(dict(host_time=4.))
        assert not disk.release.is_set() and not disk.rows
    finally:
        disk.release.set()
        archive.close()
    assert [json.loads(x)['host_time'] for x in disk.rows] == [1., 2., 3.]
    assert json.loads(disk.rows[0])['values'] == [2.]
    assert archive.statistics()['worker_terminated']
    assert archive.accepted == archive.written == 3


def test_disk_error_is_latched_and_cannot_report_success():
    class FailedDisk(BlockedDisk):
        def write(self, row):
            raise OSError('Disk full')
    archive = AsyncEventArchive(None, stream=FailedDisk())
    archive.emit(dict(host_time=1.))
    archive.thread.join(1)
    with pytest.raises(RuntimeError, match='worker failed'):
        archive.emit(dict(host_time=2.))
    with pytest.raises(RuntimeError, match='worker failed'):
        archive.close()
    assert archive.written == 0 and archive.error is not None


def test_complete_real_archive_preserves_every_record_and_rejects_nonfinite(tmp_path):
    path = tmp_path/'events.jsonl'
    with AsyncEventArchive(path) as archive:
        for i in range(300):
            archive.emit(dict(index=i, host_time=i*.04))
        with pytest.raises(ValueError):
            archive.emit(dict(value=float('nan')))
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x['index'] for x in rows] == list(range(300))
    assert archive.accepted == archive.written == len(rows)
