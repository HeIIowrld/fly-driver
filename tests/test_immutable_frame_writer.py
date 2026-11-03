import hashlib
import json
import threading

import pytest

from flybrain_drive.immutable_frame_writer import ImmutableFrameWriter, verify_persistence_archive


def files(index):
    return [(f'{index:04d}-{name}.bmp', name.encode())
            for name in ('front', 'left', 'right', 'rear')]


def test_blocked_disk_does_not_block_receipt_and_full_queue_never_drops(tmp_path, monkeypatch):
    writer = ImmutableFrameWriter(tmp_path, tmp_path/'stream.jsonl', maximum_pending=2)
    started, release = threading.Event(), threading.Event()
    persist = writer._persist

    def blocked(*args):
        started.set()
        assert release.wait(3)
        return persist(*args)

    monkeypatch.setattr(writer, '_persist', blocked)
    try:
        writer.submit(0, files(0), b'first\n')
        assert started.wait(1)
        writer.submit(1, files(1), b'second\n')
        assert not (tmp_path/'0000-front.bmp').exists()
        with pytest.raises(BufferError, match='silently dropped'):
            writer.submit(2, files(2), b'third\n')
    finally:
        release.set()
        writer.close()
    assert (tmp_path/'stream.jsonl').read_bytes() == b'first\nsecond\n'
    assert writer.statistics()['completed'] == 2
    assert not (tmp_path/'0002-front.bmp').exists()


def test_existing_file_failure_invalidates_run_and_does_not_overwrite(tmp_path):
    existing = tmp_path/'0000-front.bmp'
    existing.write_bytes(b'original evidence')
    writer = ImmutableFrameWriter(tmp_path, tmp_path/'stream.jsonl')
    writer.submit(0, files(0), b'not committed\n')
    with pytest.raises(RuntimeError, match='persistence failed'):
        writer.close()
    assert existing.read_bytes() == b'original evidence'
    assert writer.statistics()['failed'] == 1
    assert not (tmp_path/'stream.jsonl').exists()


def test_mutable_image_payload_rejected_before_submission(tmp_path):
    writer = ImmutableFrameWriter(tmp_path, tmp_path/'stream.jsonl')
    try:
        payloads = files(0)
        payloads[0] = (payloads[0][0], bytearray(b'changeable'))
        with pytest.raises(ValueError, match='immutable bytes'):
            writer.submit(0, payloads, b'{}\n')
        assert writer.submitted == 0
    finally:
        writer.close()


class Archive:
    def __init__(self, root):
        self.root = root

    def read(self, name):
        return (self.root/name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))


def test_independent_audit_rejects_unfinished_journal_backdating_and_corrupt_bytes(tmp_path):
    writer = ImmutableFrameWriter(tmp_path, tmp_path/'stream.jsonl', clock=lambda: 4.)
    row = dict(capture_id=0, received=3., views=[
        dict(file='vision/'+name, file_sha256=hashlib.sha256(raw).hexdigest())
        for name, raw in files(0)], shared_pipeline=dict(
            revision='copy_all_then_ack_then_async_persist_v2',
            stream_schema='flybrain.csp.immutable_rgb_stream.v2',
            acknowledged_copy_clock=1., persistence_started_clock=2.,
            immutable_memory_ready_clock=3., actor_published_only_after_persistence=False,
            persistence_required_before_run_admission=True))
    writer.submit(0, files(0), (json.dumps(row)+'\n').encode())
    writer.close()
    journal = writer.statistics()
    path = tmp_path/'frame-persistence-journal.json'
    path.write_text(json.dumps(journal))
    assert verify_persistence_archive(Archive(tmp_path))[0] == row
    journal['closed'] = False
    path.write_text(json.dumps(journal))
    with pytest.raises(ValueError, match='incomplete'):
        verify_persistence_archive(Archive(tmp_path))
    journal['closed'] = True
    journal['records'][0]['submitted_at'] = 2.5
    path.write_text(json.dumps(journal))
    with pytest.raises(ValueError, match='clock'):
        verify_persistence_archive(Archive(tmp_path))
    journal['records'][0]['submitted_at'] = 4.
    path.write_text(json.dumps(journal))
    (tmp_path/'0000-front.bmp').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='bytes differ'):
        verify_persistence_archive(Archive(tmp_path))
