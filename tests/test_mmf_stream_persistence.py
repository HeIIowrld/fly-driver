import hashlib
import json

import pytest

from flybrain_drive.ac_camera_archive import CameraArchive, bundle_camera_files
from flybrain_drive.ac_metadata_mmf import HEADER, encode_packet
from flybrain_drive.immutable_frame_writer import ImmutableFrameWriter, verify_persistence_archive


class Files:
    def __init__(self, root):
        self.root = root

    def read(self, name):
        return (self.root/name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))


def persisted_fixture(root):
    image_files = [(f'0000-{name}.bmp', name.encode())
                   for name in ('front', 'left', 'right', 'rear')]
    frame = dict(index=0, request_id='owned', generation='session',
        transport='shared_immutable_v1', metadata_transport='mmf_slot_sha256_v1', views=[
            dict(view_id=name, image_writer='host_lossless_bmp', shared_texture_handle=index+1)
            for index, name in enumerate(('front', 'left', 'right', 'rear'))])
    ack = dict(capture_id=0, request_id='owned', generation='session', copied_before_ack=2.)
    raw, ack_raw = json.dumps(frame).encode(), json.dumps(ack).encode()
    slots = encode_packet(0, raw)+encode_packet(0, ack_raw)
    row = dict(capture_id=0, generation='session', received=3.,
        metadata_transport='mmf_slot_sha256_v1', file_sha256=hashlib.sha256(raw).hexdigest(),
        views=[dict(view_id=name, file='vision/'+file,
                    file_sha256=hashlib.sha256(content).hexdigest(), render_started_lower=.5)
               for (file, content), name in zip(image_files,
                   ('front', 'left', 'right', 'rear'), strict=True)],
        shared_copy_receipts=[dict(view_id=name, started_at=1., finished_at=1.5,
            source_texture_modified=False, original_handle=i+1, description=dict(
                Width=320, Height=180, MipLevels=1, ArraySize=1, SampleCount=1, Format=28))
            for i, name in enumerate(('front', 'left', 'right', 'rear'))],
        metadata_mmf_receipt=dict(sequence=2, started_at=.7, finished_at=.8,
            slots_sha256=hashlib.sha256(slots).hexdigest(),
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            producer_slot_modified=False, repeated_slot_copies_equal=True),
        shared_pipeline=dict(revision='copy_all_then_mmf_ack_then_async_persist_v3',
            stream_schema='flybrain.csp.immutable_rgb_stream.v3',
            acknowledged_copy_clock=2., persistence_started_clock=2.1,
            immutable_memory_ready_clock=3., actor_published_only_after_persistence=False,
            persistence_required_before_run_admission=True))
    writer = ImmutableFrameWriter(root, root/'stream.jsonl', clock=lambda: 4., mmf_metadata=True)
    writer.submit(0, image_files, (json.dumps(row)+'\n').encode(), metadata_files=[
        ('frame-0000.json', raw), ('ack-0000.json', ack_raw), ('mmf-0000.mmf', slots)])
    writer.close()
    (root/'frame-persistence-journal.json').write_text(json.dumps(writer.statistics()))
    return row


def test_closed_v3_requires_all_seven_original_files_and_survives_bundle(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    row = persisted_fixture(source)
    assert verify_persistence_archive(Files(source), expected_mmf=True)[0] == row
    with pytest.raises(ValueError, match='journal'):
        verify_persistence_archive(Files(source))
    bundle_camera_files([source], tmp_path/'bundle')
    with CameraArchive(tmp_path/'bundle') as archive:
        assert verify_persistence_archive(archive, expected_mmf=True)[0] == row
        assert archive.read('mmf-0000.mmf') == (source/'mmf-0000.mmf').read_bytes()


def test_corrupt_actual_slot_rejected_even_when_json_is_intact(tmp_path):
    persisted_fixture(tmp_path)
    slots = bytearray((tmp_path/'mmf-0000.mmf').read_bytes())
    slots[HEADER.size+2] ^= 1
    (tmp_path/'mmf-0000.mmf').write_bytes(slots)
    with pytest.raises(ValueError, match='checksum'):
        verify_persistence_archive(Files(tmp_path), expected_mmf=True)


def test_missing_ack_is_rejected_before_queue_submission(tmp_path):
    writer = ImmutableFrameWriter(tmp_path, tmp_path/'stream.jsonl', mmf_metadata=True)
    try:
        with pytest.raises(ValueError, match='two raw slots'):
            writer.submit(0, [(str(i)+'.bmp', b'rgb') for i in range(4)], b'{}\n',
                          metadata_files=[('frame-0000.json', b'{}')])
        assert writer.submitted == 0
    finally:
        writer.close()
