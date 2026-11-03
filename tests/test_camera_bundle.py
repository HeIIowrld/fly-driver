import json

import pytest

from flybrain_drive.ac_camera_archive import CameraArchive, bundle_camera_files
from flybrain_drive.ac_csp_vision import archive_camera_files


def source_files(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'frame.png').write_bytes(b'image bytes retained exactly')
    (source/'frame.json').write_text('{"index":12}')
    return source


def test_bundle_and_legacy_reads_agree_without_extracting_members(tmp_path):
    source = source_files(tmp_path)
    packed, legacy = tmp_path/'packed', tmp_path/'legacy'
    receipt = bundle_camera_files([source], packed)
    archive_camera_files([source], legacy)
    assert {p.name for p in packed.iterdir()} == {'camera.zip', 'archive.json', 'bundle.json'}
    with CameraArchive(packed) as bundle, CameraArchive(legacy) as plain:
        bundle.verify_all()
        plain.verify_all()
        assert bundle.bundle_sha256 == receipt['bundle_sha256']
        staging = bundle.temporary.name
        for name in ('frame.png', 'frame.json'):
            assert bundle.read(name) == plain.read(name) == (source/name).read_bytes()
        assert bundle.json('frame.json') == {'index': 12}
        with pytest.raises(ValueError):
            bundle.read('../frame.png')
        with pytest.raises(ValueError):
            bundle.read('missing.png')
    from pathlib import Path
    assert not Path(staging).exists()
    assert (source/'frame.png').exists()


def test_corrupt_bundle_or_manifest_is_rejected(tmp_path):
    source = source_files(tmp_path)
    packed = tmp_path/'packed'
    bundle_camera_files([source], packed)
    with (packed/'camera.zip').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='staging hash'):
        CameraArchive(packed)
    other = tmp_path/'other'
    bundle_camera_files([source], other)
    records = json.loads((other/'archive.json').read_text())
    records[0]['file'] = '../outside.png'
    (other/'archive.json').write_text(json.dumps(records))
    with pytest.raises(ValueError, match='flat relative'):
        CameraArchive(other)


def test_bundle_collision_and_legacy_member_tampering_are_rejected(tmp_path):
    source = source_files(tmp_path)
    other = tmp_path/'other'
    other.mkdir()
    (other/'frame.png').write_bytes(b'collision')
    destination = tmp_path/'bad'
    with pytest.raises(ValueError, match='collision'):
        bundle_camera_files([source, other], destination)
    assert not (destination/'bundle.json').exists()
    legacy = tmp_path/'legacy'
    archive_camera_files([source], legacy)
    (legacy/'frame.png').write_bytes(b'altered')
    with CameraArchive(legacy) as reader:
        with pytest.raises(ValueError, match='member hash'):
            reader.verify_all()
