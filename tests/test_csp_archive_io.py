import json
from types import SimpleNamespace

import pytest

from flybrain_drive.ac_csp_vision import CSPVisionRecorder, archive_camera_files
from flybrain_drive.graph import sha256


def test_parallel_archive_preserves_files_hashes_order_and_retains_sources(tmp_path):
    first, second, target = (tmp_path/n for n in ('producer', 'consumer', 'archive'))
    first.mkdir()
    second.mkdir()
    for name, source in [('b.png', first), ('a.json', first), ('clock.json', second)]:
        (source/name).write_bytes(name.encode()*30)
    (first/'ignored.txt').write_text('not a camera artifact')
    result = archive_camera_files([first, second], target)
    assert result['all_hashes_verified'] and result['files'] == 3
    records = json.loads((target/'archive.json').read_text())
    assert [r['file'] for r in records] == ['a.json', 'b.png', 'clock.json']
    for record in records:
        source = first/record['file']
        if not source.exists():
            source = second/record['file']
        assert record['sha256'] == sha256(source) == sha256(target/record['file'])
        assert record['bytes'] == (target/record['file']).stat().st_size
    with pytest.raises(FileExistsError):
        archive_camera_files([first], target)


def test_collisions_and_live_consumers_cannot_create_a_valid_archive(tmp_path):
    first, second = (tmp_path/n for n in ('a', 'b'))
    first.mkdir()
    second.mkdir()
    (first/'same.png').write_bytes(b'first')
    (second/'same.png').write_bytes(b'second')
    target = tmp_path/'collision'
    with pytest.raises(ValueError, match='collision'):
        archive_camera_files([first, second], target)
    assert not (target/'archive.json').exists()
    assert not list(target.iterdir())
    recorder = object.__new__(CSPVisionRecorder)
    recorder.thread = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(RuntimeError, match='live'):
        recorder.archive_to(tmp_path/'live')
    assert not (tmp_path/'live').exists()


def test_corrupt_copy_is_rejected_and_source_is_retained(tmp_path, monkeypatch):
    from flybrain_drive import ac_csp_vision

    source = tmp_path/'source'
    source.mkdir()
    (source/'image.png').write_bytes(b'original')
    monkeypatch.setattr(ac_csp_vision.shutil, 'copyfile',
                        lambda src, dst: dst.write_bytes(b'corrupted'))
    target = tmp_path/'bad'
    with pytest.raises(ValueError, match='hash mismatch'):
        archive_camera_files([source], target)
    assert (source/'image.png').read_bytes() == b'original'
    assert not (target/'archive.json').exists()
