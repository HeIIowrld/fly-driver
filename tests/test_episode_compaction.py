"""Historical manifests and samples survive physical duplicate removal."""

import hashlib
import io
import json
import shutil

import numpy as np
import pytest
import torch
from test_episode_storage import merge, source

from flybrain_drive.episode_storage import load_array, resolve_payload_path
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes
from scripts.with_materialized_episodes import remove_restored, restore_dataset


def compact_fixture(tmp_path):
    runs = tmp_path / 'runs'
    runs.mkdir()
    original = runs / 'original'
    metadata = source(original)
    alias = runs / 'alias'
    shutil.copytree(original, alias)
    raw = (alias / 'dataset.json').read_bytes()
    links = dict(schema='flybrain.episode-links.v1',
                 dataset_sha256=hashlib.sha256(raw).hexdigest(),
                 entries={'episode.npz': dict(target='../original/episode.npz',
                         size=(original / 'episode.npz').stat().st_size,
                         sha256=metadata['episodes'][0]['sha256'])})
    (alias / '.episode-links.json').write_text(json.dumps(links), encoding='utf8')
    return original, alias, metadata, raw


def test_legacy_paths_hashes_samples_and_new_reference_survive(tmp_path, monkeypatch):
    original, alias, metadata, raw = compact_fixture(tmp_path)
    union = original.parent / 'union'
    merge(monkeypatch, [alias, alias], union)
    before, _ = load_episodes(alias, metadata['sensory_calibration'])
    (alias / 'episode.npz').unlink()
    assert sha256(alias / 'episode.npz') == metadata['episodes'][0]['sha256']
    assert resolve_payload_path(alias / 'episode.npz') == original / 'episode.npz'
    after, _ = load_episodes(alias, metadata['sensory_calibration'])
    repeated, _ = load_episodes(union, metadata['sensory_calibration'])
    assert len(repeated) == 2
    for actual in [after[0], *repeated]:
        torch.testing.assert_close(actual['action'], before[0]['action'], rtol=0, atol=0)
        for k in before[0]['observation']:
            torch.testing.assert_close(actual['observation'][k],
                                       before[0]['observation'][k], rtol=0, atol=0)
    assert (alias / 'dataset.json').read_bytes() == raw
    exported = original.parent / 'exported'
    merge(monkeypatch, [alias], exported, 'copy')
    assert len(list(exported.glob('*.npz'))) == 1
    created = restore_dataset(alias)
    assert len(created) == 1
    # Unmodified old NumPy code sees the original path while the wrapper runs.
    with np.load(alias / 'episode.npz', allow_pickle=False) as arrays:
        assert len(arrays['action']) == 3
    remove_restored(created)
    assert not (alias / 'episode.npz').exists()
    assert (alias / 'dataset.json').read_bytes() == raw


def test_links_reject_changed_owner_original_and_path_escape(tmp_path):
    original, alias, metadata, raw = compact_fixture(tmp_path)
    path = alias / 'episode.npz'
    path.unlink()
    owner = alias / 'dataset.json'
    owner.write_bytes(raw + b' ')
    with pytest.raises(ValueError, match='owner dataset'):
        load_array(path)
    owner.write_bytes(raw)
    links_path = alias / '.episode-links.json'
    links = json.loads(links_path.read_text())
    links['entries']['episode.npz']['target'] = '../../outside.npz'
    links_path.write_text(json.dumps(links))
    with pytest.raises(ValueError, match='escapes'):
        load_array(path)
    links['entries']['episode.npz']['target'] = '../original/episode.npz'
    links_path.write_text(json.dumps(links))
    original_path = original / 'episode.npz'
    data = original_path.read_bytes()
    original_path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match='hash differs'):
        load_array(path)
    original_path.unlink()
    with pytest.raises(ValueError, match='missing'):
        load_array(path)


def test_filelike_and_ordinary_arrays_are_unchanged(tmp_path):
    stream = io.BytesIO()
    np.save(stream, np.arange(7))
    stream.seek(0)
    np.testing.assert_array_equal(load_array(stream), np.arange(7))
    path = tmp_path / 'ordinary.npy'
    np.save(path, np.arange(3))
    np.testing.assert_array_equal(load_array(path), np.arange(3))



def test_nested_historical_runs_can_share_the_outer_runs_tree(tmp_path):
    original, alias, metadata, raw = compact_fixture(tmp_path)
    nested = original.parent / 'snapshot' / 'runs' / 'teacher'
    nested.parent.mkdir(parents=True)
    shutil.copytree(alias, nested)
    links_file = nested / '.episode-links.json'
    links = json.loads(links_file.read_text())
    links['entries']['episode.npz']['target'] = '../../../original/episode.npz'
    links_file.write_text(json.dumps(links))
    (nested / 'episode.npz').unlink()
    assert resolve_payload_path(nested / 'episode.npz') == original / 'episode.npz'
