"""Reference storage must preserve training samples and reject broken provenance."""

import json
import shutil
import sys

import numpy as np
import pytest
import torch

from flybrain_drive.episode_storage import reference_record, resolve_episode_path
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes
from scripts.merge_multiview_teacher import main


def source(path):
    path.mkdir()
    obs = {
        'view_rgb': np.ones((3, 1, 1, 3), np.float32) * .2,
        'view_contrast': np.zeros((3, 1, 1, 3), np.float32),
        'view_valid': np.ones((3, 1, 1), np.float32),
        'contrast_valid': np.zeros((3, 1, 1), np.float32),
        'view_age': np.zeros((3, 1), np.float32),
        'organs': np.zeros((3, 1), np.float32),
        'organ_valid': np.zeros((3, 1), np.float32),
    }
    payload = path / 'episode.npz'
    np.savez(payload, **{'obs_' + k: v for k, v in obs.items()},
             action=np.zeros((3, 3)), action_valid=np.array([[1, 0, 0]] * 3),
             observation_time=np.arange(3) * .04)
    manifest = dict(schema='flybrain.multiview.teacher.v1',
                    action_profile='mujoco_curvature_proxy_only',
                    sensory_calibration={'view_order': ['front']}, actor_fields=sorted(obs),
                    decisions=3, decision_dt_seconds=.04, brain_ticks_per_decision=4,
                    episodes=[dict(episode_id='original', file='episode.npz',
                                   sha256=sha256(payload), steps=3)])
    (path / 'dataset.json').write_text(json.dumps(manifest), encoding='utf8')
    return manifest


def merge(monkeypatch, inputs, output, storage='reference'):
    argv = ['merge', '--output', str(output), '--storage', storage]
    for path in inputs:
        argv += ['--input', str(path)]
    monkeypatch.setattr(sys, 'argv', argv)
    main()
    return json.loads((output / 'dataset.json').read_text(encoding='utf8'))


def test_reference_union_nested_relocation_and_copy_export(tmp_path, monkeypatch):
    left = tmp_path / 'left'
    original = source(left)
    sealed = (left / 'dataset.json').read_bytes()
    # Intentionally retain repeated logical samples and their weights.
    union = tmp_path / 'union'
    merged = merge(monkeypatch, [left, left], union)
    assert not list(union.glob('*.npz'))
    assert merged['decisions'] == 6
    assert len({r['episode_id'] for r in merged['episodes']}) == 2
    actual, _ = load_episodes(union, original['sensory_calibration'])
    baseline, _ = load_episodes(left, original['sensory_calibration'])
    for episode in actual:
        for key in ('action', 'action_valid'):
            torch.testing.assert_close(episode[key], baseline[0][key], rtol=0, atol=0)
        for key, value in episode['observation'].items():
            torch.testing.assert_close(value, baseline[0]['observation'][key], rtol=0, atol=0)
    nested = tmp_path / 'nested'
    merge(monkeypatch, [union], nested)
    assert len(load_episodes(nested, original['sensory_calibration'])[0]) == 2
    export = tmp_path / 'export'
    portable = merge(monkeypatch, [nested], export, 'copy')
    assert len(list(export.glob('*.npz'))) == 2
    assert all('source_reference' not in r for r in portable['episodes'])
    assert (left / 'dataset.json').read_bytes() == sealed
    moved = tmp_path / 'relocated'
    moved.mkdir()
    for path in (left, union, nested):
        shutil.move(str(path), moved / path.name)
    assert len(load_episodes(moved / 'nested', original['sensory_calibration'])[0]) == 2
    assert len(load_episodes(export, original['sensory_calibration'])[0]) == 2


def test_changed_manifest_missing_source_and_changed_payload_are_rejected(tmp_path, monkeypatch):
    left = tmp_path / 'left'
    original = source(left)
    out = tmp_path / 'out'
    merge(monkeypatch, [left], out)
    manifest_file = left / 'dataset.json'
    raw = manifest_file.read_bytes()
    manifest_file.write_bytes(raw + b' ')
    with pytest.raises(ValueError, match='manifest hash'):
        load_episodes(out, original['sensory_calibration'])
    manifest_file.write_bytes(raw)
    with (left / 'episode.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='path/hash'):
        load_episodes(out, original['sensory_calibration'])
    manifest_file.unlink()
    with pytest.raises(FileNotFoundError):
        load_episodes(out, original['sensory_calibration'])


def test_reference_path_scope_and_contract_guards(tmp_path):
    root = tmp_path / 'datasets'
    root.mkdir()
    left = root / 'left'
    manifest = source(left)
    with pytest.raises(ValueError, match='sibling tree'):
        reference_record(left, manifest, 0, tmp_path / 'elsewhere' / 'out')
    out = root / 'out'
    record = reference_record(left, manifest, 0, out)
    bad = dict(manifest, brain_ticks_per_decision=99)
    with pytest.raises(ValueError, match='contract'):
        resolve_episode_path(out, record, bad)
    changed = dict(record, file='../escape.npz')
    with pytest.raises(ValueError, match='sealed source'):
        resolve_episode_path(out, changed, manifest)
    legacy = dict(manifest['episodes'][0], file='../escape.npz')
    with pytest.raises(ValueError, match='escapes'):
        resolve_episode_path(left, legacy, manifest)
    record['source_reference']['dataset'] = '../../outside'
    with pytest.raises(ValueError, match='sibling dataset'):
        resolve_episode_path(out, record, manifest)
