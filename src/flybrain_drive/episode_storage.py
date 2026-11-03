"""Explicit, hash-sealed episode references; legacy contained files remain supported."""

import copy
import hashlib
import json
import os
from pathlib import Path

CONTRACT_KEYS = (
    'schema', 'action_profile', 'sensory_calibration', 'actor_fields',
    'decision_dt_seconds', 'brain_ticks_per_decision', 'teacher_label_contract',
)


def resolve_episode_path(directory, record, metadata=None, *, _seen=()):
    """Resolve only a sealed reference to a sibling dataset, never arbitrary path escape.

    The immediate source manifest and episode identity are checked at every hop.
    Payload hashes are checked by readers, preserving their original validation order.
    """
    root = Path(directory).resolve()
    name = Path(record['file'])
    if name.is_absolute():
        raise ValueError('Episode path/hash: absolute payload path is forbidden')
    file = (root / name).resolve()
    reference = record.get('source_reference')
    if reference is None:
        if not file.is_relative_to(root):
            raise ValueError('Episode path/hash: payload escapes dataset')
        return resolve_payload_path(file, verify_content=False)
    if (not isinstance(reference, dict)
            or reference.get('kind') != 'dataset_episode_v1'
            or type(reference.get('episode_index')) is not int
            or reference['episode_index'] < 0
            or not isinstance(reference.get('dataset'), str)
            or not isinstance(reference.get('dataset_sha256'), str)):
        raise ValueError('Invalid sealed episode reference')
    source_name = Path(reference['dataset'])
    source = (root / source_name).resolve()
    if (source_name.is_absolute() or source == root
            or not source.is_relative_to(root.parent)
            or not file.is_relative_to(root.parent)):
        raise ValueError('Episode reference must remain inside the sibling dataset tree')
    identity = (str(root), record['episode_id'])
    if identity in _seen or len(_seen) >= 32:
        raise ValueError('Cyclic or excessively deep episode reference')
    manifest_file = source / 'dataset.json'
    raw = manifest_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['dataset_sha256']:
        raise ValueError('Referenced dataset manifest hash changed')
    source_metadata = json.loads(raw)
    if metadata is None:
        metadata = json.loads((root / 'dataset.json').read_text(encoding='utf8'))
    if any(metadata.get(k) != source_metadata.get(k) for k in CONTRACT_KEYS):
        raise ValueError('Referenced dataset sensory/timing/label contract differs')
    try:
        source_record = source_metadata['episodes'][reference['episode_index']]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError('Referenced episode is missing') from error
    if any(record[k] != source_record[k] for k in ('sha256', 'steps')):
        raise ValueError('Referenced episode hash or decision count differs')
    resolved = resolve_episode_path(
        source, source_record, source_metadata, _seen=(*_seen, identity))
    if resolved != resolve_payload_path(file, verify_content=False):
        raise ValueError('Episode path/hash does not match the sealed source')
    return resolved


def reference_record(source, metadata, index, destination):
    """Retain episode order, multiplicity and labels while reusing payload bytes."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or not source.is_relative_to(destination.parent):
        raise ValueError('Reference inputs must be in the output sibling tree; use --storage copy')
    record = copy.deepcopy(metadata['episodes'][index])
    payload = resolve_episode_path(source, record, metadata)
    if not payload.is_relative_to(destination.parent):
        raise ValueError('Reference payload is outside the output sibling tree')
    record['file'] = Path(os.path.relpath(payload, destination)).as_posix()
    record['source_reference'] = {
        'kind': 'dataset_episode_v1',
        'dataset': Path(os.path.relpath(source, destination)).as_posix(),
        'dataset_sha256': _raw_sha256(source / 'dataset.json'),
        'episode_index': index,
    }
    return record


LINKS_NAME = '.episode-links.json'


def _raw_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_payload_path(path, *, verify_content=True):
    """Read a compacted historical payload without changing its sealed manifest.

    Ordinary existing files and file-like inputs retain their previous behavior.
    Only explicitly indexed, missing NPZ files may resolve to a retained original.
    """
    if not isinstance(path, (str, bytes, os.PathLike)):
        return path
    file = Path(os.fsdecode(path)).resolve()
    if file.exists() or file.suffix.lower() != '.npz':
        return file
    links_path = file.parent / LINKS_NAME
    if not links_path.is_file():
        return file
    links = json.loads(links_path.read_text(encoding='utf8'))
    if links.get('schema') != 'flybrain.episode-links.v1':
        raise ValueError('Unsupported compacted episode links')
    entry = links.get('entries', {}).get(file.name)
    if entry is None:
        return file
    raw = (file.parent / 'dataset.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != links.get('dataset_sha256'):
        raise ValueError('Compacted owner dataset manifest changed')
    metadata = json.loads(raw)
    records = [r for r in metadata['episodes']
               if (file.parent / r['file']).resolve() == file]
    if not records or any(r['sha256'] != entry.get('sha256') for r in records):
        raise ValueError('Compacted episode is not sealed by its owner manifest')
    scopes = [p for p in file.parents if p.name == 'runs']
    scope = scopes[-1] if scopes else None
    target_name = Path(entry['target'])
    target = (file.parent / target_name).resolve()
    if (scope is None or target_name.is_absolute() or target == file
            or not target.is_relative_to(scope) or target.suffix.lower() != '.npz'):
        raise ValueError('Compacted payload path escapes the runs tree')
    # Originals never depend on another compacted alias: no cycles or silent fallback.
    if not target.is_file() or target.stat().st_size != entry['size']:
        raise ValueError('Retained original payload is missing or has changed size')
    if verify_content and _raw_sha256(target) != entry['sha256']:
        raise ValueError('Retained original payload hash differs')
    return target


def load_array(path, *args, **kwargs):
    """NumPy reader retaining historical paths and file-like input support."""
    import numpy as np
    return np.load(resolve_payload_path(path), *args, **kwargs)
