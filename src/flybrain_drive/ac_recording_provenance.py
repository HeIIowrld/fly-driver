"""Resolve one sealed episode to its original recording, verifying intermediate manifests."""

import json
from pathlib import Path

from .graph import sha256


def recording_for_dataset(directory, seen=None):
    root = Path(directory).resolve()
    seen = set() if seen is None else seen
    if root in seen:
        raise ValueError('Cyclic recording provenance')
    seen.add(root)
    manifest = json.loads((root / 'dataset.json').read_text(encoding='utf8'))
    if manifest.get('source_recording_directory'):
        return Path(manifest['source_recording_directory'])
    builder = json.loads((root.parent / 'manifest.json').read_text(encoding='utf8'))
    args = builder['config']['arguments']
    sources = [args[k] for k in ('run', 'recording', 'trial') if k in args]
    if len(sources) == 1:
        return Path(sources[0])
    inherited = [args[k] for k in ('input', 'dataset') if k in args and isinstance(args[k], str)]
    if sources or len(inherited) != 1:
        raise ValueError('One explicit recording or inherited dataset source required')
    parent = Path(inherited[0]).resolve()
    matches = [r for r in manifest.get('sources', [])
               if Path(r['directory']).resolve() == parent]
    if len(matches) != 1 or sha256(parent / 'dataset.json') != matches[0]['sha256']:
        raise ValueError('Inherited dataset manifest differs from provenance')
    return recording_for_dataset(parent, seen)
