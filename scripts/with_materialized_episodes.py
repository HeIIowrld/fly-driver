"""Run an old reader with temporarily restored, byte-identical episode paths."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from flybrain_drive.episode_storage import LINKS_NAME, _raw_sha256, resolve_payload_path


def restore_dataset(directory):
    root = Path(directory).resolve()
    links = json.loads((root / LINKS_NAME).read_text(encoding='utf8'))
    created = []
    try:
        for name, entry in links['entries'].items():
            if Path(name).name != name:
                raise ValueError('Only contained episode filenames may be restored')
            alias = root / name
            if alias.exists():
                continue
            original = resolve_payload_path(alias)
            temporary = alias.with_name(alias.name + '.restore-partial')
            with temporary.open('xb') as out, original.open('rb') as src:
                shutil.copyfileobj(src, out, 4 * 1024 * 1024)
            if _raw_sha256(temporary) != entry['sha256']:
                temporary.unlink()
                raise ValueError('Restored bytes differ')
            # Windows rename refuses an existing destination; no user file is overwritten.
            if alias.exists():
                temporary.unlink()
                raise FileExistsError(alias)
            os.rename(temporary, alias)
            created.append((alias, entry['sha256']))
    except BaseException:
        remove_restored(created)
        raise
    return created


def remove_restored(created):
    for path, expected in reversed(created):
        if path.exists():
            if _raw_sha256(path) != expected:
                raise ValueError('Restored payload changed; retaining it: ' + str(path))
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', action='append', required=True)
    parser.add_argument('--restore-only', action='store_true',
                        help='Restore permanently for manual historical execution')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command and not args.restore_only:
        parser.error('Provide -- COMMAND or --restore-only')
    created = []
    try:
        for directory in args.dataset:
            created.extend(restore_dataset(directory))
        if command:
            return subprocess.run(command, check=False).returncode
        return 0
    finally:
        if not args.restore_only:
            remove_restored(created)


if __name__ == '__main__':
    raise SystemExit(main())
