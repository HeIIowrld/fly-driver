"""Create a unique hash-verified local code snapshot; datasets and game files stay in place."""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
                                         source_root=str(root), seed=862879))
    manifest = json.loads((target/'manifest.json').read_text(encoding='utf8'))
    hashes = dict(manifest['source_sha256'])
    tests = {p.relative_to(root).as_posix(): sha256(p)
             for p in sorted((root/'tests').rglob('*.py')) if '__pycache__' not in p.parts}

    def copy_one(item):
        relative, expected = item
        file = target/relative
        file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/relative, file)
        if sha256(file) != expected:
            raise ValueError('Code changed during snapshot: '+relative)

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='runtime-snapshot') as pool:
        list(pool.map(copy_one, {**hashes, **tests}.items()))
    write_json(target/'source-provenance.json', dict(schema='flybrain.local_source_snapshot.v1',
        source_root=str(root), source_revision=manifest['git_revision'],
        source_git_dirty=manifest['git_dirty'], source_sha256=hashes, tests_sha256=tests,
        source_manifest_sha256=sha256(target/'manifest.json'), all_copied_hashes_verified=True,
        scope='code snapshot only; data paths resolve in the original working directory'))
    write_json(target/'result.json', dict(status='completed', source_files=len(hashes),
        test_files=len(tests), source_provenance_sha256=sha256(target/'source-provenance.json'),
        all_copied_hashes_verified=True, original_files_deleted=0, goal_achieved=False))


if __name__ == '__main__':
    main()
