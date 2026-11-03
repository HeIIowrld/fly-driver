"""Remove only recent, closed camera pixels proven identical to retained run bundles."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


RUNS = (
    '20261101-dashboard-camera-stationary-30-brightness',
    '20261101-dashboard-camera-owned-idle-v3',
    '20261101-exterior-camera-owned-idle-v3',
    '20261101-fast-body-fourview-probe',
    '20261101-low-speed-fourview-prefix-reverse-probe',
    '20261101-low-speed-fourview-stationary-prefix-probe',
)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    base = Path(os.environ['LOCALAPPDATA'])/'flybrain'
    camera = (base/'csp-camera').resolve()
    status_file = camera/'status.json'
    run = begin_run(args.output, dict(program_sha256=sha256(__file__),
        selected_runs=list(RUNS), execute=args.execute,
        scope='flat pixel duplicates only after exact retained bundle and individual SHA validation'))
    rows = []
    for name in RUNS:
        source = base/'runtime-runs'/name
        receipt = read(source/'result.json')
        if receipt.get('status') not in ('completed_stationary_mmf_transport', 'completed_low_speed_pulse_diagnostic'):
            raise ValueError('Only completed original captures are eligible')
        vision = receipt.get('vision', receipt.get('camera_close', {}))
        if vision.get('status') != 'stopped' or vision.get('worker_alive') is not False:
            raise ValueError('Original camera owner must be stopped')
        request = receipt.get('camera_request_id') or vision.get('producer_terminal', {}).get('request_id')
        if not isinstance(request, str) or not request.startswith(('mmf-stream-', 'async-')):
            raise ValueError('Exact original request identity required')
        status, _ = read_published_json(status_file)
        if status.get('state') == 'capturing' and status.get('request_id') == request:
            raise ValueError('Active source request cannot be compacted')
        archive_info = receipt.get('archive', receipt.get('camera_archive', {}))
        original_root = camera/request
        root = original_root.resolve()
        if root.parent != camera or not root.is_dir() or original_root.is_symlink():
            raise ValueError('Exact source directory under camera root required')
        with CameraArchive(source/'vision') as archive:
            if archive.bundle_sha256 != archive_info.get('bundle_sha256'):
                raise ValueError('Retained bundle hash differs from original result')
            archive.verify_all()
            for entry in sorted(root.iterdir()):
                if entry.suffix.lower() not in ('.png', '.bmp'):
                    continue
                if entry.is_symlink() or not entry.is_file() or entry.resolve().parent != root:
                    raise ValueError('Flat regular source pixel required')
                stored = archive.read(entry.name)
                stat = entry.stat()
                digest = hashlib.sha256(stored).hexdigest()
                if stat.st_size != len(stored) or sha256(entry) != digest:
                    raise ValueError('Source and retained archive differ')
                rows.append(dict(run=name, request_id=request, source=str(entry), bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, sha256=digest, bundle=str(source/'vision'/'camera.zip'),
                    bundle_sha256=archive.bundle_sha256, member=entry.name))
    plan = run/'verified-pixels.json'
    write_json(plan, dict(files=rows, bytes=sum(r['bytes'] for r in rows)))
    # Persist the restore map before removing any file.
    with plan.open('rb+') as f:
        os.fsync(f.fileno())
    deleted = 0
    if args.execute:
        previous_request = None
        with (run/'deleted.jsonl').open('x', encoding='utf8') as log:
            for row in rows:
                if row['request_id'] != previous_request:
                    status, _ = read_published_json(status_file)
                    if status.get('state') == 'capturing' and status.get('request_id') == row['request_id']:
                        raise ValueError('Source became active during cleanup')
                    previous_request = row['request_id']
                path = Path(row['source'])
                if path.is_symlink() or not path.is_file() or path.resolve().parent.parent != camera:
                    raise ValueError('Source identity changed before deletion')
                st = path.stat()
                if st.st_size != row['bytes'] or st.st_mtime_ns != row['mtime_ns'] or sha256(path) != row['sha256']:
                    raise ValueError('Source changed after verification')
                path.unlink()
                log.write(json.dumps(row, ensure_ascii=False)+'\n')
                log.flush()
                os.fsync(log.fileno())
                deleted += 1
    write_json(run/'result.json', dict(status='completed', verified=len(rows), deleted=deleted,
        verified_bytes=sum(r['bytes'] for r in rows), retained_bundles=True, source_metadata_retained=True,
        plan_sha256=sha256(plan), execute=args.execute))
    print(json.dumps(read(run/'result.json')))


if __name__ == '__main__':
    main()
