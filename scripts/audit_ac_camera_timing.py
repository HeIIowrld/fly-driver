"""Compare original camera metadata timing; no image edits, game actions or relaxed age limits."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite original camera timing required')
    return dict(count=len(values), median=float(np.median(values)),
                p99=float(np.quantile(values, .99)), maximum=float(values.max()))


def inspect(root):
    root = Path(root)
    outcome = json.loads((root/'result.json').read_text(encoding='utf8'))
    with zipfile.ZipFile(root/'vision/camera.zip') as archive:
        clock = json.loads(archive.read('clock-initial.json'))
        rows = [json.loads(s) for s in archive.read('stream.jsonl').decode('utf8').splitlines()]
        metadata = []
        for row in rows:
            raw = archive.read(Path(row['file']).name)
            if hashlib.sha256(raw).hexdigest() != row['file_sha256']:
                raise ValueError('Original camera metadata hash differs')
            frame = json.loads(raw)
            if frame['index'] != row['capture_id'] or frame['generation'] != clock['generation']:
                raise ValueError('Camera source order or generation differs')
            metadata.append(frame)
    received = np.array([r['received'] for r in rows])
    rendered = np.array([min(v['render_started_clock'] for v in f['views']) for f in metadata])
    delivered_age = np.array([r['received']-min(v['render_started_lower'] for v in r['views'])
                              for r in rows])
    shared = all(f.get('transport') == 'shared_immutable_v1' for f in metadata)
    producer_save = ([sum(v['save_finished_clock']-v['render_call_finished_clock']
                         for v in f['views']) for f in metadata] if not shared else None)
    gpu_copies = [c['finished_at']-c['started_at'] for r in rows
                  for c in r.get('shared_copy_receipts', [])]
    host_gaps, source_gaps = np.diff(received), np.diff(rendered)
    worst = np.argsort(host_gaps)[-5:][::-1]
    return dict(trial=str(root), status=outcome['status'], reason=outcome.get('reason'),
        student_requests=outcome.get('student_requests', 0), camera_groups=len(rows),
        original_trial_manifest_sha256=sha256(root/'manifest.json'),
        original_result_sha256=sha256(root/'result.json'),
        recorded_bundle_sha256=outcome.get('camera_archive_receipt',
                                           outcome.get('archive'))['bundle_sha256'],
        initial_clock_uncertainty_seconds=clock['uncertainty_seconds'],
        initial_clock_exchanges=len(clock['exchanges']),
        host_received_intervals_seconds=summary(host_gaps),
        producer_render_intervals_seconds=summary(source_gaps),
        image_age_upper_at_delivery_seconds=summary(delivered_age),
        groups_older_than110ms_at_delivery=int((delivered_age > .11).sum()),
        producer_save_call_sum_seconds=summary(producer_save)
            if producer_save is not None else None,
        shared_texture_copies_seconds=summary(gpu_copies) if gpu_copies else None,
        immutable_shared_transport=shared,
        largest_host_gaps=[dict(next_capture_id=rows[i+1]['capture_id'],
            host_gap_seconds=float(host_gaps[i]), source_gap_seconds=float(source_gaps[i]),
            next_delivery_age_upper_seconds=float(delivered_age[i+1])) for i in worst],
        metadata_hashes_verified=True, image_bytes_replayed=False,
        bundle_hash_recomputed=False, physical_exposure_sync_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862992,
                                      entry_script_sha256=sha256(__file__)))
    rows = [inspect(root) for root in args.trial]
    write_json(run/'result.json', dict(status='completed', trials=rows,
        timing_only=True, original_timestamps_preserved=True, actor_arrays_modified=False,
        ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
