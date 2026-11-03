"""Audit original shared-camera copy/ack timing without changing timestamps or images."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.ac_shared_texture import verify_shared_frame
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite timing required')
    return dict(count=len(values), median=float(np.median(values)),
                p99=float(np.quantile(values, .99)), maximum=float(values.max()))


def inspect(root):
    root = Path(root)
    timings = []
    with zipfile.ZipFile(root/'vision/camera.zip') as archive:
        clock = json.loads(archive.read('clock-initial.json'))
        rows = [json.loads(line) for line in archive.read('stream.jsonl').decode().splitlines()]
        previous = None
        for index, row in enumerate(rows):
            raw = archive.read(Path(row['file']).name)
            if hashlib.sha256(raw).hexdigest() != row['file_sha256']:
                raise ValueError('Metadata hash differs')
            frame = json.loads(raw)
            ack = json.loads(archive.read(f'ack-{index:04d}.json'))
            if row['capture_id'] != index or frame['generation'] != clock['generation']:
                raise ValueError('Original sequential frame identity differs')
            verify_shared_frame(frame, row, ack)
            copies = row['shared_copy_receipts']
            if any(b['started_at'] < a['finished_at']
                   for a, b in zip(copies, copies[1:], strict=False)):
                raise ValueError('Sequential copies overlap')
            render = min(v['render_started_clock'] for v in frame['views'])
            gpu = sum(c['finished_at']-c['started_at'] for c in copies)
            gaps = [b['started_at']-a['finished_at']
                    for a, b in zip(copies, copies[1:], strict=False)]
            timing = dict(capture_id=index, source_render_clock=render,
                          received=row['received'], ack=ack['copied_before_ack'],
                          copy_call_sum_seconds=gpu,
                          between_copy_uninstrumented_seconds=sum(gaps),
                          largest_between_copy_gap_seconds=max(gaps),
                          last_copy_to_ack_seconds=ack['copied_before_ack']-copies[-1]['finished_at'],
                          source_to_ack_upper_seconds=ack['copied_before_ack']
                          -render-clock['offset_lower_seconds'],
                          source_to_delivery_upper_seconds=row['received']
                          -render-clock['offset_lower_seconds'])
            if previous:
                timing['source_render_interval_seconds'] = render-previous['source_render_clock']
                timing['host_delivery_interval_seconds'] = row['received']-previous['received']
                timing['previous_ack_to_next_render_lower_seconds'] = (
                    render+clock['offset_lower_seconds']-previous['ack'])
                timing['previous_ack_to_next_render_upper_seconds'] = (
                    render+clock['offset_upper_seconds']-previous['ack'])
            timings.append(timing)
            previous = timing
    keys = [k for k in timings[-1] if k.endswith('_seconds')]
    return dict(trial=str(root), bundle_sha256=sha256(root/'vision/camera.zip'),
                frames=len(rows), clock_uncertainty_seconds=clock['uncertainty_seconds'],
                metadata_hashes_and_copy_ack_contract_verified=True,
                statistics={k: stats([r[k] for r in timings if k in r]) for k in keys},
                worst_source_intervals=sorted(timings[1:], key=lambda r:
                    r['source_render_interval_seconds'], reverse=True)[:12],
                final_frames=timings[-15:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args),
                                      script_sha256=sha256(__file__)))
    rows = [inspect(p) for p in args.trial]
    write_json(run/'result.json', dict(status='completed', trials=rows,
        original_timestamps_preserved=True, pixels_modified=False, ui_actions_sent=0,
        actuator_messages_sent=0, optimizer_updates=0, goal_achieved=False,
        uninstrumented_gaps_not_attributed_to_specific_operation=True))
    print(json.dumps([dict(trial=r['trial'], frames=r['frames'], statistics=r['statistics'])
                      for r in rows]), flush=True)


if __name__ == '__main__':
    main()
