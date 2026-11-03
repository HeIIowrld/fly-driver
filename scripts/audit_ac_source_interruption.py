"""Locate original source-frame gaps and concurrent persistence intervals without fitting."""

import argparse
import json
from pathlib import Path

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_frame_writer import verify_persistence_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.recording)
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        recording_manifest_sha256=sha256(source/'manifest.json'), scope=__doc__, fitting=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    with CameraArchive(source/'vision') as camera:
        mmf = json.loads((source/'manifest.json').read_text('utf8'))['config']['arguments'].get(
            'mmf_metadata', False)
        rows = verify_persistence_archive(camera, expected_mmf=mmf)
        groups = camera.json('batch-copy-journal.json')['groups']
        persisted = camera.json('frame-persistence-journal.json')['records']
        initial = camera.json('clock-initial.json')
        offset = initial['offset_lower_seconds']
        frames = [camera.json(f'frame-{i:04d}.json') for i in range(len(rows))]
        gaps = []
        for i in range(1, len(frames)):
            previous, current = frames[i-1], frames[i]
            previous_render = previous['script_clock']+offset
            current_render = current['script_clock']+offset
            if current_render <= previous_render:
                raise ValueError('Original source frame clocks are not increasing')
            disk = [dict(group_id=r['group_id'], started_at=r['started_at'],
                         finished_at=r['finished_at'],
                         write_seconds=r['finished_at']-r['started_at']) for r in persisted
                    if r['started_at'] < current_render and r['finished_at'] > previous_render]
            gaps.append(dict(index=i, source_gap_seconds=current_render-previous_render,
                render_frame_advance=int(current['render_frame'].removesuffix('ULL'))
                    - int(previous['render_frame'].removesuffix('ULL')),
                previous_capture_published_at=groups[i-1]['phases']['actor_published_at'],
                next_metadata_seen_at=groups[i]['phases'][
                    'metadata_mmf_seen_at' if mmf else 'frame_path_seen_at'],
                next_render_to_seen_upper_seconds=(
                    groups[i]['phases'][
                        'metadata_mmf_seen_at' if mmf else 'frame_path_seen_at']-current_render),
                overlapping_disk_intervals=disk))
        gaps.sort(key=lambda r: r['source_gap_seconds'], reverse=True)
    write_json(run/'result.json', dict(status='completed_source_interruption_audit',
        groups=len(rows), worst_source_gaps=gaps[:5],
        source_gaps_above_150ms=sum(g['source_gap_seconds'] > .15 for g in gaps),
        source_clock_offset_width_seconds=initial['uncertainty_seconds'],
        simultaneous_delays_do_not_identify_unique_cause=True,
        physical_exposure_synchronization_verified=False,
        all_original_clocks_and_failures_retained=True, student_deployed=False,
        training_admissible=False, goal_achieved=False))


if __name__ == '__main__':
    main()
