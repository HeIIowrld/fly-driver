"""Measure immutable receipt/publication and subsequent verified disk persistence separately."""

import argparse
import json
from pathlib import Path

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_frame_writer import verify_persistence_archive
from scripts.audit_ac_batch_copy_timing import distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.recording)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, no_fitting=True,
        program_sha256=sha256(__file__),
        recording_manifest_sha256=sha256(source/'manifest.json'),
        camera_descriptor_sha256=sha256(source/'vision/bundle.json')))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    with CameraArchive(source/'vision') as camera:
        mmf = json.loads((source/'manifest.json').read_text('utf8'))['config']['arguments'].get(
            'mmf_metadata', False)
        rows = verify_persistence_archive(camera, expected_mmf=mmf)
        journal = camera.json('frame-persistence-journal.json')
        groups = camera.json('batch-copy-journal.json')['groups']
        if len(groups) != len(rows):
            raise ValueError('Copy/persistence group count differs')
        measurements = {key: [] for key in ('memory_preparation', 'queue_delay',
            'disk_write_and_readback', 'durable_after_receipt', 'publication_after_receipt')}
        early = 0
        for group, persisted in zip(groups, journal['records'], strict=True):
            index = persisted['group_id']
            row, phases = rows[index], group['phases']
            if (group['index'] != index or group['status'] != 'published'
                    or phases['immutable_memory_ready_at'] != row['received']
                    or not row['received'] <= phases['persistence_enqueued_at']
                    <= phases['actor_published_at']):
                raise ValueError('Actual memory/publication timing differs')
            measurements['memory_preparation'].append(
                row['received']-phases['persistence_started_at'])
            measurements['queue_delay'].append(persisted['started_at']-persisted['submitted_at'])
            measurements['disk_write_and_readback'].append(
                persisted['finished_at']-persisted['started_at'])
            measurements['durable_after_receipt'].append(persisted['finished_at']-row['received'])
            measurements['publication_after_receipt'].append(
                phases['actor_published_at']-row['received'])
            early += phases['actor_published_at'] < persisted['finished_at']
    events = [json.loads(line) for line in (source/'events.jsonl').read_text('utf8').splitlines()]
    write_json(run/'result.json', dict(status='completed_deferred_persistence_timing',
        groups=len(rows), maximum_pending_observed=journal['maximum_queued'],
        groups_published_before_disk_finished=early,
        distributions={key: distribution(values) for key, values in measurements.items()},
        measured_native_steer_angle_raw_min=min(
            e['native']['fields']['steerAngle'] for e in events),
        measured_native_steer_angle_raw_max=max(
            e['native']['fields']['steerAngle'] for e in events),
        steering_unit_or_action_codec_not_qualified_by_this_audit=True,
        teacher_calibration_only=True, student_deployed=False, training_admissible=False,
        goal_achieved=False))


if __name__ == '__main__':
    main()
