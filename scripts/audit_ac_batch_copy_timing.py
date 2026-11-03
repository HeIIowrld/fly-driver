"""Audit every retained batch readback journal, not just actor-consumed frames."""

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.ac_shared_texture import parse_shared_handle
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def distribution(values):
    return dict(count=len(values), median_seconds=float(np.median(values)),
        p95_seconds=float(np.percentile(values, 95)),
        maximum_seconds=float(max(values))) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    bundle = json.loads((root/'vision/bundle.json').read_text(encoding='utf8'))
    assert sha256(root/'vision/camera.zip') == bundle['sha256']
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        camera_bundle_sha256=bundle['sha256'], scope=__doc__, no_fitting=True))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    groups, opens, maps, packing, ack_upper = [], [], [], [], []
    phases = {k: [] for k in ('render_to_path_seen_upper', 'metadata_load_and_validation',
                              'render_to_mmf_seen_upper', 'ack_write', 'persistence',
                              'event_log_flush')}
    failure = None
    with zipfile.ZipFile(root/'vision/camera.zip') as archive:
        clock = json.loads(archive.read('clock-initial.json'))
        if 'batch-copy-journal.json' in archive.namelist():
            recorded = json.loads(archive.read('batch-copy-journal.json'))
            assert recorded['schema'] == 'flybrain.shared_copy.journal.v2'
            assert recorded['serialized_only_after_capture_loop']
            entries = [(r['index'], r) for r in recorded['groups']]
            assert len(entries) <= recorded['maximum_groups']
        else:
            entries = []
            for name in archive.namelist():
                match = re.fullmatch(r'copy-(\d{4})\.json', name)
                if match:
                    entries.append((int(match[1]), json.loads(archive.read(name))))
        for index, journal in entries:
            frame = json.loads(archive.read(f'frame-{index:04d}.json'))
            ack = json.loads(archive.read(f'ack-{index:04d}.json'))
            assert journal['acknowledged'] == ack
            assert frame['index'] == ack['capture_id'] == index
            assert frame['generation'] == ack['generation'] == clock['generation']
            assert frame['request_id'] == ack['request_id']
            views = {v['view_id']: v for v in frame['views']}
            copies = journal['copies']
            assert len(copies) == 4 and {c['view_id'] for c in copies} == set(views)
            for copied in copies:
                assert copied['original_handle'] == parse_shared_handle(
                    views[copied['view_id']]['shared_texture_handle'])
                stage = copied['stage']
                stamps = [copied['started_at'], stage['open_started_at'],
                    stage['open_finished_at'], stage['copy_enqueued_at'],
                    copied['flush_finished_at'], stage['map_started_at'],
                    stage['map_succeeded_at'], stage['unmapped_at'], ack['copied_before_ack']]
                assert np.isfinite(stamps).all() and np.all(np.diff(stamps) >= 0)
                assert copied['source_texture_modified'] is False
                opens.append(stage['open_finished_at']-stage['open_started_at'])
                maps.append(stage['map_succeeded_at']-stage['map_started_at'])
                packing.append(stage['unmapped_at']-stage['map_succeeded_at'])
            elapsed = ack['copied_before_ack']-min(c['started_at'] for c in copies)
            upper = ack['copied_before_ack']-clock['offset_lower_seconds']-frame['script_clock']
            ack_upper.append(upper)
            groups.append(dict(index=index, group_copy_to_ack_seconds=elapsed,
                               render_group_to_ack_upper_seconds=upper))
            p = journal.get('phases')
            if p:
                seen = p.get('metadata_mmf_seen_at', p.get('frame_path_seen_at'))
                phases['render_to_mmf_seen_upper' if 'metadata_mmf_seen_at' in p
                       else 'render_to_path_seen_upper'].append(seen
                    - clock['offset_lower_seconds'] - frame['script_clock'])
                phases['metadata_load_and_validation'].append(
                    p['metadata_validated_at']-seen)
                phases['ack_write'].append(p['ack_write_finished_at']-ack['copied_before_ack'])
                if 'persistence_finished_at' in p:
                    phases['persistence'].append(
                        p['persistence_finished_at']-p['persistence_started_at'])
                if 'event_log_flushed_at' in p:
                    phases['event_log_flush'].append(
                        p['event_log_flushed_at']-p['persistence_finished_at'])
        if 'shared-copy-failure.json' in archive.namelist():
            failure = json.loads(archive.read('shared-copy-failure.json'))
    assert groups, 'No batch timing journals in this source'
    write_json(run/'result.json', dict(status='completed_copy_stage_timing_audit',
        groups=len(groups), group_copy_to_ack=distribution(
            [g['group_copy_to_ack_seconds'] for g in groups]),
        render_group_to_ack_upper=distribution(ack_upper), open_and_pin=distribution(opens),
        map_poll=distribution(maps), host_pack_and_unmap=distribution(packing),
        worker_phases={k: distribution(v) for k, v in phases.items()},
        slowest_groups=sorted(groups, key=lambda g: g['group_copy_to_ack_seconds'])[-8:],
        failure_trace=failure, metadata_clock_bounds_not_physical_exposure_sync=True,
        latency_cause_not_uniquely_identified=True, actor_or_training_qualification=False,
        ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
