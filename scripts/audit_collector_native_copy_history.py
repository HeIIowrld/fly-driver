"""Audit diagnostic native copy history against the original camera event clocks."""
import argparse
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.audit_causal_copy_body_context import first_observed_packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--raw-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    audit = json.loads((Path(args.raw_audit)/'result.json').read_text(encoding='utf8'))
    audit_manifest = json.loads((Path(args.raw_audit)/'manifest.json').read_text(encoding='utf8'))
    history_path, event_path = root/'native-copy-history.json', root/'events.jsonl'
    if (result['status'] != 'completed' or result['training_admissible'] is not False
            or audit['status'] != 'completed'
            or audit_manifest['config']['recording_manifest_sha256'] != sha256(root/'manifest.json')
            or audit_manifest['config']['events_sha256'] != sha256(event_path)
            or sha256(history_path) !=
            result['native_copy_history']['sha256']):
        raise ValueError('Completed diagnostic and original raw audit required')
    rows = json.loads(history_path.read_text(encoding='utf8'))
    events = [json.loads(line) for line in event_path.read_text(encoding='utf8').splitlines()]
    if len(rows) != result['native_copy_history']['rows'] or len(events) != result['decisions']:
        raise ValueError('Original counts differ')
    packets = first_observed_packets(rows)
    receipts = [p['first_end'] for p in packets]
    gaps = np.diff([row['copy']['second_copy_end'] for row in rows])
    distinct, missing, old, valid_ages = 0, 0, 0, []
    for event in events:
        camera = event['camera']
        selected = []
        for view in camera['views']:
            index = bisect_left(receipts, view['render_started_lower'])-1
            if index < 0:
                missing += 1
                continue
            age = camera['received']-receipts[index]
            if not 0 <= age <= .05:
                old += 1
                continue
            selected.append(packets[index]['packet'])
            valid_ages.append(age)
        distinct += len(set(selected)) > 1
    output = begin_run(args.output, dict(recording_result_sha256=sha256(root/'result.json'),
        raw_audit_sha256=sha256(Path(args.raw_audit)/'result.json'),
        native_sha256=sha256(history_path), events_sha256=sha256(event_path),
        program_sha256=sha256(__file__), scope='host receipt bounds; diagnostic only'))
    summary = dict(status='completed', original_recording_status=result['status'],
        rows=len(rows), unique_packets=len(packets),
        events=len(events), groups_distinct_packets=distinct, missing_prior_views=missing,
        old_prior_views=old, copy_interval_ms_p50_p95_max=(np.percentile(gaps, [50,95,100])*1000).tolist(),
        valid_prior_age_ms_p50_p95_max=(np.percentile(valid_ages, [50,95,100])*1000).tolist(),
        fit_allowed=False, physical_sample_or_exposure_verified=False)
    write_json(output/'result.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
