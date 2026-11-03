"""Inspect retained failed diagnostic native copies; never admit the recording."""
import argparse
import json
import re
import zipfile
from bisect import bisect_left
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.audit_causal_copy_body_context import first_observed_packets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.recording)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    copies, events, archive = root/'native-copy-history.json', root/'events.jsonl', root/'vision/camera.zip'
    bundle = json.loads((root/'vision/bundle.json').read_text(encoding='utf8'))
    if (result['status'] != 'failed' or result['training_admissible'] is not False
            or sha256(copies) != result['native_copy_history']['sha256']
            or sha256(archive) != bundle['sha256']):
        raise ValueError('Exact failed diagnostic and original hashes required')
    rows = json.loads(copies.read_text(encoding='utf8'))
    event_rows = [json.loads(x) for x in events.read_text(encoding='utf8').splitlines()]
    packets = first_observed_packets(rows)
    receipts = [row['first_end'] for row in packets]
    gaps = np.diff([row['copy']['second_copy_end'] for row in rows])
    distinct, old, missing = 0, 0, 0
    for event in event_rows:
        camera = event['camera']
        chosen = []
        for view in camera['views']:
            index = bisect_left(receipts, view['render_started_lower'])-1
            if index < 0:
                missing += 1
            elif camera['received']-receipts[index] > .05:
                old += 1
            elif 0 <= camera['received']-receipts[index]:
                chosen.append(packets[index]['packet'])
            else:
                raise ValueError('Camera receipt precedes native receipt')
        distinct += len(set(chosen)) > 1
    with zipfile.ZipFile(archive) as z:
        frame_count = sum(bool(re.fullmatch(r'frame-\d{4}\.json', name)) for name in z.namelist())
    run = begin_run(a.output, dict(source_result_sha256=sha256(root/'result.json'),
        events_sha256=sha256(events), native_sha256=sha256(copies),
        archive_sha256=sha256(archive), program_sha256=sha256(__file__),
        scope='failed diagnostic native bytes and metadata count only'))
    out = dict(status='completed_partial_inspection', original_status='failed',
        native_rows=len(rows), unique_packets=len(packets), events=len(event_rows),
        distinct_prior_packet_groups=distinct, missing_prior_views=missing, old_prior_views=old,
        copy_interval_ms_p50_p95_max=(1000*np.percentile(gaps,[50,95,100])).tolist(),
        recorded_vision_frames=result['vision']['frames'], archived_frame_metadata=frame_count,
        camera_raw_audit_passed=False, fit_allowed=False, physical_timing_verified=False)
    write_json(run/'result.json', out)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
