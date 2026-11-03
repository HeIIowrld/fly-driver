"""Bound body receipt order relative to four camera renders; no physical-time claim."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.causal_render_body_match import select_prior_body_receipt
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recording', 'raw-audit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, audit = Path(args.recording), Path(args.raw_audit)
    report, config = read(audit/'result.json'), read(audit/'manifest.json')['config']
    events_path = root/'events.jsonl'
    samples_path = root/'control/native-motion-samples.json'
    if (report['status'] != 'completed' or not report['camera_aid_state_verified']
            or any(report['actor_replay_max_errors'].values())
            or sha256(events_path) != config['events_sha256']):
        raise ValueError('Exact original raw replay audit required')
    events = [json.loads(line) for line in events_path.read_text(encoding='utf8').splitlines()]
    samples, seen = [], set()
    for row in read(samples_path):
        raw = base64.b64decode(row['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != row['prefix_sha256']:
            raise ValueError('Native raw prefix hash differs')
        sample = decode_physics_prefix(raw, row['received_monotonic_seconds'])
        if sample.packet_id != row['packet_id']:
            raise ValueError('Native packet differs')
        if sample.packet_id not in seen:
            samples.append(sample)
            seen.add(sample.packet_id)
    times = np.array([sample.received_monotonic_seconds for sample in samples])
    rows = []
    for event in events:
        camera = event['camera']
        lower = np.array([view['render_started_lower'] for view in camera['views']])
        upper = np.full(4, camera['received'])
        match = select_prior_body_receipt(times, lower, upper)
        sample = samples[match.native_index] if match.native_index is not None else None
        rows.append({'index': event['index'], 'camera_capture_id': camera['capture_id'],
                     'reason': match.reason, 'maximum_host_age_seconds': (
                         match.maximum_host_age_seconds),
                     'prior_packet': sample.packet_id if sample is not None else None,
                     'prior_receipt': sample.received_monotonic_seconds if sample else None,
                     'actor_packet': event['native']['packet_id'],
                     'render_start_lower_min': float(lower.min()),
                     'camera_received_upper': float(upper.max())})
    counts = {reason: sum(row['reason'] == reason for row in rows)
              for reason in sorted({row['reason'] for row in rows})}
    matched = [row for row in rows if row['prior_packet'] is not None]
    by_packet = {sample.packet_id: sample for sample in samples}
    differences = {}
    for field in ('speedKmh', 'brake'):
        values = [abs(by_packet[row['prior_packet']].fields[field]
                      - event['native']['fields'][field])
                  for row, event in zip(rows, events, strict=True)
                  if row['prior_packet'] is not None]
        differences[field] = {'median_p95_max': (
            np.percentile(values, [50, 95, 100]).tolist() if values else None),
                              'over_0_05_count': sum(value > .05 for value in values)}
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'recording': str(root), 'events_sha256': sha256(events_path),
        'native_samples_sha256': sha256(samples_path),
        'raw_audit_sha256': sha256(audit/'result.json'), 'scope': __doc__})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'receipt-pairs.json', rows)
    result = {'status': 'completed_causal_receipt_bound_audit', 'observations': len(rows),
              'reason_counts': counts,
              'matched_actor_same_packet': sum(row['prior_packet'] == row['actor_packet']
                                               for row in matched),
              'host_age_median_seconds': float(np.median([
                  row['maximum_host_age_seconds'] for row in matched])) if matched else None,
              'prior_vs_original_actor_absolute_difference': differences,
              'pair_sha256': sha256(run/'receipt-pairs.json'),
              'upper_bound': 'camera group host receipt, not exact exposure',
              'physics_source_time_verified': False, 'camera_exposure_time_verified': False,
              'training_admissible': False, 'actor_deployed': False, 'optimizer_updates': 0}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
