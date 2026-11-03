"""Audit decision-time validity of prior camera-bound body observations."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body-pairs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.body_pairs)
    report = json.loads((root/'result.json').read_text(encoding='utf8'))
    path = root/'body-pairs.npz'
    if (report['status'] != 'completed_prior_body_counterfactual'
            or report['training_admissible'] is not False
            or sha256(path) != report['body_pairs_sha256']):
        raise ValueError('Exact diagnostic body pairs required')
    with np.load(path, allow_pickle=False) as data:
        valid = data['valid'].copy()
        times = data['decision_monotonic_seconds'].copy()
    if (valid.ndim != 3 or valid.shape[1:] != (2, 21) or valid.dtype != bool
            or len(times) != len(valid) or np.any(np.diff(times) <= 0)):
        raise ValueError('Original paired validity and advancing decision clocks required')
    runnable = valid[:, 1, 4] & valid[:, 1, 16]
    changes = np.diff(np.r_[False, runnable, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    segments = [{'start': int(start), 'stop_exclusive': int(end), 'length': int(end-start)}
                for start, end in zip(starts, ends, strict=True)]
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'body_pairs_sha256': sha256(path), 'source_result_sha256': sha256(root/'result.json'),
        'required_channels': ['yaw_rate', 'speed'], 'scope': __doc__})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'valid-segments.json', segments)
    result = {'status': 'completed_decision_time_body_validity_audit',
              'observations': len(valid), 'original_yaw_speed_valid': int(
                  (valid[:, 0, 4] & valid[:, 0, 16]).sum()),
              'prior_yaw_speed_valid': int(runnable.sum()),
              'prior_yaw_speed_invalid': int((~runnable).sum()),
              'valid_segments': len(segments),
              'longest_valid_segment': max((x['length'] for x in segments), default=0),
              'full_continuous_prior_replay_admissible': bool(runnable.all()),
              'original_validity_masks_preserved': True, 'filled_missing_data': False,
              'segments_sha256': sha256(run/'valid-segments.json'),
              'training_admissible': False, 'actor_deployed': False, 'optimizer_updates': 0}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
