"""Decode recorded recurrent intents to requests; no simulator or actuator access."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.actuator_codec import project_pedals
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.replay)
    result = json.loads((source / 'result.json').read_text(encoding='utf8'))
    path = source / 'predictions.npz'
    if (result['status'] != 'completed_stateful_joint_sensor_replay'
            or sha256(path) != result['predictions_sha256']
            or result['actor_deployed'] is not False):
        raise ValueError('Exact completed, non-deployed recurrent replay required')
    with load_array(path, allow_pickle=False) as archive:
        intent, time = archive['output'].copy(), archive['time'].copy()
    if (intent.shape != (len(time), 2) or len(time) != result['original_observations']
            or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)):
        raise ValueError('Original ordered complete two-intent history required')
    requests = np.stack([canonical_controls(row) for row in intent])
    encoded = np.asarray([project_pedals(row)['encoded_request'] for row in requests])
    if np.any((requests[:, 1] > 0) & (requests[:, 2] > 0)):
        raise ValueError('Signed drive cannot request both pedals')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'source_result_sha256': sha256(source / 'result.json'),
        'source_predictions_sha256': result['predictions_sha256'],
        'actor_deployed': False, 'native_applied_controls': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run / 'controls.npz', time=time, intent=intent,
                        canonical_request=requests, encoded_request=encoded)
    out = {'status': 'completed_offline_control_decode_audit', 'observations': len(time),
        'steer_min_max': [float(requests[:, 0].min()), float(requests[:, 0].max())],
        'throttle_positive': int((requests[:, 1] > 0).sum()),
        'brake_positive': int((requests[:, 2] > 0).sum()),
        'encoded_throttle_positive': int((encoded[:, 1] > 0).sum()),
        'encoded_brake_positive': int((encoded[:, 2] > 0).sum()),
        'brake_mean_max': [float(requests[:, 2].mean()), float(requests[:, 2].max())],
        'controls_sha256': sha256(run / 'controls.npz'),
        'raw_request_not_applied_control': True, 'teacher_labels_used': False,
        'actor_deployed': False, 'goal_achieved': False}
    write_json(run / 'result.json', out)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
