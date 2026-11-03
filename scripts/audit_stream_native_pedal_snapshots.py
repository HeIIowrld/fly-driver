"""Compare saved requests with same-decision native snapshots, not teacher labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'controls', 'recording', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    replay, controls, recording = map(Path, (a.replay, a.controls, a.recording))
    rr, cr = read(replay / 'result.json'), read(controls / 'result.json')
    rm = read(replay / 'manifest.json')['config']
    cm = read(controls / 'manifest.json')['config']
    events_path = recording / 'events.jsonl'
    control_path = controls / 'controls.npz'
    if (rr['status'] != 'completed_stateful_joint_sensor_replay'
            or cr['status'] != 'completed_offline_control_decode_audit'
            or sha256(events_path) != rm['recording_events_sha256']
            or sha256(control_path) != cr['controls_sha256']
            or cm['source_predictions_sha256'] != rr['predictions_sha256']
            or sha256(replay / 'predictions.npz') != rr['predictions_sha256']
            or rr['actor_deployed'] is not False):
        raise ValueError('Exact recorded event, replay and control lineage required')
    events = [json.loads(line) for line in events_path.read_text(encoding='utf8').splitlines()]
    with load_array(control_path, allow_pickle=False) as x:
        times, request = x['time'].copy(), x['canonical_request'].copy()
    with load_array(replay / 'predictions.npz', allow_pickle=False) as x:
        np.testing.assert_array_equal(times, x['time'])
        np.testing.assert_array_equal(request[:, 0], -x['output'][:, 0])
        np.testing.assert_array_equal(request[:, 1], np.maximum(x['output'][:, 1], 0))
        np.testing.assert_array_equal(request[:, 2], np.maximum(-x['output'][:, 1], 0))
    np.testing.assert_array_equal(times, [row['decision_monotonic_seconds'] for row in events])
    receipts = np.array([row['native']['received_monotonic_seconds'] for row in events])
    native = np.array([[row['native']['fields'][name] for name in ('gas', 'brake', 'speedKmh')]
                       for row in events])
    if (native.shape != (len(times), 3) or request.shape != (len(times), 3)
            or not np.isfinite(native).all() or not np.isfinite(receipts).all()
            or np.any((native[:, :2] < 0) | (native[:, :2] > 1))
            or np.any(native[:, 2] < 0)):
        raise ValueError('Finite observed pedal fractions and speed required')
    age = times - receipts
    valid = (age >= 0) & (age <= .05) & (native[:, 2] > 3.6)
    partitions = {'moving_all': valid, 'moving_at_most_30_kmh': valid & (native[:, 2] <= 30),
                  'moving_above_30_kmh': valid & (native[:, 2] > 30)}
    summary = {}
    for name, mask in partitions.items():
        if not mask.any():
            raise ValueError('Nonempty explicit speed strata required')
        summary[name] = {
            'observations': int(mask.sum()),
            'requested_brake_over_002': int((mask & (request[:, 2] > .02)).sum()),
            'native_gas_over_002': int((mask & (native[:, 0] > .02)).sum()),
            'both_observed': int((mask & (request[:, 2] > .02) & (native[:, 0] > .02)).sum()),
            'gas_brake_mean_absolute_difference': np.mean(
                np.abs(request[mask, 1:] - native[mask, :2]), axis=0).tolist(),
        }
    run = begin_run(a.output, {'program_sha256': sha256(__file__),
        'source_events_sha256': sha256(events_path), 'controls_sha256': cr['controls_sha256'],
        'replay_predictions_sha256': rr['predictions_sha256'], 'maximum_receipt_age_seconds': .05,
        'model_goal_kmh': 30, 'recorded_teacher_goal_kmh': 60,
        'teacher_snapshot_is_not_synchronized_label': True, 'actor_deployed': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(run / 'comparison.npz', time=times, requested=request,
                        native=native, native_receipt_age=age, moving_mask=valid)
    result = {'status': 'completed_native_pedal_snapshot_comparison', 'groups': summary,
        'comparison_sha256': sha256(run / 'comparison.npz'),
        'same_decision_host_receipts_not_physics_alignment': True,
        'matched_ai_lease_or_future_labels_verified': False,
        'recorded_teacher_goal_differs_from_model_goal': True,
        'observations_are_not_independent': True, 'actor_deployed': False,
        'optimizer_updates': 0, 'goal_achieved': False}
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
