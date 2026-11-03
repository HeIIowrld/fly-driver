"""Recompute every stored request from wing MN activity and original replay clocks."""

import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_actuator import canonical_controls
from flybrain_drive.graph import sha256
from scripts.build_retinotopic_preview_dataset import read
from scripts.train_visual_body_preview import direct_body_reference


def decode(muscles, reference):
    lateral = np.stack([muscles[..., 1]-muscles[..., 0],
                        muscles[..., 3]-muscles[..., 2]], axis=-1)
    logits = np.asarray(reference['steering_mix'], float)
    mix = np.exp(logits-logits.max())
    mix /= mix.sum()
    turn = np.exp(reference['turn_log_gain'])*(
        (lateral-np.asarray(reference['lateral_reference']))@mix)
    common = .5*(muscles[..., 4]+muscles[..., 5])
    forward = np.exp(reference['forward_log_gain'])*(common-reference['power_reference'])
    return np.tanh(np.stack([turn, forward], axis=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.probe)
    result, manifest = read(root/'result.json'), read(root/'manifest.json')
    if result['status'] != 'completed_physical_maneuver_component_probe':
        raise ValueError('Complete original component probe required')
    plan = manifest['config']['plan']
    for path, digest in plan['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Original probe input changed')
    old, new = result['old_motor_reference'], result['rebased_motor_reference']
    for field in ('turn_log_gain', 'forward_log_gain', 'steering_mix'):
        np.testing.assert_array_equal(old[field], new[field])
    entries = read(Path(plan['registered_data'])/'dataset.json')['episodes']
    entry = next(e for e in entries if e['role'] == 'development')
    if sha256(entry['targets_file']) != entry['targets_sha256']:
        raise ValueError('Original clock changed')
    with load_array(entry['targets_file'], allow_pickle=False) as a:
        clock = a['observation_time'].copy()
    intervals = np.diff(clock)
    if len(clock) != entry['observations'] or not np.all((intervals > 0) & (intervals <= .08)):
        raise ValueError('Whole clock and original gap guard required')
    body, valid, _ = direct_body_reference(entry, clock,
        sha256('configs/ac-virtual-navigation-body-v2.json'))
    with load_array(plan['prediction_file'], allow_pickle=False) as a:
        predicted = a['adapted'].copy()
    commands = np.column_stack([-body[:, 16]*predicted[:, 5],
                                np.full(len(clock), 60/3.6)]).astype(np.float32)
    errors, hashes, replay = [], {}, {}
    for variant in ('intact', 'desired_yaw', 'speed_error', 'yaw_feedback'):
        for kind in ('injection', 'whole-replay'):
            path = root/(kind+'-'+variant+'.npz')
            hashes[path.name] = sha256(path)
            with load_array(path, allow_pickle=False) as a:
                intent, muscles = a['intent'].copy(), a['muscles'].copy()
                if (intent.shape[-1] != 2 or muscles.shape[:-1] != intent.shape[:-1]
                        or muscles.shape[-1] != 6 or not np.isfinite(intent).all()
                        or not np.isfinite(muscles).all() or np.any(abs(intent) > 1)
                        or np.any((muscles < 0) | (muscles > 1))):
                    raise ValueError('Bounded actual wing MN activity and intentions required')
                error = float(abs(decode(muscles.astype(float), new)-intent).max())
                errors.append(error)
                if error > 2e-5:
                    raise ValueError('Intent has an unexplained path outside motor readout')
                if kind == 'whole-replay':
                    np.testing.assert_array_equal(a['observation_time'], clock)
                    np.testing.assert_array_equal(a['body_valid'], valid)
                    np.testing.assert_array_equal(a['physical_commands'], commands)
                    controls = np.asarray([canonical_controls(x) for x in intent])
                    np.testing.assert_array_equal(a['canonical_requests'], controls)
                    replay[variant] = dict(intent=intent, controls=controls)
    intact = replay['intact']
    differences = {variant: dict(
        max_absolute_intent_change=abs(intact['intent']-value['intent']).max(0).tolist(),
        mean_absolute_intent_change=abs(intact['intent']-value['intent']).mean(0).tolist())
        for variant, value in replay.items() if variant != 'intact'}
    run = begin_run(args.output, dict(arguments=vars(args), no_fitting=True,
        program_sha256=sha256(__file__), probe_result_sha256=sha256(root/'result.json'),
        original_payload_sha256=hashes, original_clock_sha256=entry['targets_sha256']))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'result.json', dict(status='completed_physical_maneuver_component_audit',
        original_observations=len(clock), original_intervals=len(clock)-1,
        original_elapsed_seconds=float(clock[-1]-clock[0]), all_clocks_and_masks_exact=True,
        maximum_motor_readout_error=max(errors), requests_exactly_motor_codec=True,
        whole_replay_path_intervention_differences=differences,
        component_probe_passed=result['all_component_checks_passed'],
        optimizer_updates=0, actuator_messages_sent=0, actor_deployed=False,
        visual_anticipation_qualified=False, physical_vehicle_plant_tested=False,
        physiological_function_verified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
