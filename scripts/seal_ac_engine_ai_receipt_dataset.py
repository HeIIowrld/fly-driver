"""Seal explicitly experimental future-applied-control targets; preserve actual clocks and gaps."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_engine_ai_labels import common_expert_settings
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes
from flybrain_drive.training_source_guard import require_training_source


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = read(args.config)
    if type(config.get('require_aid_off', False)) is not bool:
        raise ValueError('Explicit boolean aid admission policy required')
    sources, calibration = [], None
    aid_proofs = {}
    expert_requests = []
    for source in config['sources']:
        root, targets = Path(source['recording']), Path(source['targets'])
        report = read(targets/'result.json')
        provenance = read(targets/'manifest.json')['config']
        raw_audit = Path(source['raw_audit'])
        raw_report = read(raw_audit/'result.json')
        aid_verified = (raw_report.get('camera_aid_state_verified') is True
                        and type(raw_report.get('explicit_aid_off_frames_verified')) is int
                        and raw_report['explicit_aid_off_frames_verified'] > 0)
        if config.get('require_aid_off', False) and not aid_verified:
            raise ValueError('Every source requires an explicit all-frame aid-OFF audit')
        require_training_source(recording=read(root/'result.json'), raw_audit=raw_report, label_audit=report)
        aid_proofs[source['recording']] = aid_verified
        if (report['status'] != 'completed' or raw_report['status'] != 'completed'
                or report['clipped_labels'] != 0 or report['actor_arrays_modified']
                or any(raw_report['actor_replay_max_errors'].values())
                or sha256(targets/'proposal-targets.npz') != report['target_sha256']
                or provenance['source_observation_sha256'] != sha256(root/'observations.npz')
                or provenance['events_sha256'] != sha256(root/'events.jsonl')
                or provenance['control_receipt_sha256'] != sha256(root/'control/api-receipt.json')
                or provenance['raw_audit_manifest_sha256'] != sha256(raw_audit/'manifest.json')):
            raise ValueError('Matching raw replay, exact proposal '
                             'and unchanged actor inputs required')
        recording_manifest = read(root/'manifest.json')
        current = CSPCameraRetina(recording_manifest['config']['sensory_configuration']).calibration
        observation_file = root/'observations.npz'
        if source.get('repack'):
            repack = Path(source['repack'])
            repack_report = read(repack/'result.json')
            repack_config = read(repack/'manifest.json')['config']
            observation_file = repack/'repacked-observations.npz'
            current = CSPCameraRetina(repack_config['repack_configuration']).calibration
            if (repack_report['status'] != 'completed'
                    or any(repack_report['actor_replay_max_errors'].values())
                    or repack_config['observation_sha256'] != sha256(root/'observations.npz')
                    or repack_config['events_sha256'] != sha256(root/'events.jsonl')
                    or repack_report['repacked_sensory_calibration'] != current
                    or repack_report['repacked_observations_sha256'] != sha256(observation_file)):
                raise ValueError('Matching raw audited registered repack required')
        if calibration is not None and current != calibration:
            raise ValueError('Sensory calibrations differ')
        calibration = current
        if read(root/'control/api-receipt.json')['ai_steer_multiplier'] != .9:
            raise ValueError('Verified bounded engine-AI steering configuration required')
        request = read(root/'control/request.json')
        receipt = read(root/'control/api-receipt.json')
        if receipt.get('requested_ai_speed_hint_kmh', request['speed_limit_kmh']) != (
                request['speed_limit_kmh']):
            raise ValueError('Executed expert speed metadata differs from owned request')
        expert_requests.append(request)
        sources.append((source, report, provenance, observation_file))
    expert_settings = common_expert_settings(expert_requests)
    run = begin_run(args.output, dict(configuration=config, scope=__doc__, seed=config['seed'],
        config_file_sha256=sha256(args.config),
        sources=[dict(source, target_manifest_sha256=sha256(
            Path(source['targets'])/'manifest.json'), control_request_sha256=sha256(
                Path(source['recording'])/'control/request.json'))
                 for source, _, _, _ in sources]))
    records, counts = [], np.zeros(3, dtype=np.int64)
    actor_fields = None
    for number, (source, report, provenance, observation_file) in enumerate(sources):
        root, targets = Path(source['recording']), Path(source['targets'])
        with load_array(observation_file) as data:
            actor = {key: data[key] for key in data.files}
        if source.get('repack'):
            with load_array(root/'observations.npz') as original:
                for key in ('obs_view_rgb', 'obs_view_contrast', 'obs_view_valid',
                            'obs_contrast_valid', 'obs_view_age', 'observation_time'):
                    if not np.array_equal(actor[key], original[key]):
                        raise ValueError('Registered organ repack changed '
                                         'original vision or clocks')
        with load_array(targets/'proposal-targets.npz') as data:
            labels = {key: data[key] for key in data.files}
        actor_fields = sorted(key[4:] for key in actor if key.startswith('obs_'))
        for clip, (start, stop) in enumerate(report['observation_segments']):
            action, mask = labels['action'][start:stop], labels['action_valid'][start:stop]
            if not mask.any():
                continue
            if (not actor['obs_view_valid'][start:stop].all()
                    or np.any(np.diff(actor['observation_time'][start:stop]) > .08)):
                raise ValueError('All four views and explicit gap boundaries required')
            path = run/f'episode-{number:02d}-{clip:02d}.npz'
            np.savez_compressed(path, **{key: value[start:stop] for key, value in actor.items()},
                                action=action, action_valid=mask)
            with load_array(path) as saved:
                for key, value in actor.items():
                    if not np.array_equal(saved[key], value[start:stop]):
                        raise ValueError('Actor array changed during sealing')
            counts += mask.sum(0)
            records.append(dict(file=path.name, sha256=sha256(path), steps=stop-start,
                episode_id=f'engine-ai-source{number}-clip{clip}',
                original_recording=source['recording'],
                camera_aid_state_verified=aid_proofs[source['recording']],
                source_actor_observation_sha256=sha256(observation_file),
                source_task_failed_partial_phase=report.get('partial_failed_task_phase', False),
                original_observation_start=start, original_observation_stop=stop,
                boundary='new owned teacher lease or explicit observation gap over80ms',
                target_sha256=report['target_sha256'],
                source_manifest_sha256=provenance['recording_manifest_sha256']))
    contract = dict(kind='engine_ai_future_applied_control_receipt_v1',
        actor_steering_device_scale=.6, **expert_settings,
        target_receipt_horizon_seconds=.04, maximum_target_lateness_seconds=.025,
        clock='host_monotonic_QPC_time.perf_counter', physical_command_time_verified=False,
        interpretation='Executed applied native-control receipt surrogate, '
                       'not internal AI request. '
                       'Inverse stationary codec; moving control lag remains '
                       'an engineering approximation.')
    write_json(run/'dataset.json', dict(schema='flybrain.multiview.teacher.v1',
        action_profile='normalized_ac_absolute', sensory_calibration=calibration,
        actor_fields=actor_fields, decision_dt_seconds=.04, brain_ticks_per_decision=4,
        actual_observation_clocks_preserved=True, episodes=records,
        camera_aid_state_verified=bool(aid_proofs) and all(aid_proofs.values()),
        decisions=sum(record['steps'] for record in records), teacher_kind='engine_ai',
        teacher_label_contract=contract, supervised_action_counts=counts.tolist(),
        expert_labels_available=True, labels_are_executed_requests=False,
        labels_are_executed_applied_controls=True, native_AC_training_data=True,
        training_admissible=True, experimental_offline_replay_only=True,
        internal_ai_commands_observed=False, full_lap_verified=False,
        limitations='Exploratory applied-control surrogate only; no exact internal command timing '
                    'or race-optimal expert claim. '
                    'Do not merge with geometric desired-command task.'))
    episodes, _ = load_episodes(run, calibration)
    write_json(run/'result.json', dict(status='completed', episodes=len(episodes),
        camera_aid_state_verified=bool(aid_proofs) and all(aid_proofs.values()),
        decisions=sum(record['steps'] for record in records), supervised_heads=counts.tolist(),
        actor_arrays_equal_selected_sources=True, original_vision_and_clocks_preserved=True,
        registered_organ_reencoding_from_raw=any(
            source.get('repack') for source, _, _, _ in sources),
        clipped_labels=0, experimental_surrogate_only=True, closed_loop_verified=False,
        dataset_manifest_sha256=sha256(run/'dataset.json'), goal_achieved=False))


if __name__ == '__main__':
    main()
