"""Query one fixed geometric teacher on actual recorded states; labels are NOT executed."""

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_geometric_teacher import GeometricRecoveryTeacher
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX
from flybrain_drive.ac_native_relabel import relabel_episode
from flybrain_drive.ac_recording_provenance import recording_for_dataset
from flybrain_drive.ac_sac_teacher import legacy_request_to_normalized_controls
from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry
from flybrain_drive.episode_storage import load_array, resolve_episode_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--kinematic-calibration', required=True)
    parser.add_argument('--maximum-speed-kmh', type=float, default=20.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.input)
    manifest = json.loads((root / 'dataset.json').read_text(encoding='utf8'))
    if len(manifest['episodes']) != 1 or manifest['action_profile'] != 'normalized_ac_absolute':
        raise ValueError('One original native-action recording episode required')
    load_episodes(root, manifest['sensory_calibration'])
    recording = recording_for_dataset(root)
    events_file = recording / manifest.get('raw_events_file', 'observations-and-requests.jsonl')
    if sha256(events_file) != manifest['raw_events_sha256']:
        raise ValueError('Original recording changed')
    recording_manifest = json.loads((recording / 'manifest.json').read_text(encoding='utf8'))
    geometry_config = recording_manifest['config'].get('geometry_configuration')
    if geometry_config is None:
        path = Path('configs/ac-teacher-geometry-result.json')
        if sha256(path) != recording_manifest['source_sha256'][path.as_posix()]:
            raise ValueError('Recorded diagnostic geometry configuration changed')
        geometry_config = json.loads(path.read_text(encoding='utf8'))['geometry_configuration']
    geometry = PrivilegedTrackGeometry(
        geometry_config['track_csv'], geometry_config['reference_csv'],
        track_sha256=geometry_config['track_sha256'],
        reference_sha256=geometry_config['reference_sha256'])
    fit = json.loads(Path(args.kinematic_calibration).read_text(encoding='utf8'))
    if fit['status'] != 'completed' or not fit['teacher_only']:
        raise ValueError('Completed teacher-only measured steering fit required')
    recorded_profile = recording_manifest['config']['axis_configuration']['profile_sha256']
    if recorded_profile != fit['profile_sha256']:
        raise ValueError('Teacher steering fit differs from the recording control profile')
    teacher = GeometricRecoveryTeacher(geometry.line, fit['curvature_per_canonical_steering'],
                                       maximum_speed_kmh=args.maximum_speed_kmh)
    contract = dict(kind='geometric_recovery_fixed_task',
        maximum_speed_kmh=args.maximum_speed_kmh,
        reference_sha256=geometry_config['reference_sha256'],
        track_sha256=geometry_config['track_sha256'],
        kinematic_calibration_sha256=sha256(args.kinematic_calibration),
        teacher_implementation_sha256=sha256('src/flybrain_drive/ac_geometric_teacher.py'),
        interpretation='teacher-only geometric desired canonical control at recorded state; '
                       'execution status recorded separately')
    student = manifest.get('source_request_kind') == 'student'
    pairs, native_to_pair, requests = {}, {}, []
    for row in (json.loads(s) for s in events_file.read_text(encoding='utf8').splitlines()):
        if row['kind'] == 'raw_pair':
            for source, tag in [('sdk', 'wire'), ('native', 'prefix')]:
                raw = base64.b64decode(row[source][tag+'_base64'], validate=True)
                if hashlib.sha256(raw).hexdigest() != row[source][tag+'_sha256']:
                    raise ValueError('Original native/SDK bytes changed')
            key = (row['sdk']['sequence'], row['native']['packet_id'])
            if key in pairs:
                raise ValueError('Duplicate original sensor identity')
            pairs[key] = row
            native_to_pair[key[1]] = key
        elif row['kind'] == ('student_request' if student else 'teacher_request'):
            if student:
                row = copy.deepcopy(row)
                key = native_to_pair[row['actor_input']['native_packet_id']]
                row.update(sdk_sequence=key[0], native_packet_id=key[1])
            requests.append(row)
    qualities = json.loads((root.parent / 'sample-quality.json').read_text(encoding='utf8'))
    episode = manifest['episodes'][0]
    file = resolve_episode_path(root, episode, manifest)
    if sha256(file) != episode['sha256']:
        raise ValueError('Sealed episode path/hash differs')
    with load_array(file, allow_pickle=False) as data:
        original = {key: data[key] for key in data.files}
    if not len(requests) == len(qualities) == len(original['action']):
        raise ValueError('One raw request/identity per measured actor row required')
    labels, audit = [], []
    for index, (request, quality) in enumerate(zip(requests, qualities, strict=True)):
        key = (quality['source_sdk_sequence'], quality['source_native_packet_id'])
        if key != (request['sdk_sequence'], request['native_packet_id']):
            raise ValueError('Actor row order differs from raw recording')
        clock = request['actor_input']['decision_monotonic_seconds'] if student else (
            request['image_candidate']['actor_decision_monotonic_seconds']
            if request.get('image_candidate', {}).get('source_kind') == 'csp_four_camera'
            else request['request']['requested_monotonic_seconds'])
        if original['observation_time'][index] != clock:
            raise ValueError('Measured actor decision clock changed')
        pair = pairs[key]
        sdk = json.loads(base64.b64decode(pair['sdk']['wire_base64']))
        truth = sdk['privileged_teacher']
        if (truth['actor_input'] is not False
                or truth.get('protocol') != 'flybrain.ac.privileged_teacher.v1'
                or sdk['producer_session_id'] != fit['producer_session_id']):
            raise ValueError('Explicit teacher-only truth and reviewed engine session required')
        state = {k: truth['raw'][k] for k in ('world_position_x', 'world_position_y')}
        state['speed'] = sdk['speed']
        values = PHYSICS_PREFIX.unpack(base64.b64decode(pair['native']['prefix_base64']))
        offset = 0
        for name, _, count in PHYSICS_LAYOUT:
            if name == 'heading':
                state['yaw'] = -values[offset]
            offset += count
        state.update(geometry.features(state['world_position_x'], state['world_position_y'],
                                       state['yaw']))
        value = legacy_request_to_normalized_controls(teacher.request(state))
        labels.append(value)
        audit.append(dict(index=index, executed_source_request=original['action'][index].tolist(),
            counterfactual_teacher_request=value.tolist(), source_sdk_sequence=key[0],
            source_native_packet_id=key[1], teacher_diagnostics=teacher.last_diagnostics,
            counterfactual_request_executed=False))
    arrays = relabel_episode(original, labels)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=861801,
        source_manifest_sha256=sha256(root / 'dataset.json'), label_contract=contract,
        actual_game_actions_sent=0))
    dataset = run / 'dataset'
    dataset.mkdir()
    target = dataset / 'native-segment.npz'
    np.savez_compressed(target, **arrays)
    with load_array(target, allow_pickle=False) as saved:
        for key in original:
            if key != 'action' and not np.array_equal(saved[key], original[key]):
                raise ValueError('Measured actor arrays or masks/clocks changed')
    updated = copy.deepcopy(manifest)
    updated['episode_storage'] = 'copy'
    updated.update(source='actual AC actor arrays with unexecuted fixed-task geometric labels',
        action_provider='offline geometric teacher query', teacher_label_contract=contract,
        source_recording_action_provider=manifest.get('action_provider', 'public SAC'),
        source_recording_collection_speed_targets_kmh=manifest.get(
            'collection_speed_targets_kmh', []),
        teacher_source_kinds=['geometric_recovery_counterfactual'],
        collection_speed_targets_kmh=[], governor_changed_requests=0,
        native_AC_training_data=True,
        expert_labels_available=True,
        action_semantics='counterfactual canonical desired request; not applied-control feedback',
        labels_are_executed_requests=False, source_actor_arrays_unchanged=True,
        sources=[dict(directory=str(root), sha256=sha256(root / 'dataset.json'))],
        episodes=[dict(episode_id='native-segment', file=target.name,
                       sha256=sha256(target), steps=len(labels))])
    write_json(dataset / 'dataset.json', updated)
    write_json(run / 'sample-quality.json', qualities)
    write_json(run / 'counterfactual-labels.json', audit)
    load_episodes(dataset, manifest['sensory_calibration'])
    labels = np.array(labels)
    write_json(run / 'result.json', dict(status='completed', decisions=len(labels),
        actor_arrays_masks_clocks_unchanged=True, labels_executed=False, game_actions_sent=0,
        brake_over_005=int((labels[:, 2] > .05).sum()),
        brake_over_03=int((labels[:, 2] > .3).sum()),
        source_request_label_rmse=np.sqrt(np.mean((labels-original['action'])**2, axis=0)).tolist(),
        actual_braking_response_validated=False, actor_truth_inputs_added=False,
        full_lap_verified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
