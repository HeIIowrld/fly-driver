"""Rebuild native organs from original prefixes; add only the declared yaw pair to sealed data."""

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_recording_provenance import recording_for_dataset
from flybrain_drive.episode_storage import load_array, resolve_episode_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes


def load(file):
    return json.loads(Path(file).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', default='configs/ac-csp-yaw-retina.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.dataset)
    manifest = load(root / 'dataset.json')
    config = load(args.config)
    retina = CSPCameraRetina(config)
    if retina.yaw_spec is None or len(manifest['episodes']) != 1:
        raise ValueError('One sealed episode and explicit native yaw registration required')
    spec = retina.yaw_spec
    if sha256(spec['review']) != spec['review_sha256']:
        raise ValueError('Native yaw consistency review changed')
    base = copy.deepcopy(config)
    del base['native_retina']['yaw_rotation_organs']
    load_episodes(root, CSPCameraRetina(base).calibration)
    recording = recording_for_dataset(root)
    student = manifest.get('source_request_kind') == 'student'
    event_file = recording / manifest.get('raw_events_file',
        'events.jsonl' if student else 'observations-and-requests.jsonl')
    if sha256(event_file) != manifest['raw_events_sha256']:
        raise ValueError('Original raw sensor recording changed')
    pairs, requests = {}, []
    for row in (json.loads(line) for line in event_file.read_text(encoding='utf8').splitlines()):
        if row['kind'] == 'raw_pair':
            raw = base64.b64decode(row['native']['prefix_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != row['native']['prefix_sha256']:
                raise ValueError('Original native prefix changed')
            sample = decode_physics_prefix(raw, row['native']['received_monotonic_seconds'])
            if sample.packet_id != row['native']['packet_id'] or sample.packet_id in pairs:
                raise ValueError('Native packet identity is ambiguous')
            pairs[sample.packet_id] = (row['sdk']['sequence'], sample)
        elif row['kind'] == ('student_request' if student else 'teacher_request'):
            requests.append(row)
    episode = manifest['episodes'][0]
    file = resolve_episode_path(root, episode, manifest)
    with load_array(file, allow_pickle=False) as archive:
        original = {k: archive[k].copy() for k in archive.files}
    quality = load(root.parent / 'sample-quality.json')
    if not len(requests) == len(quality) == len(original['action']):
        raise ValueError('One recorded request per original actor observation required')
    updated = {k: v.copy() for k, v in original.items()}
    for index, (request, row) in enumerate(zip(requests, quality, strict=True)):
        packet = (request['actor_input']['native_packet_id'] if student
                  else request['native_packet_id'])
        clock = request['actor_input']['decision_monotonic_seconds'] if student else \
            request['image_candidate']['actor_decision_monotonic_seconds']
        sequence, sample = pairs[packet]
        if (row['source_native_packet_id'] != packet or row['source_sdk_sequence'] != sequence
                or original['observation_time'][index] != clock):
            raise ValueError('Original actor/native row alignment differs')
        fields = {k: sample.fields.get(k) for k in retina.required_native_fields}
        values, masks = retina.body._organs(fields, sample.received_monotonic_seconds, clock)
        values, masks = values.astype(np.float32), masks.astype(np.float32)
        if (not np.array_equal(values[:126], original['obs_organs'][index, :126])
                or not np.array_equal(masks[:126], original['obs_organ_valid'][index, :126])
                or original['obs_organs'][index, 126:].any()
                or original['obs_organ_valid'][index, 126:].any()):
            raise ValueError('Old organ reconstruction differs or yaw slots were already used')
        updated['obs_organs'][index] = values
        updated['obs_organ_valid'][index] = masks
    for key in original:
        if key not in ('obs_organs', 'obs_organ_valid') and not np.array_equal(
                original[key], updated[key]):
            raise ValueError('Visual input, label, mask or time changed')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862839,
        source_manifest_sha256=sha256(root / 'dataset.json'),
        raw_events_sha256=sha256(event_file), sensory_configuration=config))
    target = run / 'dataset'
    target.mkdir()
    destination = target / 'yaw-episode.npz'
    np.savez_compressed(destination, **updated)
    sealed = copy.deepcopy(manifest)
    sealed['episode_storage'] = 'copy'
    sealed.update(source='original sealed observations with raw native yaw pair in reserved slots',
        sensory_calibration=retina.calibration, source_recording_directory=str(recording),
        raw_events_file=event_file.name,
        native_yaw_registration=spec, physical_yaw_units_verified=False,
        sources=[dict(directory=str(root), sha256=sha256(root / 'dataset.json'))],
        episodes=[dict(episode_id='yaw-episode', file=destination.name,
                       sha256=sha256(destination), steps=len(requests))])
    write_json(target / 'dataset.json', sealed)
    write_json(run / 'sample-quality.json', quality)
    load_episodes(target, retina.calibration)
    write_json(run / 'result.json', dict(status='completed', decisions=len(requests),
        new_valid_yaw_rows=int(updated['obs_organ_valid'][:, 126:128].all(1).sum()),
        visual_arrays_labels_masks_and_times_unchanged=True,
        old_organ_slots0_to125_exactly_reproduced=True,
        absolute_heading_or_pose_added=False, game_actions_sent=0,
        dataset_manifest_sha256=sha256(target / 'dataset.json'), goal_achieved=False))


if __name__ == '__main__':
    main()
