"""Re-encode recorded wheel signals into unused slots; images and labels stay unchanged."""

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_pixel_retina import NativePixelRetina
from flybrain_drive.episode_storage import load_array, resolve_episode_path
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes


def find_recording(root):
    seen = set()
    while True:
        root = Path(root).resolve()
        if root in seen:
            raise ValueError('Cyclic source lineage')
        seen.add(root)
        manifest = json.loads((root / 'dataset.json').read_text(encoding='utf8'))
        generated = json.loads((root.parent / 'manifest.json').read_text(encoding='utf8'))
        args = generated['config'].get('arguments', {})
        if args.get('run'):
            return Path(args['run']), generated['config']['pixel_configuration'], manifest
        sources = manifest.get('sources', [])
        if len(sources) != 1:
            raise ValueError('Augment individual recording leaves, not merged datasets')
        source = sources[0]
        root = Path(source['directory'])
        if sha256(root / 'dataset.json') != source['sha256']:
            raise ValueError('Source lineage manifest changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--retina-config', default='configs/ac-native-wheel-retina.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.input)
    manifest = json.loads((root / 'dataset.json').read_text(encoding='utf8'))
    if len(manifest['episodes']) != 1:
        raise ValueError('Exactly one original recording episode required')
    load_episodes(root, manifest['sensory_calibration'])
    front = NativePixelRetina(json.loads(Path(args.retina_config).read_text(encoding='utf8')))
    if front.wheel_spec is None:
        raise ValueError('Explicit wheel encoding configuration required')
    recording, original_config, recording_manifest = find_recording(root)
    original_front = NativePixelRetina(original_config)
    if original_front.wheel_spec is not None:
        raise ValueError('Source slots already have wheel semantics')
    if manifest['sensory_calibration'] != original_front.calibration:
        raise ValueError('Source body/visual calibration does not match its original encoder')
    for key in ('image_shape', 'receptors_per_view', 'columns', 'acceptance_sigma_pixels',
                'max_view_age', 'contrast_scale', 'regions', 'scalar_organs'):
        if front.config[key] != original_config[key]:
            raise ValueError('Only reserved wheel channels may change, not other sensors or optics')
    events_file = recording / recording_manifest.get('raw_events_file',
                                                       'observations-and-requests.jsonl')
    if sha256(events_file) != manifest['raw_events_sha256']:
        raise ValueError('Original raw recording changed')
    pairs = {}
    for row in (json.loads(s) for s in events_file.read_text(encoding='utf8').splitlines()):
        if row['kind'] != 'raw_pair':
            continue
        for source, tag in [('sdk', 'wire'), ('native', 'prefix')]:
            raw = base64.b64decode(row[source][tag+'_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != row[source][tag+'_sha256']:
                raise ValueError('Recorded sensor bytes changed')
        key = (row['sdk']['sequence'], row['native']['packet_id'])
        if key in pairs:
            raise ValueError('Duplicate recorded sensor identity')
        native = row['native']
        sample = decode_physics_prefix(base64.b64decode(native['prefix_base64']),
                                        native['received_monotonic_seconds'])
        if sample.packet_id != key[1]:
            raise ValueError('Native identity differs from raw prefix')
        pairs[key] = sample
    episode = manifest['episodes'][0]
    path = resolve_episode_path(root, episode, manifest)
    if sha256(path) != episode['sha256']:
        raise ValueError('Input episode path/hash differs')
    with load_array(path, allow_pickle=False) as stored:
        original = {key: stored[key] for key in stored.files}
    arrays = {key: value.copy() for key, value in original.items()}
    if original['obs_organs'][:, :18].any() or original['obs_organ_valid'][:, :18].any():
        raise ValueError('Reserved slots must previously be explicitly invalid and zero')
    quality = json.loads((root.parent / 'sample-quality.json').read_text(encoding='utf8'))
    if len(quality) != len(original['action']):
        raise ValueError('One source identity per actor row required')
    for index, item in enumerate(quality):
        sample = pairs[(item['source_sdk_sequence'], item['source_native_packet_id'])]
        fields = {key: sample.fields.get(key) for key in front.required_native_fields}
        values, valid = front._organs(fields, sample.received_monotonic_seconds,
                                       float(original['observation_time'][index]))
        values, valid = values.astype(np.float32), valid.astype(np.float32)
        if (not np.array_equal(values[18:], original['obs_organs'][index, 18:])
                or not np.array_equal(valid[18:], original['obs_organ_valid'][index, 18:])):
            raise ValueError('Existing native organs do not reproduce; no silent sensor changes')
        arrays['obs_organs'][index, :18] = values[:18]
        arrays['obs_organ_valid'][index, :18] = valid[:18]
        item['wheel_rotation_valid_features'] = int(valid[:18].sum())
        item['native_wheel_rotation_observed'] = fields['wheelAngularSpeed'] is not None
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862101,
        source_manifest_sha256=sha256(root / 'dataset.json'),
        new_pixel_configuration=front.config, wheel_units_independently_validated=False))
    dataset = run / 'dataset'
    dataset.mkdir()
    output = dataset / 'native-segment.npz'
    np.savez_compressed(output, **arrays)
    with load_array(output, allow_pickle=False) as saved:
        for key in original:
            if key in ('obs_organs', 'obs_organ_valid'):
                unchanged = np.array_equal(saved[key][:, 18:], original[key][:, 18:])
            else:
                unchanged = np.array_equal(saved[key], original[key])
            if not unchanged:
                raise ValueError('Source views, other organs, labels, masks or clocks changed')
    updated = copy.deepcopy(manifest)
    updated['episode_storage'] = 'copy'
    updated.update(sensory_calibration=front.calibration,
        source='measured AC actor states with recorded wheel proprioception and fixed-task labels',
        source_actor_arrays_unchanged=False, only_organ_slots_0_17_changed=True,
        visual_arrays_labels_and_clocks_unchanged=True,
        wheel_rotation_units='native game signal, independently uncalibrated',
        sources=[dict(directory=str(root), sha256=sha256(root / 'dataset.json'))],
        episodes=[dict(episode_id='native-segment', file=output.name,
                       sha256=sha256(output), steps=len(quality))])
    write_json(dataset / 'dataset.json', updated)
    write_json(run / 'sample-quality.json', quality)
    load_episodes(dataset, front.calibration)
    write_json(run / 'result.json', dict(status='completed', decisions=len(quality),
        only_reserved_organ_slots_changed=True, actor_fields_pose_free=True,
        visual_arrays_labels_clocks_and_other_organs_unchanged=True,
        first_organ_valid_features=int(arrays['obs_organ_valid'][0].sum()),
        all_organ_valid_feature_counts=np.unique(arrays['obs_organ_valid'].sum(1)).tolist(),
        physical_units_validated=False, game_actions_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
