"""Reconstruct CSP receptors from archived pixels and native prefix bytes, offline only."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_sensory import MultiViewPacker


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def replay_recording(root, *, allow_native_only=False):
    root = Path(root)
    manifest, result = load(root / 'manifest.json'), load(root / 'result.json')
    native_only = result.get('sdk_omitted') is True
    if (result['status'] != 'completed' or
            (not result['actor_stream_created'] and not (allow_native_only and native_only))):
        raise ValueError('Completed actor-format recording or explicit native-only diagnostic required')
    for entry in load(root / 'archive.json'):
        file = (root / entry['file']).resolve()
        if not file.is_relative_to(root.resolve()) or sha256(file) != entry['sha256']:
            raise ValueError('Sensory archive identity changed')
    initial, final = load(root / 'initial-clock.json'), load(root / 'final-clock.json')
    clock = CSPClockBounds(initial['generation'], maximum_width=initial['maximum_width_seconds'],
        host_quantization=initial.get('host_clock_quantization_seconds', 0.))
    for ping in initial['exchanges']:
        clock.observe(ping['sent'], ping['received'], ping['producer_clock'], initial['generation'])
    if clock.record() != initial:
        raise ValueError('Initial clock bounds do not reproduce')
    retina = CSPCameraRetina(manifest['config']['sensory_configuration'])
    events = [json.loads(x) for x in (root / 'client/events.jsonl').read_text(
        encoding='utf8').splitlines()]
    with load_array(root / 'client/observations.npz', allow_pickle=False) as raw:
        arrays = {key: raw[key].copy() for key in raw.files}
    if (set(arrays) != {'obs_'+key for key in MultiViewPacker.fields} | {'observation_time'}
            or len(events) != result['frames']
            or len(events) != len(arrays['observation_time'])):
        raise ValueError('Exact pose-free observation fields and recording lengths required')
    errors, sources = {key: 0. for key in MultiViewPacker.fields}, []
    for index, event in enumerate(events):
        if (event['kind'] != 'sensory_pair' or event['index'] != index
                or set(event['actor_field_names']) != MultiViewPacker.fields
                or set(event['actor_body_field_names']) != retina.required_native_fields
                or event['decision_monotonic_seconds'] != arrays['observation_time'][index]):
            raise ValueError('Actor fields, source order or decision time changed')
        native, sdk = event['native'], event['sdk']
        if native_only != (sdk is None):
            raise ValueError('SDK absence must match recorded mode')
        for record, tag in ([(native, 'prefix')] if native_only else [(native, 'prefix'), (sdk, 'wire')]):
            raw = base64.b64decode(record[tag+'_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != record[tag+'_sha256']:
                raise ValueError('Raw sensor bytes changed')
        sample = decode_physics_prefix(base64.b64decode(native['prefix_base64']),
                                        native['received_monotonic_seconds'])
        if not native_only:
            sdk_data = json.loads(base64.b64decode(sdk['wire_base64']))
            if sdk_data['producer_session_id'] != result['native_sdk_producer']:
                raise ValueError('SDK producer differs from sealed recording')
        frame_path = root / 'camera' / f'frame-{index:04d}.json'
        if (event['camera_metadata_file'] != frame_path.name
                or sha256(frame_path) != event['camera_metadata_sha256']):
            raise ValueError('Camera metadata identity changed')
        frame = load(frame_path)
        if frame['generation'] != clock.generation or frame['index'] != index:
            raise ValueError('Clock producer or camera order changed')
        image_records = {row['view_id']: row for row in event['image_sources']}
        samples = {}
        for view in frame['views']:
            name = view['view_id']
            image = image_records[name]
            file = root / 'camera' / f'{index:04d}-{name}.png'
            if image['file'] != file.name or sha256(file) != image['sha256']:
                raise ValueError('Source pixel file identity changed')
            with Image.open(file) as raw:
                pixels = np.array(raw.convert('RGB'))
            if hashlib.sha256(pixels.tobytes()).hexdigest() != image['rgb_sha256']:
                raise ValueError('Decoded source pixel hash differs')
            clock.age_upper(event['decision_monotonic_seconds'], view['render_started_clock'],
                            frame['generation'])
            samples[name] = dict(rgb=pixels, capture_id=index, generation=frame['generation'],
                render_started_lower=clock.render_start_lower(
                    view['render_started_clock'], frame['generation']),
                received=event['host_image_receipt'])
        fields = {key: sample.fields.get(key) for key in retina.required_native_fields}
        packed = retina.pack(samples, decision_time=event['decision_monotonic_seconds'],
            native_scalars=fields, native_receipt=sample.received_monotonic_seconds)
        for key, value in packed.items():
            errors[key] = max(errors[key], float(np.max(np.abs(value-arrays['obs_'+key][index]))))
        sources.append(dict(index=index, view_valid=packed['view_valid'].sum(1).tolist(),
                            organ_valid=int(packed['organ_valid'].sum())))
    for ping in final['exchanges'][len(initial['exchanges']):]:
        clock.observe(ping['sent'], ping['received'], ping['producer_clock'], final['generation'])
    if clock.record() != final or any(error != 0 for error in errors.values()):
        raise ValueError('Clock end check or exact pixel/body replay differs')
    return dict(status='completed', frames=len(events), actor_fields=sorted(MultiViewPacker.fields),
        replay_max_absolute_error=errors, exact_replay_verified=True,
        actor_input_pose_free=True, sdk_omitted=native_only, actor_stream_created=not native_only,
        initial_clock_uncertainty_seconds=initial['uncertainty_seconds'],
        final_clock_uncertainty_seconds=final['uncertainty_seconds'],
        sources=sources, actuator_messages_sent=0, physical_exposure_sync_verified=False,
        training_admissible=False, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-native-only', action='store_true', help='Audit explicitly recorded diagnostic SDK absence')
    args = parser.parse_args()
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862809,
        recording_manifest_sha256=sha256(Path(args.recording) / 'manifest.json'),
        actuator_messages_sent=0))
    try:
        result = replay_recording(args.recording, allow_native_only=args.allow_native_only)
    except Exception as error:
        write_json(run / 'result.json', dict(status='failed', error=type(error).__name__,
            reason=str(error), actuator_messages_sent=0, goal_achieved=False))
        raise
    write_json(run / 'result.json', result)


if __name__ == '__main__':
    main()
