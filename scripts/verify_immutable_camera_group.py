"""Compare immutable shared-camera snapshots against retained original MMF images."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector, VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_camera_group import freeze_camera_group, require_matching_camera_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shadow-config', default='configs/flight-stream-batched-v1-20261029.json')
    parser.add_argument('--all-decisions', action='store_true',
                        help='Verify every recorded decision without selecting a subset')
    args = parser.parse_args()
    root = Path(args.recording)
    source = json.loads((root / 'result.json').read_text(encoding='utf8'))
    bundle = json.loads((root / 'vision/bundle.json').read_text(encoding='utf8'))
    archive = root / 'vision' / bundle['file']
    if source['status'] != 'completed' or sha256(archive) != bundle['sha256']:
        raise ValueError('Completed unchanged original camera bundle required')
    config = json.loads((root / 'manifest.json').read_text(encoding='utf8'))['config']
    camera_config = config.get('pixel_configuration') or config['sensory_configuration']
    shadow_config = json.loads(Path(args.shadow_config).read_text(encoding='utf8'))
    shadow_camera = Path(shadow_config['camera_config'])
    if sha256(shadow_camera) != shadow_config['camera_config_sha256']:
        raise ValueError('Pinned shadow camera config changed')
    try:
        require_matching_camera_geometry(camera_config,
            json.loads(shadow_camera.read_text(encoding='utf8')))
        geometry = {'compatible': True, 'reason': None}
    except ValueError as error:
        geometry = {'compatible': False, 'reason': str(error)}
    coordinates = [(q, r) for q in range(-15, 16)
                   for r in range(max(-15, -15-q), min(15, 15-q)+1)]
    projector = FlyvisCameraProjector(camera_config, np.array([x[0] for x in coordinates]),
                                     np.array([x[1] for x in coordinates]))
    rows = [json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) < 8:
        raise ValueError('At least eight original decisions required')
    chosen = (list(range(len(rows))) if args.all_decisions
              else np.linspace(0, len(rows)-1, 8, dtype=int).tolist())
    results = []
    with zipfile.ZipFile(archive) as z:
        for index in chosen:
            event = rows[index]
            camera = event['camera']
            if camera.get('metadata_transport') != 'mmf_slot_sha256_v1':
                raise ValueError('Original MMF metadata receipt required')
            samples, reference = {}, {}
            for view in camera['views']:
                name = view['view_id']
                raw = z.read(view['file'].removeprefix('vision/'))
                if hashlib.sha256(raw).hexdigest() != view['file_sha256']:
                    raise ValueError('Archived image file changed')
                with Image.open(io.BytesIO(raw)) as image:
                    rgb = np.asarray(image.convert('RGB')).copy()
                reference[name] = projector.sample(rgb)
                samples[name] = dict(rgb=rgb, capture_id=camera['capture_id'],
                    generation=camera['generation'], received=camera['received'],
                    render_started_lower=view['render_started_lower'])
            frozen = freeze_camera_group(camera, samples,
                                         selected_at=event['decision_monotonic_seconds'])
            errors = [float(np.max(np.abs(projector.sample(frozen['pixels'][name])-reference[name])))
                      for name in VIEW_ORDER]
            if any(errors) or any(frozen['pixels'][name].flags.writeable for name in VIEW_ORDER):
                raise ValueError('Immutable projection or write protection differs')
            results.append({'event_index': index, 'capture_id': frozen['capture_id'],
                            'maximum_retinal_error': max(errors), 'views': frozen['views']})
    run = begin_run(args.output, {'recording': str(root), 'seed': None,
        'bundle_sha256': bundle['sha256'], 'events_sha256': sha256(root/'events.jsonl'),
        'source_sha256': sha256('src/flybrain_drive/immutable_camera_group.py'),
        'program_sha256': sha256(__file__), 'shadow_config_sha256': sha256(args.shadow_config),
        'shadow_camera_config_sha256': sha256(shadow_camera),
        'selection': 'all original decisions' if args.all_decisions
            else 'eight equally spaced original decisions'})
    result = {'status': 'completed_immutable_camera_group_replay', 'groups': len(results),
              'views': len(results)*4, 'maximum_retinal_error': 0., 'events': results,
              'shadow_camera_geometry': geometry,
              'live_shared_memory_delivery_verified': False,
              'physical_exposure_or_sensor_time_verified': False,
              'training_admissible': False, 'vehicle_control_sent': False}
    write_json(run/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'events'}))


if __name__ == '__main__':
    main()
