"""Replay actual camera/body inputs to the existing flight-stream input boundary."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import CameraSequence
from flybrain_drive.flyvis_camera import VIEW_ORDER, FlyvisCameraProjector, integration_intervals
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import VirtualBodyEncoder, decode_virtual_body_prefix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, audit = Path(args.recording), Path(args.audit)
    result, proof, audit_manifest = (read(root/'result.json'), read(audit/'result.json'),
                                     read(audit/'manifest.json'))
    if (result['status'] != 'completed' or proof['status'] != 'completed'
            or not proof['producer_time_metadata_replay_verified']
            or audit_manifest['config']['events_sha256'] != sha256(root/'events.jsonl')
            or audit_manifest['config']['observation_sha256'] != sha256(root/'observations.npz')):
        raise ValueError('Original completed producer-time camera replay proof required')
    config, manifest = read(args.config), read(root/'manifest.json')
    if config['schema'] != 'flybrain.flight_stream.v1':
        raise ValueError('Original flight stream configuration required')
    for key in ('camera_config', 'body_contract', 'retina_registration'):
        if sha256(config[key]) != config[key+'_sha256']:
            raise ValueError('Original flight input source changed: '+key)
    camera_config = read(config['camera_config'])
    if manifest['config']['sensory_configuration']['views'] != camera_config['views']:
        raise ValueError('Recorded and flight camera configurations differ')
    with load_array(config['retina_registration'], allow_pickle=False) as reg:
        projector = FlyvisCameraProjector(camera_config, reg['u'], reg['v'])
        np.testing.assert_array_equal(projector.uv, reg['image_uv'])
        np.testing.assert_array_equal(projector.directions_body, reg['body_directions'])
    body = VirtualBodyEncoder(config['body_contract'], native_identity=read(root/'identity.json'))
    sequence = CameraSequence(maximum_interval=config['maximum_interval_seconds'],
                              maximum_camera_age=config['maximum_camera_age_seconds'])
    events = [json.loads(line) for line in (root/'events.jsonl').open(encoding='utf8')]
    hashes = {str(p): sha256(p) for p in (root/'events.jsonl', root/'identity.json',
        root/'vision/camera.zip', root/'vision/bundle.json', audit/'result.json',
        audit/'manifest.json', Path(args.config))}
    bundle = read(root/'vision/bundle.json')
    if hashes[str(root/'vision/camera.zip')] != bundle['sha256']:
        raise ValueError('Original camera ZIP changed')
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'source_hashes': hashes,
        'scope': 'actual inputs through sequence, pixel projection, and body encoder only',
        'optimizer_updates': 0, 'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    views, masks, times = [], [], []
    ticks, repeated = 0, 0
    with CameraArchive(root/'vision') as archive:
        for i, event in enumerate(events):
            camera, raw_native = event['camera'], event['native']
            pixels = {}
            for view in camera['views']:
                raw = archive.read(Path(view['file']).name)
                if hashlib.sha256(raw).hexdigest() != view['file_sha256']:
                    raise ValueError('Original camera pixel bytes changed')
                rgb = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
                if hashlib.sha256(rgb.tobytes()).hexdigest() != view['rgb_sha256']:
                    raise ValueError('Original camera RGB changed')
                pixels[view['view_id']] = rgb
            clock = event['decision_monotonic_seconds']
            elapsed, signature = sequence.preview(clock, camera, pixels)
            native = decode_virtual_body_prefix(base64.b64decode(
                raw_native['prefix_base64'], validate=True),
                raw_native['received_monotonic_seconds'])
            if native.prefix_sha256 != raw_native['prefix_sha256']:
                raise ValueError('Original body bytes changed')
            encoded = body.encode(native, episode_id='producer-time-live-probe',
                                  decision_monotonic_seconds=clock)
            if not encoded.valid[[0, 1, 4]].all():
                raise ValueError('Flight stream required tilt/yaw invalid')
            repeated += int(camera['capture_id'] == sequence.capture)
            if elapsed is not None:
                ticks += len(integration_intervals(elapsed))
            views.append(np.stack([projector.sample(pixels[name]) for name in VIEW_ORDER]))
            masks.append(encoded.valid)
            times.append(clock)
            sequence.commit(clock, camera, signature)
            if (i+1) % 25 == 0:
                print(json.dumps({'input_observations': i+1}), flush=True)
    if len(views) != result['decisions']:
        raise ValueError('Every original camera decision required')
    np.savez_compressed(run/'projected-inputs.npz', retinal=np.stack(views),
                        body_valid=np.stack(masks), decision_clock=np.array(times))
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('Original input evidence changed')
    output = {'status': 'completed_actual_fourview_flight_input_preflight',
        'observations': len(views), 'retinal_shape': list(np.stack(views).shape),
        'retinal_input_sha256': sha256(run/'projected-inputs.npz'),
        'body_valid_counts': np.stack(masks).sum(axis=0).tolist(),
        'reused_capture_decisions': repeated, 'scheduled_visual_integration_ticks': ticks,
        'flyvis_network_executed': False, 'flight_network_executed': False,
        'physical_exposure_or_body_ray_registration_verified': False,
        'optimizer_updates': 0, 'actor_deployed': False, 'goal_achieved': False}
    write_json(run/'result.json', output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
