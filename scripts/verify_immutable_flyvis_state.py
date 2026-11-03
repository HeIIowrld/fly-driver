"""Compare continuous FlyVis states from archived MMF RGB and immutable snapshots."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_camera_group import (
    freeze_camera_group,
    require_matching_camera_geometry,
)
from flybrain_drive.online_neural_shadow import OnlineNeuralShadow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('recording', 'config', 'training', 'opponent', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    root = Path(a.recording)
    result, manifest = read(root/'result.json'), read(root/'manifest.json')['config']
    bundle = read(root/'vision/bundle.json')
    archive = root/'vision'/bundle['file']
    config = read(a.config)
    if (result['status'] != 'completed' or sha256(archive) != bundle['sha256']
            or sha256(config['camera_config']) != config['camera_config_sha256']):
        raise ValueError('Completed original archive and pinned shadow geometry required')
    require_matching_camera_geometry(manifest.get('pixel_configuration') or
        manifest['sensory_configuration'], read(config['camera_config']))
    rows = [json.loads(x) for x in (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
    if not rows:
        raise ValueError('Complete chronological original decisions required')
    run = begin_run(a.output, {'program_sha256': sha256(__file__), 'recording': str(root),
        'recording_result_sha256': sha256(root/'result.json'), 'bundle_sha256': bundle['sha256'],
        'events_sha256': sha256(root/'events.jsonl'), 'config_sha256': sha256(a.config),
        'adapter_sha256': sha256('src/flybrain_drive/immutable_camera_group.py'),
        'seed': 863009, 'scope': 'offline recorded decision-time paired visual trajectories'})
    output = {'status': 'failed', 'training_admissible': False, 'vehicle_control_sent': False}
    try:
        torch.manual_seed(863009)
        shadow = OnlineNeuralShadow(config_path=a.config, training=a.training, opponent=a.opponent)
        # Graph backend copies its inputs and returns cloned states; branches share the initial
        # immutable grey state but independently advance their own recurrent tensors.
        states = [shadow.vis_state, shadow.vis_state]
        held = [None, None]
        previous_clock, previous_id, generation = None, None, None
        records, duplicates, missing, ticks = [], 0, 0, 0
        with zipfile.ZipFile(archive) as z, torch.no_grad(), torch.device('cuda'):
            for index, event in enumerate(rows):
                camera, clock = event['camera'], event['decision_monotonic_seconds']
                capture = camera['capture_id']
                if camera.get('metadata_transport') != 'mmf_slot_sha256_v1':
                    raise ValueError('Original MMF metadata required')
                if generation is None:
                    generation = camera['generation']
                if (camera['generation'] != generation
                        or previous_id is not None and capture < previous_id):
                    raise ValueError('Original camera reset or order reversed')
                if previous_id is not None:
                    duplicates += int(capture == previous_id)
                    missing += max(0, capture-previous_id-1)
                    batches = held_input_batches(clock-previous_clock)
                    for dt, count in batches:
                        remaining = count
                        while remaining:
                            n = min(remaining, 16)
                            for branch in range(2):
                                movie = held[branch][:, None, None, :].expand(-1, n, -1, -1)
                                states[branch] = shadow.network.simulate(
                                    movie, dt=dt, initial_state=states[branch], as_states=True)[-1]
                            ticks += n
                            remaining -= n
                samples, file_retina = {}, []
                for view in camera['views']:
                    name = view['view_id']
                    raw = z.read(view['file'].removeprefix('vision/'))
                    if hashlib.sha256(raw).hexdigest() != view['file_sha256']:
                        raise ValueError('Original archived camera image differs')
                    with Image.open(io.BytesIO(raw)) as image:
                        rgb = np.asarray(image.convert('RGB')).copy()
                    file_retina.append(shadow.projector.sample(rgb))
                    samples[name] = dict(rgb=rgb, capture_id=capture, generation=generation,
                        received=camera['received'],
                        render_started_lower=view['render_started_lower'])
                frozen = freeze_camera_group(camera, samples, selected_at=clock)
                frozen_retina = np.stack([shadow.projector.sample(frozen['pixels'][v])
                                          for v in VIEW_ORDER])
                file_retina = np.stack(file_retina)
                held = [torch.as_tensor(x, dtype=torch.float32, device='cuda')
                        for x in (file_retina, frozen_retina)]
                state_error = float(torch.max(torch.abs(states[0].nodes.activity -
                                                       states[1].nodes.activity)).item())
                retinal_error = float(np.max(np.abs(file_retina-frozen_retina)))
                scenes = [shadow.model.backbone.scene_pool(torch.cat((held[b][:, None],
                          states[b].nodes.activity[:, shadow.motion_indices]), dim=1)[None])
                          for b in range(2)]
                scene_error = float(torch.max(torch.abs(scenes[0]-scenes[1])).item())
                if max(state_error, retinal_error, scene_error) > 2e-6:
                    raise ValueError('Continuous original visual state or scene differs')
                records.append({'index': index, 'capture_id': capture, 'decision': clock,
                    'state_error': state_error, 'retinal_error': retinal_error,
                    'scene_error': scene_error})
                previous_clock, previous_id = clock, capture
        write_json(run/'events.json', records)
        output.update(status='completed_immutable_flyvis_state_replay', decisions=len(records),
            duplicated_decisions=duplicates, absent_intermediate_captures=missing,
            internal_ticks_per_branch=ticks,
            maximum_state_error=max(x['state_error'] for x in records),
            maximum_retinal_error=max(x['retinal_error'] for x in records),
            maximum_scene_error=max(x['scene_error'] for x in records),
            events_sha256=sha256(run/'events.json'), initialization=shadow.provenance,
            physical_exposure_or_source_time_verified=False, real_time_verified=False,
            body_circuit_or_control_verified=False, optimizer_updates=0)
    except Exception as error:
        output.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', output)
        print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
