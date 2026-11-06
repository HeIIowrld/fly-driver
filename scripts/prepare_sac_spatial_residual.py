"""Recover original retinal intensity before pooling from qualified camera pixels.

No FlyVis re-simulation or invented T4/T5 states. Reconstruct the original static
intensity pool first, then quantify information provably discarded by that pool.
All image identities, clocks and model-event joins stay from the old episode.
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flyvis_camera import FlyvisCameraProjector, VIEW_ORDER
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_pooling import window_pool_weights
from flybrain_drive.sac_unified_student import assemble_student, read
from flybrain_drive.spatial_pool_residual import pool_null_projector, discarded_retinal_component
from scripts.diagnose_sac_anticipation import load_verified
from scripts.sac_cohort_contract import DEFAULT_SPLIT,three_episode_cohort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--split',default=DEFAULT_SPLIT)
    a = p.parse_args()
    torch.set_num_threads(1)
    entries = three_episode_cohort(a.split)
    data = [load_verified(entry) for entry in entries]
    model, _ = assemble_student()
    spec = model.motor.backbone.spec
    old = model.scene_pool.weight.numpy().copy()
    with load_array(spec['scene_registration_file'], allow_pickle=False) as z:
        u, v, uv = z['u'].copy(), z['v'].copy(), z['image_uv'].copy()
    config = read('configs/ac-csp-exterior-front-v3-retina.json')
    projector = FlyvisCameraProjector(config, u, v)
    if not np.allclose(projector.uv, uv, atol=1e-8, rtol=0):
        raise ValueError('Original retinal registration and camera sampling coordinates differ')
    inverse = pool_null_projector(old)
    pools = {name: window_pool_weights(uv, bounds=(0, 1, 0, 1),
                 columns=side, rows=side, sigma=sigma)
             for name, side, sigma in [('grid4_wide', 4, .18), ('grid8_wide', 8, .18),
                                      ('grid4_narrow', 4, .07), ('grid8_narrow', 8, .07)]}
    if not np.allclose(pools['grid4_wide'], old, atol=1e-7, rtol=0):
        raise ValueError('Original pool reconstruction differs')
    run = begin_run(a.output, dict(arguments=vars(a), seed=780402, program_sha256=sha256(__file__),
        entries=entries, retinal_registration_sha256=spec['scene_registration_sha256'],
        camera_configuration_sha256=sha256('configs/ac-csp-exterior-front-v3-retina.json'),
        pool_variants={k: dict(shape=list(v.shape)) for k, v in pools.items()},
        interpretation='Static sampled LDR intensity only; pre-pooling T4/T5 motion information not recovered',
        nullspace='x - (x P.T) pinv(P).T, independent of labels; residual P.T must vanish',
        optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False))
    np.savez_compressed(run / 'pool-geometry.npz', old=old, inverse=inverse, uv=uv, **pools)
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False)
    begin = time.monotonic()
    try:
        reports, hashes = [], {}
        for episode, (entry, d) in enumerate(zip(entries, data)):
            source_cfg = read(Path(entry['data']) / 'manifest.json')['config']
            recording = Path(source_cfg['arguments']['recording'])
            source_cert = read(Path(source_cfg['arguments']['qualification']) / 'result.json')
            needed = np.unique(d['camera_index']).astype(int)
            image_features, ids, old_errors, null_errors = [], [], [], []
            with CameraArchive(recording / 'camera/vision') as archive:
                if archive.bundle_sha256 != source_cert['camera_bundle_sha256']:
                    raise ValueError('Qualified original camera bundle differs')
                stream = [json.loads(line) for line in archive.read('stream.jsonl').decode('utf8').splitlines()]
                for camera_index in needed:
                    row = stream[camera_index]
                    frame = archive.json(Path(row['file']).name)
                    byname = {item['view_id']: item for item in frame['views']}
                    retina = []
                    for name in VIEW_ORDER:
                        rgb = np.asarray(Image.open(io.BytesIO(archive.read(byname[name]['file']))).convert('RGB')).copy()
                        retina.append(np.tanh(projector.sample(rgb)))
                    retina = np.asarray(retina)
                    event = int(np.flatnonzero(d['camera_index'] == camera_index)[0])
                    expected = d['scene'][event].reshape(4, 9, 16)[:, 0]
                    error = float(abs(retina @ old.T - expected).max())
                    if error > 2e-6:
                        raise ValueError(f'Original static scene pool differs at camera {camera_index}: {error}')
                    residual = discarded_retinal_component(retina, old, inverse)
                    null_error = float(abs(residual @ old.T).max())
                    if null_error > 1e-6:
                        raise ValueError('Discarded component is visible to the old pool')
                    image_features.append(retina)
                    ids.append(camera_index)
                    old_errors.append(error)
                    null_errors.append(null_error)
                    if len(ids) % 100 == 0:
                        write_json(run / 'progress.json', dict(episode=episode, camera_frames=len(ids),
                            total=len(needed), elapsed_seconds=time.monotonic()-begin))
            retina = np.asarray(image_features)
            residual = discarded_retinal_component(retina, old, inverse)
            arrays = dict(camera_indices=np.asarray(ids), retina_tanh=retina,
                event_to_camera=np.searchsorted(np.asarray(ids), d['camera_index']))
            for name, pool in pools.items():
                arrays[name] = (retina @ pool.T).reshape(len(ids), -1).astype(np.float32)
            arrays['discarded_grid8'] = (residual @ pools['grid8_narrow'].T).reshape(len(ids), -1).astype(np.float32)
            path = run / f'spatial-{episode}.npz'
            np.savez_compressed(path, **arrays)
            hashes[path.name] = sha256(path)
            reports.append(dict(camera_groups=len(ids), original_events=len(d['labels']),
                original_intensity_pool_max_error=max(old_errors), discarded_old_pool_max_error=max(null_errors)))
        result.update(status='completed_qualified_static_spatial_recovery', episodes=reports,
            files_sha256=hashes, geometry_sha256=sha256(run / 'pool-geometry.npz'),
            physiological_validation=False, prepool_motion_recovered=False,
            termination_reason='All original actor image groups sampled; old static pooling and nullspace identity verified')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit spatial recovery failure')
    result['elapsed_seconds'] = time.monotonic()-begin
    write_json(run / 'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
