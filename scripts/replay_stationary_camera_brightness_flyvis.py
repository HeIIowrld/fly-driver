"""Replay original stationary frame brightness through FlyVis versus first-frame hold."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector, VIEW_ORDER
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices
from flybrain_drive.visual_self_motion import VisualSelfMotionPool
from scripts.audit_stationary_camera_mount_pair import load
from scripts.replay_ac_camera_motion_probe import diagnostic_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config_path = Path('configs/ac-csp-yaw-retina.json')
    result, frames = load(args.source, config_path, 2, count=30)
    if result['status'] != 'completed_stationary_mmf_transport' or not torch.cuda.is_available():
        raise ValueError('Original completed diagnostic and CUDA required')
    model = Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000/000')
    cache = Path('C:/Users/jeff4/AppData/Local/flybrain-bio-20261029/flyvis-data-windows-v1')
    run = begin_run(args.output, dict(program_sha256=sha256(__file__),
        source_result_sha256=sha256(args.source/'result.json'), bundle_sha256=result['archive']['bundle_sha256'],
        model_sha256=sha256(model/'best_chkpt'), config_sha256=sha256(config_path), seed=863009,
        timeline='producer frame script_clock, not physical exposure',
        control='same first frame held in every view for full episode', actor_deployed=False))
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    os.environ['FLYVIS_ROOT_DIR'] = str(cache.resolve())
    install_datamate_cache_compat(cache)
    import flyvis
    torch.set_default_device('cuda')
    network_config = yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))['config']['network']
    net = flyvis.Network(**network_config).to('cuda')
    net.load_state_dict(torch.load(model/'best_chkpt', map_location='cpu', weights_only=True)['network'], strict=True)
    net.eval().requires_grad_(False)
    nodes = net.connectome.nodes
    receptors = net.stimulus.input_index[0]
    u, v = nodes.u[:][receptors], nodes.v[:][receptors]
    indices = column_indices(nodes.type[:].astype(str), nodes.u[:], nodes.v[:], u, v,
                             [f'{kind}{direction}' for kind in ('T4', 'T5') for direction in 'abcd'])
    camera = json.loads(config_path.read_text(encoding='utf8'))
    projector = FlyvisCameraProjector(camera, u, v)
    pool = VisualSelfMotionPool(camera, projector.directions_body)
    clocks = np.asarray([frame['script_clock'] for frame, _ in frames])
    images = np.stack([np.stack([projector.sample(views[name][1]) for name in VIEW_ORDER])
                       for _, views in frames])
    if np.any(np.diff(clocks) <= 0):
        raise ValueError('Original producer clocks must advance')
    activity, features, ticks = [], [], 0
    with torch.no_grad():
        state = net.steady_state(1., .005, batch_size=8, value=.5)
        held = torch.tensor(np.concatenate([images[0], images[0]]), device='cuda')
        for i, now in enumerate(clocks):
            if i:
                for dt in diagnostic_steps(float(now-clocks[i-1])):
                    state = net.simulate(held[:, None, None, :], dt=dt, initial_state=state, as_states=True)[-1]
                    ticks += 1
            held = torch.tensor(np.concatenate([images[i], images[0]]), device='cuda')
            x = state.nodes.activity[:, indices].cpu().numpy()
            activity.append(x)
            features.append(np.stack([pool.features(x[:4]), pool.features(x[4:])]))
    activity, features = np.asarray(activity), np.asarray(features)
    np.savez_compressed(run/'responses.npz', clocks=clocks, retina=images, motion=activity, features=features)
    delta = features[:, 0]-features[:, 1]
    record = dict(status='completed_stationary_flyvis_brightness_diagnostic', frames=len(clocks), ticks=ticks,
                  feature_max_absolute_delta=np.abs(delta).max(axis=0).tolist(),
                  motion_max_absolute_delta=float(np.abs(activity[:, :4]-activity[:, 4:]).max()),
                  responses_sha256=sha256(run/'responses.npz'), physical_exposure_verified=False,
                  physiology_verified=False, actor_deployed=False, training_performed=False, vehicle_control_sent=False)
    write_json(run/'result.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
