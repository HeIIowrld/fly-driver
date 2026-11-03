"""Replay unchanged four-view synthetic yaw movies with two FlyVis checkpoints."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flyvis_parameter_integrity import assert_nonnegative_synapse_strength
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    run = begin_run(args.output, dict(plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), actor=False, optimizer_updates=0))
    result = dict(status='failed', actor_qualified=False, vehicle_control_sent=False)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(863009)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        model = Path(plan['model_dir'])
        candidate = Path(plan['candidate'])
        if sha256(model/'best_chkpt') != plan['original_sha256'] or sha256(candidate) != plan['candidate_sha256']:
            raise ValueError('checkpoint hash mismatch')
        os.environ['FLYVIS_ROOT_DIR'] = str(Path(plan['cache']).resolve())
        install_datamate_cache_compat(plan['cache'])
        import flyvis
        torch.set_default_device('cuda')
        config = yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))['config']['network']
        net = flyvis.Network(**config).to('cuda').eval().requires_grad_(False)
        nodes = net.connectome.nodes
        receptor = net.stimulus.input_index[0]
        u, v = nodes.u[:][receptor], nodes.v[:][receptor]
        indices = column_indices(nodes.type[:].astype(str), nodes.u[:], nodes.v[:], u, v,
            [f'{kind}{direction}' for kind in ('T4', 'T5') for direction in 'abcd'])
        output = {}
        stats = []
        for source in plan['sources']:
            path = Path(source['path'])
            if sha256(path) != source['sha256']:
                raise ValueError('source hash mismatch')
            with load_array(path, allow_pickle=False) as z:
                retina, prior, times = z['retina'].copy(), z['motion'].copy(), z['times'].copy()
                bends, yaws = z['bend'].copy(), z['yaw_rate'].copy()
            if retina.shape != (3,3,41,4,721) or prior.shape != (3,3,41,4,8,721) or not np.isfinite(retina).all() or not np.array_equal(bends,[0,-1,1]) or not np.allclose(yaws,[0.,-.12,.12]):
                raise ValueError('invalid source dimensions')
            retina=retina.reshape(9,41,4,721);prior=prior.reshape(9,41,4,8,721)
            replay = []
            for label, cp in [('original', model/'best_chkpt'), ('candidate', candidate)]:
                payload = torch.load(cp, map_location='cpu', weights_only=True)
                if label == 'candidate' and payload.get('actor_qualified') is not False:
                    raise ValueError('experimental qualification missing')
                assert_nonnegative_synapse_strength(payload['network']['edges_syn_strength'])
                net.load_state_dict(payload['network'], strict=True)
                responses = []
                for case in range(9):
                    movie = torch.tensor(np.transpose(np.repeat(retina[case,:-1],8,axis=0),(1,0,2))[:,:,None,:],device='cuda')
                    with torch.no_grad():
                        state = net.steady_state(1.,.005,batch_size=4,value=.5)
                        first = state.nodes.activity[:,indices].cpu().numpy()
                        activity = net.simulate(movie,dt=.005,initial_state=state)
                        motion = np.concatenate([first[None],activity[:,7::8,indices].permute(1,0,2,3).cpu().numpy()],axis=0)
                    if not np.isfinite(motion).all():
                        raise ValueError('nonfinite response')
                    responses.append(motion)
                    del movie, activity, state
                replay.append(np.stack(responses))
            error = float(np.max(np.abs(replay[0]-prior)))
            if error > 1e-5:
                raise ValueError(f'original source replay mismatch: {error}')
            key = str(source.get('key', source['seed']))
            if key+'_retina' in output:
                raise ValueError('duplicate source key')
            output[key+'_retina'] = retina.reshape(3,3,41,4,721)
            output[key+'_bend'] = bends
            output[key+'_yaw'] = yaws
            output[key+'_original'] = replay[0].reshape(3,3,41,4,8,721)
            output[key+'_candidate'] = replay[1].reshape(3,3,41,4,8,721)
            output[key+'_times'] = times
            stats.append(dict(seed=source['seed'], original_replay_max_abs=error,
                candidate_difference_rms=float(np.sqrt(np.mean((replay[1]-replay[0])**2))),
                candidate_difference_max_abs=float(np.max(np.abs(replay[1]-replay[0])))))
            write_json(run/'progress.json',dict(completed=len(stats), total=len(plan['sources'])))
        path=run/'responses.npz'
        np.savez_compressed(path,**output)
        result.update(status='completed', sources=stats, responses_sha256=sha256(path),
            scope='Same saved synthetic retinal input; no biological or vehicle qualification')
    except Exception as exc:
        result.update(error=type(exc).__name__,reason=str(exc))
    write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
