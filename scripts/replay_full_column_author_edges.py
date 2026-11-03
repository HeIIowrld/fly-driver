"""Replay pinned author edge stimuli to retain all original retinotopic motion columns."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_readout import column_indices


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    source = Path('runs/flyvis-edge-author-grid-20261029-t1019')
    model = Path('assets/external/flyvis-pretrained-20261029/unpacked/results/flow/0000/000')
    cache = Path('C:/Users/jeff4/AppData/Local/flybrain-bio-20261029/flyvis-data-windows-v1')
    receipt,stimulus = read(source/'result.json'),read(source/'stimulus.json')
    if (receipt['status'] != 'completed' or sha256(source/'responses.npz') != receipt['responses_sha256']
            or stimulus['dt'] != .005 or stimulus['mode'] != 'edge'):
        raise ValueError('Exact successful original author edge experiment required')
    params = stimulus['parameters']
    cases = [(contrast,angle) for contrast in (0,1) for angle in (0,180)]
    indices = [next(i for i,x in enumerate(params) if x['intensity'] == intensity
                    and x['angle'] == angle and x['speed'] == 19.) for intensity,angle in cases]
    with load_array(source/'responses.npz',allow_pickle=False) as archive:
        lengths = archive['valid_frames'][indices]
        length = int(min(lengths))
        movies = archive['stimulus'][indices,:length].copy()
        prior = archive['activity'][indices,:length].copy()
        names = archive['central_cell_types'].tolist()
        old_central = archive['central_indices'].copy()
    if not np.isfinite(movies).all() or not np.isfinite(prior).all():
        raise ValueError('Complete original finite stimulus and reference required')
    run = begin_run(a.output,dict(program_sha256=sha256(__file__),
        original_responses_sha256=receipt['responses_sha256'], stimulus_sha256=sha256(source/'stimulus.json'),
        checkpoint_sha256=sha256(model/'best_chkpt'), cases=cases, indices=indices,
        frames=length, dt=.005, seed=863009, optimizer_updates=0,
        scope='synthetic author edges, full retinotopic FlyVis columns, no vehicle'))
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False,
        physiological_reproduction_verified=False, vehicle_control_sent=False)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(863009)
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required')
        os.environ['FLYVIS_ROOT_DIR'] = str(cache.resolve())
        os.environ['MPLBACKEND'] = 'Agg'
        install_datamate_cache_compat(cache)
        import flyvis
        torch.set_default_device('cuda')
        config = yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))['config']['network']
        network = flyvis.Network(**config).to('cuda')
        weights = torch.load(model/'best_chkpt',map_location='cpu',weights_only=True)['network']
        network.load_state_dict(weights,strict=True)
        network.eval().requires_grad_(False)
        nodes = network.connectome.nodes
        receptors = network.stimulus.input_index[0]
        u,v = nodes.u[:][receptors],nodes.v[:][receptors]
        motion_types = [f'{kind}{direction}' for kind in ('T4','T5') for direction in 'abcd']
        selected = column_indices(nodes.type[:].astype(str),nodes.u[:],nodes.v[:],u,v,motion_types)
        central = network.connectome.central_cells_index[:]
        if not np.array_equal(central,old_central):
            raise ValueError('Original central-cell registration changed')
        outputs,errors = [],[]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for j,movie in enumerate(movies):
                x = torch.as_tensor(movie,device='cuda')
                activity = network.simulate(x[None],dt=.005,initial_state='auto')[0]
                central_current = activity[:,central].cpu().numpy()
                error = float(np.max(np.abs(central_current-prior[j])))
                if error > 2e-6:
                    raise ValueError(f'Original central responses differ by {error}')
                columns = activity[:,selected].cpu().numpy()
                if not np.isfinite(columns).all():
                    raise ValueError('Nonfinite column activity')
                outputs.append(columns)
                errors.append(error)
                write_json(run/'progress.json',dict(completed=j+1,total=len(movies),central_max_error=error))
                del activity,columns,x
        path = run/'columns.npz'
        np.savez_compressed(path,retina=movies[:,:,0,:],motion=np.stack(outputs),
                            case_indices=np.asarray(indices),dt=np.array(.005))
        result.update(status='completed_author_edge_full_columns',cases=cases,frames=length,
            central_replay_max_error=errors,columns_sha256=sha256(path),
            peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
            spatial_columns_are_original_network=True, synthetic_visual_stimuli=True,
            same_stimulus_applied_to_one_view_not_proven_real_fly_optics=True)
    except Exception as e:
        result.update(error=type(e).__name__,reason=str(e))
    finally:
        write_json(run/'result.json',result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
