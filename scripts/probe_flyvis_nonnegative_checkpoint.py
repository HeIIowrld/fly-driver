"""Run a pinned author visual network on component stimuli; never controls a vehicle."""

import argparse
import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.flyvis_parameter_integrity import assert_nonnegative_synapse_strength


def movie_with_channel_axis(movie, input_elements):
    """Author flash datasets omit the singleton input-channel axis; preserve all samples."""
    if movie.ndim == 2:
        movie = movie[:, None, :]
    if movie.ndim != 3 or movie.shape[1:] != (1, input_elements):
        raise ValueError('Stimulus must retain frame, singleton channel and input hexal axes')
    return movie


def valid_stimulus_prefix(movie, allow_author_padding=False):
    """Accept only finite frames followed by all-NaN storage padding, never holes."""
    finite = torch.isfinite(movie).all(dim=(1, 2))
    count = int(finite.sum().item())
    if count == 0 or not bool(finite[:count].all()):
        raise ValueError('Stimulus has missing or invalid frames inside its temporal history')
    if count < len(movie):
        if not allow_author_padding or not bool(torch.isnan(movie[count:]).all()):
            raise ValueError('Only declared author all-NaN trailing storage padding is allowed')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--checkpoint-override', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--mode', choices=['smoke', 'flash', 'edge'], default='smoke')
    parser.add_argument('--edge-protocol', choices=['cardinal', 'author-grid'], default='cardinal')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU unavailable; no CPU fallback')
    os.environ['FLYVIS_ROOT_DIR'] = str(Path(args.cache_root).resolve())
    os.environ['MPLBACKEND'] = 'Agg'
    root = Path(args.model_dir)
    checkpoint = Path(args.checkpoint_override)
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError('Exact declared checkpoint required')
    run = begin_run(args.output, dict(arguments=vars(args), seed=863009, scope=__doc__,
        checkpoint_sha256=sha256(checkpoint), config_sha256=sha256(root/'_meta.yaml'),
        source_registry_sha256=sha256('assets/external/flyvis-source-20261029/source.json'),
        pretrained_registry_sha256=sha256('assets/external/flyvis-pretrained-20261029/source.json'),
        actuator_messages_sent=0))
    result = dict(status='failed', actuator_messages_sent=0, optimizer_updates=0,
                  physiological_reproduction_verified=False, AC_integration_verified=False,
                  goal_achieved=False)
    try:
        compatibility = install_datamate_cache_compat(args.cache_root)
        import flyvis

        torch.set_default_device(args.device)
        config = yaml.safe_load((root/'_meta.yaml').read_text(encoding='utf8'))['config']['network']
        network = flyvis.Network(**config).to(args.device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if Path(args.checkpoint_override).resolve() != (root/'best_chkpt').resolve() and payload.get('actor_qualified') is not False:
            raise ValueError('Experimental candidate must explicitly forbid actor qualification')
        loaded = payload['network']
        assert_nonnegative_synapse_strength(loaded['edges_syn_strength'])
        network.load_state_dict(loaded, strict=True)
        network.eval().requires_grad_(False)
        write_json(run/'network.json', dict(nodes=network.n_nodes, edges=network.n_edges,
            input_elements=network.stimulus.n_input_elements,
            parameter_devices=sorted({str(p.device) for p in network.parameters()}),
            compatibility=compatibility, strict_checkpoint_load=True,
            versions={name: importlib.metadata.version(name)
                      for name in ('flyvis', 'datamate', 'torch', 'torchvision', 'numpy', 'h5py')}))
        central = network.connectome.central_cells_index[:]
        names = network.connectome.nodes.type[:].astype(str)[central]
        if args.mode == 'smoke':
            dt = .005
            movies = [torch.full((4, 1, network.stimulus.n_input_elements), .5,
                                 device=args.device)]
            parameters = [dict(kind='grey4frame_smoke_only')]
        else:
            if args.mode == 'flash':
                from flyvis.datasets.flashes import Flashes

                dataset = Flashes(dynamic_range=[0, 1], t_stim=1., t_pre=1., dt=.005,
                                  radius=[-1, 6], alternations=(0, 1, 0))
            else:
                from flyvis.datasets.moving_bar import MovingEdge

                full = args.edge_protocol == 'author-grid'
                dataset = MovingEdge(offsets=[-10, 11], intensities=[0, 1],
                    speeds=[2.4, 4.8, 9.7, 13, 19, 25] if full else [19],
                    height=80, post_pad_mode='continue', t_pre=1., t_post=1., dt=.005,
                    angles=list(range(0, 360, 30)) if full else [0, 90, 180, 270],
                    device=args.device)
            dt = dataset.dt
            parameters = dataset.arg_df.to_dict(orient='records')
            movies = [movie_with_channel_axis(dataset[i], network.stimulus.n_input_elements)
                      .to(args.device) for i in range(len(dataset))]
            write_json(run/'stimulus.json', dict(mode=args.mode, dt=dt,
                edge_protocol=args.edge_protocol if args.mode == 'edge' else None,
                parameters=parameters, source='author dataset implementation',
                independent_stimulus_episodes=True, neural_reset_within_episode=False))
        responses, inputs, timings, valid_lengths = [], [], [], []
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for index, movie in enumerate(movies):
                valid_length = valid_stimulus_prefix(movie,
                    allow_author_padding=args.mode == 'edge'
                    and args.edge_protocol == 'author-grid')
                started = time.perf_counter()
                # Each stimulus is an independent episode; its whole temporal history is kept.
                activity = network.simulate(movie[None], dt=dt,
                    initial_state=None if args.mode == 'smoke' else 'auto')
                if not torch.isfinite(activity[:, :valid_length]).all():
                    raise ValueError('Nonfinite author-network response')
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                responses.append(activity[0, :, central].cpu().numpy())
                inputs.append(movie.cpu().numpy())
                timings.append(time.perf_counter()-started)
                valid_lengths.append(valid_length)
                write_json(run/'progress.json', dict(completed=index+1, total=len(movies),
                    last_seconds=timings[-1], observation_device=str(movie.device)))
                del activity
        np.savez_compressed(run/'responses.npz', activity=np.stack(responses),
                            stimulus=np.stack(inputs), central_cell_types=names,
                            central_indices=central, dt=np.array(dt),
                            valid_frames=np.array(valid_lengths))
        changed = [key for key, value in network.state_dict().items()
                   if key in loaded and not torch.equal(value.cpu(), loaded[key].cpu())]
        result.update(status='completed', mode=args.mode, samples=len(responses),
            central_cell_types=names.tolist(), frames_per_sample=len(responses[0]),
            dt=dt, finite_responses=all(n == len(movies[0]) for n in valid_lengths),
            finite_valid_responses=True, valid_frames=valid_lengths,
            padding_policy=('Original all-NaN trailing storage padding retained; '
                            'no actual frames removed'),
            seconds_per_sample=timings,
            weights_changed_by_author_forward_clamp=changed,
            responses_sha256=sha256(run/'responses.npz'),
            peak_gpu_tensor_bytes=torch.cuda.max_memory_allocated()
                if args.device == 'cuda' else None,
            scope='Technical execution and raw responses, not full biological replication')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run/'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
