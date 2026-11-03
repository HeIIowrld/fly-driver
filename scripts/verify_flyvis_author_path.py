"""Compare saved wrapper output against the unmodified author stimulus_response path."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--flash-run', required=True)
    parser.add_argument('--edge-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required, no fallback')
    os.environ['FLYVIS_ROOT_DIR'] = str(Path(args.cache_root).resolve())
    os.environ['MPLBACKEND'] = 'Agg'
    model = Path(args.model_dir)
    run = begin_run(args.output, dict(arguments=vars(args), seed=863009,
        checkpoint_sha256=sha256(model/'best_chkpt'),
        model_config_sha256=sha256(model/'_meta.yaml'), actuator_messages_sent=0,
        tolerance=dict(rtol=2e-5, atol=2e-6)))
    result = dict(status='failed', actuator_messages_sent=0, optimizer_updates=0,
                  physiological_reproduction_verified=False, AC_integration_verified=False)
    try:
        install_datamate_cache_compat(args.cache_root)
        import flyvis
        from flyvis.datasets.flashes import Flashes
        from flyvis.datasets.moving_bar import MovingEdge

        torch.set_default_device('cuda')
        config = yaml.safe_load((model/'_meta.yaml').read_text(encoding='utf8'))
        network = flyvis.Network(**config['config']['network']).to('cuda')
        state = torch.load(model/'best_chkpt', map_location='cpu', weights_only=True)['network']
        network.load_state_dict(state, strict=True)
        network.eval().requires_grad_(False)
        indices = network.connectome.central_cells_index[:]
        summaries = {}
        for mode, directory in [('flash', args.flash_run), ('edge', args.edge_run)]:
            directory = Path(directory)
            saved_result = json.loads((directory/'result.json').read_text(encoding='utf8'))
            if saved_result['status'] != 'completed':
                raise ValueError('Original response run must be completed')
            original_hash = sha256(directory/'responses.npz')
            if original_hash != saved_result['responses_sha256']:
                raise ValueError('Original response hash mismatch')
            original = load_array(directory/'responses.npz', allow_pickle=False)
            declared = json.loads((directory/'stimulus.json').read_text(encoding='utf8'))
            if mode == 'flash':
                dataset = Flashes(dynamic_range=[0, 1], t_stim=1., t_pre=1., dt=.005,
                    radius=[-1, 6], alternations=(0, 1, 0))
            else:
                dataset = MovingEdge(offsets=[-10, 11], intensities=[0, 1],
                    speeds=[2.4, 4.8, 9.7, 13, 19, 25], height=80,
                    post_pad_mode='continue', t_pre=1., t_post=1., dt=.005,
                    angles=list(range(0, 360, 30)), device='cuda')
            if dataset.arg_df.to_dict(orient='records') != declared['parameters']:
                raise ValueError('Author stimulus ordering/configuration differs')
            maximum = 0.
            samples = 0
            with torch.no_grad():
                for index, (stimulus, response) in enumerate(network.stimulus_response(
                        dataset, dt=.005, t_pre=1., t_fade_in=0., batch_size=1, grad=False)):
                    np.testing.assert_allclose(stimulus[0], original['stimulus'][index],
                                               rtol=0, atol=0, equal_nan=True)
                    central = response[0, :, indices].T
                    expected = original['activity'][index]
                    if central.shape != expected.shape:
                        raise ValueError('Author central response axes differ')
                    np.testing.assert_array_equal(np.isnan(central), np.isnan(expected))
                    np.testing.assert_allclose(central, expected, rtol=2e-5, atol=2e-6,
                                               equal_nan=True)
                    finite = np.isfinite(expected)
                    maximum = max(maximum, float(np.max(np.abs(central[finite]-expected[finite]))))
                    samples += 1
                    write_json(run/'progress.json', dict(mode=mode, completed=samples,
                        total=len(dataset), maximum_absolute_error=maximum))
            if samples != len(dataset):
                raise ValueError('Incomplete author response iteration')
            summaries[mode] = dict(samples=samples, original_response_sha256=original_hash,
                                  maximum_absolute_error=maximum)
            original.close()
        changed = [key for key, value in network.state_dict().items()
                   if key in state and not torch.equal(value.cpu(), state[key].cpu())]
        if changed:
            raise ValueError('Original weights changed during author forward')
        result.update(status='completed', termination_reason='all author-path comparisons passed',
                      summaries=summaries, weights_changed=[],
                      scope='Execution path parity, not a physiology gate')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run/'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result['reason'])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
