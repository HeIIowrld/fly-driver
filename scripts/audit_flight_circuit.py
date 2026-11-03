"""Audit actual prototype matrices and identical adapter initialization across sign variants."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--spec', default='configs/flight-circuit-prototype-20261029.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.probe)
    spec = json.loads(Path(args.spec).read_text(encoding='utf8'))
    receipt = json.loads((root/'result.json').read_text(encoding='utf8'))
    if receipt['status'] != 'completed_untrained_technical_probe':
        raise ValueError('Completed untrained probe required')
    run = begin_run(args.output, dict(arguments=vars(args), seed=None,
        probe_result_sha256=sha256(root/'result.json'), spec_sha256=sha256(args.spec),
        actuator_messages_sent=0))
    if sha256(spec['graph_file']) != spec['graph_sha256']:
        raise ValueError('Measured topology changed')
    with load_array(spec['graph_file'], allow_pickle=False) as archive:
        graph = {key: archive[key] for key in archive.files}
    boundary = json.loads(Path(spec['boundary_file']).read_text(encoding='utf8'))['boundary']
    totals = np.array([row['incoming_synapses_total'] for row in boundary])
    nt = {str(row['body']): row['consensus_nt'] for row in json.loads(
        Path(spec['transmitter_file']).read_text(encoding='utf8'))}
    order = np.lexsort((graph['pre'], graph['post']))
    first_parameters, results = None, []
    for sign in (-1, 0, 1):
        label = f'glutamate-{sign:+d}'
        checkpoint = root/(label+'-untrained.pt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        mapping = {'acetylcholine': 1, 'gaba': -1, 'glutamate': sign}
        signs = np.array([mapping.get(nt[body], 0) for body in graph['ids']])
        expected = (graph['weights']*signs[graph['pre']]
                    /np.maximum(totals[graph['post']], 1)).astype(np.float32)
        matrix = state['circuit.matrix'].coalesce()
        np.testing.assert_array_equal(matrix.indices().numpy(),
            np.stack([graph['post'][order], graph['pre'][order]]))
        np.testing.assert_array_equal(matrix.values().numpy(), expected[order])
        parameters = {name: value for name, value in state.items()
                      if name.startswith(('vision.', 'body.', 'goal.', 'context.', 'readout.'))}
        if first_parameters is None:
            first_parameters = parameters
        elif any(not torch.equal(value, first_parameters[name])
                 for name, value in parameters.items()):
            raise ValueError('Sign variants changed their adapter/readout initialization')
        with load_array(root/(label+'-responses.npz'), allow_pickle=False) as response:
            muscles = response['muscles']
            cases = response['cases'].tolist()
            time = float(response['dt'])
        norms = np.linalg.norm(muscles-muscles[:, 0:1], axis=2)
        latencies = {}
        for name in ['visual_yaw_positive', 'body_yaw_positive', 'goal_forward_positive',
                     'spatial_scene_change']:
            index = cases.index(name)
            pulse = norms[50:100, index]
            peak = float(pulse.max())
            first = int(np.flatnonzero(pulse >= .1*peak)[0])
            latencies[name] = dict(peak_motor_delta_norm=peak,
                                   ten_percent_onset_seconds=(first+1)*time)
        results.append(dict(glutamate_sign=sign, checkpoint_sha256=sha256(checkpoint),
            exact_original_edge_indices=True, exact_declared_signed_normalized_counts=True,
            all_structural_edges_preserved=matrix._nnz(),
            nonzero_effective_edges=int(np.count_nonzero(expected)), pulse_metrics=latencies))
    write_json(run/'result.json', dict(status='completed',
        termination_reason='all three checkpoint matrices and controlled initializations verified',
        variants=results, adapters_identical_across_sign_hypotheses=True,
        scene_path_preserved=True, untrained_model=True,
        physiological_latency_validated=False, balance_reflex_validated=False,
        actual_actor_integration=False, actuator_messages_sent=0, goal_achieved=False))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
