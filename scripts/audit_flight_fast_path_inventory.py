"""Inventory measured chemical haltere-to-motor routes; never infer electrical edges or identity."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    for key in ('graph', 'annotations', 'boundary'):
        assert sha256(spec[key+'_file']) == spec[key+'_sha256']
    with load_array(spec['graph_file'], allow_pickle=False) as data:
        ids, pre, post, weights = (data[k] for k in ('ids', 'pre', 'post', 'weights'))
    annotations = {str(row['bodyId']): row for row in read(spec['annotations_file'])}
    boundary = read(spec['boundary_file'])['boundary']
    assert [str(row['body_id']) for row in boundary] == ids.tolist()
    index = {str(body): n for n, body in enumerate(ids)}
    afferents = np.array([index[body] for body in spec['body_input_ids']])
    for body in spec['body_input_ids']:
        assert annotations[body]['subclass'] == 'haltere'
    target_ids = sorted({body for group in spec['motor_groups'].values()
                         for side in ('L', 'R') for body in group[side]})
    target_indices = np.array([index[body] for body in target_ids])
    adjacency = coo_matrix((np.ones(len(pre)), (pre, post)), shape=(len(ids), len(ids))).tocsr()
    distances = dijkstra(adjacency, directed=True, indices=afferents,
                         unweighted=True, limit=3)[:, target_indices]
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        spec_sha256=sha256(args.spec), graph_sha256=spec['graph_sha256'],
        annotations_sha256=spec['annotations_sha256'],
        boundary_sha256=spec['boundary_sha256']))
    outputs = []
    source_mask = np.isin(pre, afferents)
    for column, body in enumerate(target_ids):
        selected = source_mask & (post == index[body])
        edges = [dict(pre_id=str(ids[p]), post_id=body, synapses=float(weight),
                      pre_type=annotations[str(ids[p])]['type'],
                      pre_root_side=annotations[str(ids[p])]['rootSide'])
                 for p, weight in zip(pre[selected], weights[selected], strict=True)]
        total = float(boundary[index[body]]['incoming_synapses_total'])
        counts = Counter(str(int(value)) if np.isfinite(value) else 'beyond_3_or_absent'
                         for value in distances[:, column])
        outputs.append(dict(body_id=body, type=annotations[body]['type'],
            soma_side=annotations[body]['somaSide'], direct_afferent_neurons=len(edges),
            direct_chemical_synapses=float(weights[selected].sum()),
            full_source_incoming_synapses=total,
            selected_afferent_fraction_of_total=float(weights[selected].sum())/max(total, 1),
            chemical_shortest_hops_from_each_body_port=dict(counts), direct_edges=edges))
    write_json(run/'result.json', dict(status='chemical_fast_route_inventory_completed',
        goal_achieved=False, actuator_messages_sent=0, optimizer_updates=0,
        body_ports=len(afferents), motor_neurons=len(outputs), targets=outputs,
        electrical_connections_measured=False, electrical_conductances_inferred=False,
        physiology_verified=False, fanc_identity_mapping_verified=False, model_changed=False))


if __name__ == '__main__':
    main()
