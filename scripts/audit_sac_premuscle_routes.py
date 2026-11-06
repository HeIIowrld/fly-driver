"""Recorded operating-point and signed-graph audit, not a physiology certificate."""
import argparse
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.episode_storage import load_array
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import assemble_student, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    root = Path(a.trace)
    source = read(root/'result.json')
    if source['status'] != 'completed_premuscle_state_trace':
        raise ValueError('Completed exact neural trace required')
    for name, expected in source['files_sha256'].items():
        if sha256(root/name) != expected:
            raise ValueError('Original neural trace changed')
    groups = read(root/'groups.json')
    model, _ = assemble_student()
    b = model.motor.backbone
    matrix = b.circuit.matrix.to_sparse_coo().coalesce()
    row, col = matrix.indices().numpy()
    value = matrix.values().numpy()
    nonzero = value != 0
    row, col, value = row[nonzero], col[nonzero], value[nonzero]
    adjacency = [[] for _ in range(model.motor.neurons)]
    for pre, post in zip(col, row):
        adjacency[int(pre)].append(int(post))
    paths = {}
    for name in ['vision', 'body', 'steering']:
        distance = np.full(model.motor.neurons, -1)
        distance[groups[name]] = 0
        queue = deque(groups[name])
        while queue:
            pre = queue.popleft()
            for post in adjacency[pre]:
                if distance[post] < 0:
                    distance[post] = distance[pre]+1
                    queue.append(post)
        paths[name] = dict(to_steering=distance[groups['steering']].tolist(),
            to_motor=distance[groups['motor']].tolist())
    with load_array(b.spec['graph_file'], allow_pickle=False) as z:
        ids = z['ids'].astype(str).tolist()
    annotations = {str(r['bodyId']): r for r in read(b.spec['annotations_file'])}
    used_ids = [str(body) for group in b.spec['motor_groups'].values() for side in group.values() for body in side]
    used = [ids.index(body) for body in used_ids]
    run = begin_run(a.output, dict(arguments=vars(a), seed=780407, program_sha256=sha256(__file__),
        trace_result_sha256=sha256(root/'result.json'), original_graph_sha256=b.spec['graph_sha256'],
        scope='Rate-model operating point and structural reachability only; no spikes/physiology or new measured synapse claim',
        optimizer_updates=0, vehicle_control_sent=False))
    reports = []
    for i in range(3):
        states = np.load(root/f'state-{i}.npy', mmap_mode='r', allow_pickle=False)
        dn = states[:, groups['steering']].astype(np.float64)
        nets = []
        for node in groups['steering']:
            keep = row == node
            nets.append((states[:, col[keep]].astype(np.float64)*value[keep]).sum(1))
        nets = np.column_stack(nets)
        reports.append(dict(episode=i, dn_min=float(dn.min()), dn_max=float(dn.max()),
            dn_std=dn.std(0).tolist(), dn_fraction_above_1e_8=float((dn>1e-8).mean()),
            recurrent_input_quantiles=np.quantile(nets, [0, .05, .5, .95, 1], axis=0).tolist()))
    write_json(run/'groups-identified.json', dict(used_wing_motor_indices=used,
        steering=[annotations[ids[i]] for i in groups['steering']],
        visual=[annotations[ids[i]] for i in groups['vision']],
        output_types=dict(Counter(annotations[ids[i]].get('type') for i in groups['motor']))))
    write_json(run/'result.json', dict(status='completed_premuscle_route_audit', paths=paths,
        dn_incoming_nonzero_counts=[int((row==n).sum()) for n in groups['steering']],
        operating_point=reports, raw_motor_states=len(groups['motor']), used_wing_motor_states=len(used),
        source_command='sensory_decision requests desired yaw0 and measured speed; therefore external steering drive0',
        interpretation='DNa04/05 rates are numerically at floor under these inputs despite reachable paths; this is not an anatomical disconnection or real-neuron silence claim',
        optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False,
        termination_reason='Exact stored states and fixed signed graph audited'))


if __name__ == '__main__':
    main()
