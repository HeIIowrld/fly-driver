"""Directed graph reachability among visual, descending and motor candidates."""

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import load_graph, sha256


def distances(n, pre, post, starts, blocked=None):
    blocked = np.zeros(n, dtype=bool) if blocked is None else blocked
    adjacency = [[] for _ in range(n)]
    for source, target in zip(pre, post, strict=True):
        if not blocked[source] and not blocked[target]:
            adjacency[source].append(target)
    dist = np.full(n, -1, dtype=int)
    queue = deque()
    for node in starts:
        if not blocked[node]:
            dist[node] = 0
            queue.append(node)
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            if dist[target] < 0:
                dist[target] = dist[source]+1
                queue.append(target)
    return dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    read_spec_hash = sha256(args.spec)
    spec = json.loads(Path(args.spec).read_text(encoding='utf8'))
    graph_file = spec['graph_file']
    if sha256(graph_file) != spec['graph_sha256']:
        raise ValueError('Original measured-graph artifact changed')
    graph = load_graph(graph_file)
    n, pre, post, weights = len(graph['ids']), graph['pre'], graph['post'], graph['weights']
    types = graph['types']
    groups = {'HS': np.char.startswith(types, 'HS'), 'VS': np.char.startswith(types, 'VS'),
              'DNg02': np.char.startswith(types, 'DNg02'), 'output': graph['roles'] == 'output'}
    if [int(x.sum()) for x in groups.values()] != [8, 34, 29, 67]:
        raise ValueError('Expected original candidate graph populations required')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'spec_sha256': sha256(args.spec), 'graph_sha256': spec['graph_sha256'],
        'edges_are_structural_not_functional_effects': True, 'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports = {}
    for name in ('HS', 'VS', 'DNg02'):
        starts = np.flatnonzero(groups[name])
        distance = distances(n, pre, post, starts)
        reports[name] = {'nodes': len(starts), 'targets': {}}
        for target in ('DNg02', 'output'):
            ids = np.flatnonzero(groups[target])
            edges = groups[name][pre] & groups[target][post]
            histogram = Counter(int(x) for x in distance[ids] if x >= 0)
            reports[name]['targets'][target] = {
                'direct_edge_pairs': int(edges.sum()),
                'direct_structural_synapses': float(weights[edges].sum()),
                'reachable_targets': int((distance[ids] >= 0).sum()),
                'shortest_hops_histogram': dict(sorted(histogram.items()))}
    intermediates = {}
    for source_name in ('HS', 'VS'):
        incoming_edges = groups[source_name][pre]
        incoming = np.bincount(post[incoming_edges], minlength=n)
        for target_name in ('DNg02', 'output'):
            outgoing_edges = groups[target_name][post]
            outgoing = np.bincount(pre[outgoing_edges], minlength=n)
            selected = np.flatnonzero((incoming > 0) & (outgoing > 0))
            by_type = {}
            for node in selected:
                label = str(types[node]) or '<untyped>'
                value = by_type.setdefault(label, {'nodes': 0, 'two_edge_routes': 0})
                value['nodes'] += 1
                value['two_edge_routes'] += int(incoming[node]*outgoing[node])
            intermediates[source_name+'->'+target_name] = {
                'nodes': len(selected), 'two_edge_routes': int((incoming*outgoing).sum()),
                'types': dict(sorted(by_type.items(),
                    key=lambda item: item[1]['two_edge_routes'], reverse=True))}
    for path in (args.spec, graph_file):
        if sha256(path) != (spec['graph_sha256'] if path == graph_file
                            else read_spec_hash):
            raise ValueError('Original source changed during graph analysis')
    result = {'status': 'completed_visual_flight_structural_path_audit',
              'reports': reports, 'two_hop_intermediates': intermediates,
              'graph_nodes': n, 'graph_edges': len(pre),
              'reachability_is_not_physiological_or_causal_proof': True,
              'measured_synapse_counts_do_not_establish_functional_sign': True,
              'optimizer_updates': 0, 'actor_deployed': False, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
