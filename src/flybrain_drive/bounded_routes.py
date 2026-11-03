"""Bounded structural path selection, never a claim of physiological causality."""

import numpy as np


def shortest_route_union(incoming, sources, targets, *, max_hops, max_nodes):
    """Keep all equally short directed routes per source/target pair within a bound.

    Incoming CSR is row=post, column=pre. Distances ignore synapse counts. Callers
    must preserve all induced structural edges afterwards, not just path edges.
    Requires the optional SciPy analysis dependency.
    """
    from scipy.sparse.csgraph import dijkstra

    sources, targets = np.asarray(sources), np.asarray(targets)
    n = incoming.shape[0]
    if (incoming.shape != (n, n) or not n or not incoming.nnz
            or not np.isfinite(incoming.data).all() or np.any(incoming.data <= 0)
            or type(max_hops) is not int or not 1 <= max_hops <= 16
            or type(max_nodes) is not int or max_nodes < 1
            or any(a.ndim != 1 or not len(a) or a.dtype.kind not in "iu"
                   or a.min() < 0 or a.max() >= n or len(np.unique(a)) != len(a)
                   for a in (sources, targets))):
        raise ValueError("Positive incoming graph, unique endpoints and finite bounds required")
    forward = dijkstra(incoming.T.tocsr(), directed=True, indices=sources,
                       unweighted=True, limit=max_hops)
    reverse = dijkstra(incoming, directed=True, indices=targets,
                       unweighted=True, limit=max_hops)
    nodes, pairs = set(), []
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            hops = forward[i, target]
            if np.isfinite(hops):
                selected = np.flatnonzero(forward[i] + reverse[j] == hops)
                nodes.update(map(int, selected))
                if len(nodes) > max_nodes:
                    raise ValueError("Route union exceeds declared node budget; no truncation")
                pairs.append({"source_index": int(source), "target_index": int(target),
                              "hops": int(hops), "route_union_nodes": len(selected)})
    return sorted(nodes), pairs
