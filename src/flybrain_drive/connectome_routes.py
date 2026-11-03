"""Measured direct body-sensory routes; no inferred functional subtype or port mapping."""

from collections import defaultdict

import numpy as np


def audit_direct_routes(raw, annotations, selectors):
    crow, col, counts, ids = (np.asarray(raw[k]) for k in ("crow", "col", "counts", "body_ids"))
    n = len(ids)
    if (not n or any(a.ndim != 1 for a in (crow, col, counts, ids)) or len(crow) != n + 1
            or crow.dtype.kind not in "iu" or col.dtype.kind not in "iu"
            or crow[0] != 0 or crow[-1] != len(col) or np.any(np.diff(crow) < 0)
            or len(col) != len(counts) or np.any(col < 0) or np.any(col >= n)
            or not np.isfinite(counts).all() or np.any(counts <= 0)
            or len(set(ids.tolist())) != n):
        raise ValueError("Invalid incoming CSR")
    if not selectors or any(s.get("status") != "Traced" for s in selectors.values()):
        raise ValueError("Explicit Traced annotation selectors are required")
    metadata = {}
    for row in annotations:
        body = int(row["bodyId"])
        if body in metadata:
            raise ValueError("Duplicate annotation body ID")
        metadata[body] = row
    groups = sorted(selectors)
    graph_ids = set(map(int, ids))
    if any(body not in graph_ids
           and any(all(row.get(k) == v for k, v in selector.items())
                   for selector in selectors.values()) for body, row in metadata.items()):
        raise ValueError("Selected annotated source is absent from recovered graph")
    source_group = np.full(n, -1, dtype=np.int16)
    if len(groups) > np.iinfo(np.int16).max:
        raise ValueError("Too many selector groups")
    for index, body in enumerate(ids):
        row = metadata.get(int(body), {})
        matched = [g for g, name in enumerate(groups)
                   if all(row.get(k) == value for k, value in selectors[name].items())]
        if len(matched) > 1:
            raise ValueError("Sensory selectors overlap")
        if matched:
            source_group[index] = matched[0]
    if any(not np.any(source_group == g) for g in range(len(groups))):
        raise ValueError("Some selectors have no graph neurons")
    edge_groups = source_group[col]
    positions = np.flatnonzero(edge_groups >= 0)
    post = np.searchsorted(crow, positions, side="right") - 1
    weights = counts[positions]
    group_for_edge = edge_groups[positions]
    prefix = np.empty(len(counts) + 1, np.float64)
    prefix[0] = 0
    np.cumsum(counts, dtype=np.float64, out=prefix[1:])
    total_input = prefix[crow[1:]] - prefix[crow[:-1]]
    reports = {}
    for g, name in enumerate(groups):
        selected = group_for_edge == g
        incoming = np.bincount(post[selected], weights=weights[selected], minlength=n)
        targets, by_class = [], defaultdict(float)
        for target in np.flatnonzero(incoming > 0):
            body = int(ids[target])
            row = metadata.get(body, {})
            by_class[str(row.get("superclass"))] += float(incoming[target])
            targets.append({
                "body_id": str(body), "type": row.get("type"), "superclass": row.get("superclass"),
                "soma_side": row.get("somaSide"), "root_side": row.get("rootSide"),
                "synapses_from_group": float(incoming[target]),
                "total_incoming_in_recovered_graph": float(total_input[target]),
                "fraction_of_target_input": float(incoming[target] / total_input[target]),
            })
        targets.sort(key=lambda r: (-r["synapses_from_group"], int(r["body_id"])))
        reports[name] = {
            "selector": selectors[name],
            "source_body_ids": [str(v) for v in ids[source_group == g]],
            "source_neurons": int(np.sum(source_group == g)), "direct_targets": len(targets),
            "edge_records": int(np.sum(selected)),
            "synapses": float(weights[selected].sum(dtype=np.float64)),
            "target_superclass_synapses": dict(by_class), "targets": targets,
        }
    edges = {"pre_body_ids": ids[col[positions]], "post_body_ids": ids[post],
             "counts": weights, "source_group": group_for_edge,
             "group_names": np.array(groups)}
    return reports, edges
