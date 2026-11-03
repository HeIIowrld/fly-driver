"""Explicit type-selected structural subgraphs; no automatic anatomical port inference."""

from collections import deque

import numpy as np

from .graph import validate_graph


def extract_type_subgraph(raw, annotations, type_roles):
    """Select exact Traced types and preserve all measured internal links."""
    return _extract_subgraph(raw, annotations, type_roles, "type")


def extract_id_subgraph(raw, annotations, id_roles):
    """Select exact IDs so an ambiguous/shared type cannot add unrelated organs."""
    normalized = {str(int(body)): role for body, role in id_roles.items()}
    if len(normalized) != len(id_roles):
        raise ValueError("Duplicate normalized body IDs")
    return _extract_subgraph(raw, annotations, normalized, "bodyId")


def _extract_subgraph(raw, annotations, declared_roles, selection_field):
    """Read incoming CSR (row=post, col=pre), preserve every internal selected edge.

    The caller declares a scoped type/role hypothesis. This does not establish that
    the selected boundary is a complete visual/organ circuit or a driving controller.
    """
    if (not declared_roles or set(declared_roles.values()) - {"input", "internal", "output"}
            or not {"input", "output"} <= set(declared_roles.values())):
        raise ValueError("Explicit input/internal/output type roles are required")
    crow, col, counts, body_ids = (np.asarray(raw[k]) for k in
                                  ("crow", "col", "counts", "body_ids"))
    n = len(body_ids)
    if (any(a.ndim != 1 for a in (crow, col, counts, body_ids)) or len(crow) != n+1
            or col.dtype.kind not in "iu" or crow.dtype.kind not in "iu"
            or len(col) != len(counts) or crow[0] != 0 or crow[-1] != len(col)
            or np.any(np.diff(crow) < 0) or not len(col) or col.min() < 0 or col.max() >= n
            or not np.isfinite(counts).all() or np.any(counts <= 0)
            or len(set(body_ids.tolist())) != n):
        raise ValueError("Invalid incoming-CSR structural graph")
    global_index = {int(body): i for i, body in enumerate(body_ids)}
    selected = {}
    def selection_key(row):
        return str(int(row["bodyId"])) if selection_field == "bodyId" else row.get("type")

    for row in annotations:
        if selection_key(row) not in declared_roles or row.get("status") != "Traced":
            continue
        body = int(row["bodyId"])
        if body not in global_index:
            raise ValueError("Selected annotated neuron is absent from the source graph")
        if body in selected:
            raise ValueError("Duplicate selected annotation")
        selected[body] = row
    found_keys = {selection_key(row) for row in selected.values()}
    if found_keys != set(declared_roles):
        raise ValueError("Some declared selectors have no selected Traced neurons: "
                         + str(set(declared_roles) - found_keys))
    ordered = sorted(selected)
    local_index = {body: i for i, body in enumerate(ordered)}
    remap = np.full(n, -1, dtype=np.int64)
    for body in ordered:
        remap[global_index[body]] = local_index[body]
    pre, post, weights, boundary = [], [], [], []
    all_outgoing = np.bincount(col, weights=counts, minlength=n)
    for body in ordered:
        g = global_index[body]
        incoming = slice(int(crow[g]), int(crow[g+1]))
        local_pre = remap[col[incoming]]
        keep = local_pre >= 0
        pre.extend(local_pre[keep].tolist())
        post.extend([local_index[body]] * int(keep.sum()))
        weights.extend(counts[incoming][keep].tolist())
        total = float(counts[incoming].sum(dtype=np.float64))
        retained = float(counts[incoming][keep].sum(dtype=np.float64))
        boundary.append({"body_id": str(body), "type": selected[body]["type"],
                         "soma_side": selected[body].get("somaSide"),
                         "incoming_synapses_total": total,
                         "incoming_synapses_retained": retained,
                         "incoming_fraction_retained": retained / total if total else None,
                         "outgoing_synapses_total": float(all_outgoing[g])})
    graph = {
        "ids": np.array([str(body) for body in ordered]),
        "types": np.array([selected[body].get("type") or f"untyped:{body}" for body in ordered]),
        "roles": np.array([declared_roles[selection_key(selected[body])] for body in ordered]),
        "pre": np.array(pre, dtype=np.int64), "post": np.array(post, dtype=np.int64),
        "weights": np.array(weights, dtype=np.float32),
    }
    validate_graph(graph)
    adjacency = [[] for _ in ordered]
    for a, b in zip(pre, post, strict=True):
        adjacency[a].append(b)
    distances = {int(i): 0 for i in np.flatnonzero(graph["roles"] == "input")}
    queue = deque(distances)
    while queue:
        source = queue.popleft()
        for target in adjacency[source]:
            if target not in distances:
                distances[target] = distances[source] + 1
                queue.append(target)
    unreachable = [str(graph["ids"][i]) for i in np.flatnonzero(graph["roles"] == "output")
                   if i not in distances]
    if unreachable:
        raise ValueError("Selected outputs have no input path: " + str(unreachable))
    pair_counts = {}
    outgoing_retained = np.bincount(graph["pre"], weights=graph["weights"], minlength=len(ordered))
    for a, b, weight in zip(pre, post, weights, strict=True):
        pair = graph["types"][a] + " -> " + graph["types"][b]
        row = pair_counts.setdefault(pair, {"neuron_pairs": 0, "synapses": 0})
        row["neuron_pairs"] += 1
        row["synapses"] += weight
    for i, row in enumerate(boundary):
        row["outgoing_synapses_retained"] = float(outgoing_retained[i])
        total = row["outgoing_synapses_total"]
        row["outgoing_fraction_retained"] = float(outgoing_retained[i] / total) if total else None
    return graph, {
        "scope": "explicit structural candidate; not a complete sensorimotor model",
        ("type_roles" if selection_field == "type" else "id_roles"): dict(declared_roles),
        "selection": "exact " + selection_field + " and status Traced",
        "untyped_body_ids": [str(body) for body in ordered if not selected[body].get("type")],
        "untyped_label_rule": "untyped:<body_id> is an engineering label, not a measured cell type",
        "direction": "source incoming CSR row=post, col=pre; exported pre->post",
        "all_internal_selected_edges_preserved": True,
        "boundary": boundary, "type_pair_connections": pair_counts,
        "output_minimum_hops": {str(graph["ids"][i]): distances[int(i)]
                                for i in np.flatnonzero(graph["roles"] == "output")},
        "input_mapping": "cut-boundary learned adapter required, no retinal/organ mapping supplied",
        "side_mapping": "somaSide reported; vehicle turn sign is not established",
        "biophysics": "structural counts only; signs and dynamics require a separate model",
    }
