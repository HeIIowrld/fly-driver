"""Reviewed graph I/O and a directed, per-node degree preserving control."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .episode_storage import load_array, resolve_payload_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(resolve_payload_path(path, verify_content=False)).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_graph(graph):
    ids, types, roles = graph["ids"], graph["types"], graph["roles"]
    pre, post, weights = graph["pre"], graph["post"], graph["weights"]
    if any(array.ndim != 1 for array in (ids, types, roles, pre, post, weights)):
        raise ValueError("All graph arrays must be one dimensional")
    if pre.dtype.kind not in "iu" or post.dtype.kind not in "iu":
        raise ValueError("Edge indices must be integers")
    n = len(ids)
    if n == 0 or len(set(ids.tolist())) != n or len(types) != n or len(roles) != n:
        raise ValueError("Empty/inconsistent node table or duplicate IDs")
    if not set(roles.tolist()) <= {"input", "internal", "output"}:
        raise ValueError("Roles must be explicitly reviewed: input/internal/output")
    if not np.any(roles == "input") or not np.any(roles == "output"):
        raise ValueError("Both input and output ports are required")
    if len(pre) == 0 or len(pre) != len(post) or len(pre) != len(weights):
        raise ValueError("Empty/inconsistent edge table")
    if np.any(pre < 0) or np.any(post < 0) or np.any(pre >= n) or np.any(post >= n):
        raise ValueError("Edge endpoint not found")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Structural synapse counts must be positive and finite")
    if len(set(zip(pre.tolist(), post.tolist(), strict=True))) != len(pre):
        raise ValueError("Aggregate duplicate directed neuron pairs before importing")


def save_graph(path, graph, metadata):
    validate_graph(graph)
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Graph output already exists")
    if path.suffix != ".npz":
        raise ValueError("Graph output must have .npz suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **graph)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                **metadata,
                "nodes": len(graph["ids"]),
                "aggregate_edges": len(graph["pre"]),
                "graph_sha256": sha256(path),
                "biology": "structural counts; no neurotransmitter/receptor sign model",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_graph(path):
    with load_array(path, allow_pickle=False) as arrays:
        graph = {
            key: arrays[key].copy() for key in ("ids", "types", "roles", "pre", "post", "weights")
        }
    validate_graph(graph)
    return graph


def import_csv(nodes, edges):
    with Path(nodes).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {row["id"]: i for i, row in enumerate(rows)}
    with Path(edges).open(encoding="utf-8-sig", newline="") as stream:
        links = list(csv.DictReader(stream))
    graph = {
        "ids": np.array([row["id"] for row in rows]),
        "types": np.array([row["type"] for row in rows]),
        "roles": np.array([row["role"] for row in rows]),
        "pre": np.array([index[row["pre_id"]] for row in links], dtype=np.int64),
        "post": np.array([index[row["post_id"]] for row in links], dtype=np.int64),
        "weights": np.array([float(row["synapse_count"]) for row in links], dtype=np.float32),
    }
    validate_graph(graph)
    return graph


def demo_graph(seed=7):
    """Synthetic circuit for code smoke tests; not MaleCNS or FlyWire."""
    rng = np.random.default_rng(seed)
    n = 56
    edges = {(i, (i + 1) % n) for i in range(n)}
    edges.update((int(a), int(b)) for a, b in rng.integers(n, size=(350, 2)) if a != b)
    ordered = sorted(edges)
    return {
        "ids": np.array([f"synthetic_{i}" for i in range(n)]),
        "types": np.array(["sensory"] * 16 + ["internal"] * 32 + ["output"] * 8),
        "roles": np.array(["input"] * 16 + ["internal"] * 32 + ["output"] * 8),
        "pre": np.array([a for a, _ in ordered], dtype=np.int64),
        "post": np.array([b for _, b in ordered], dtype=np.int64),
        "weights": rng.integers(1, 10, len(ordered)).astype(np.float32),
    }


def rewire(graph, seed, swaps=None):
    """a->b, c->d become a->d, c->b. IDs, ports and each node's degrees stay fixed."""
    validate_graph(graph)
    result = {key: value.copy() for key, value in graph.items()}
    pre, post = result["pre"], result["post"]
    edges = set(zip(pre.tolist(), post.tolist(), strict=True))
    target = swaps if swaps is not None else len(pre) * 5
    if target < 0:
        raise ValueError("swaps must be nonnegative")
    rng = np.random.default_rng(seed)
    achieved = 0
    attempts = 0
    while achieved < target and attempts < max(100, target * 50):
        attempts += 1
        i, j = rng.integers(len(pre), size=2)
        a, b, c, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
        if a == c or b == d or a == d or c == b or (a, d) in edges or (c, b) in edges:
            continue
        edges.remove((a, b))
        edges.remove((c, d))
        edges.update(((a, d), (c, b)))
        post[i], post[j] = d, b
        achieved += 1
    return result, {
        "method": "directed_degree_preserving_edge_swap",
        "seed": seed,
        "requested_swaps": target,
        "achieved_swaps": achieved,
        "attempts": attempts,
        "self_loops": "no new self loops",
        "reachability_and_path_matching": "not enforced; measure before comparison",
    }
