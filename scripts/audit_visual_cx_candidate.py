"""Extend an explicit body/steering candidate with reviewed visual/CX type lists."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

from flybrain_drive.connectome_selection import extract_id_subgraph
from flybrain_drive.episode_storage import load_array
from flybrain_drive.graph import load_graph, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "annotations", "base", "groups", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    import pyarrow.feather as feather

    selection = json.loads(Path(args.groups).read_text(encoding="utf8"))
    groups = selection["groups"]
    declared = {}
    for name, group in groups.items():
        if group["role"] not in ("input", "internal") or not group["types"]:
            raise ValueError("Explicit nonempty input/internal type lists required")
        for kind in group["types"]:
            if kind in declared:
                raise ValueError("Overlapping type groups")
            declared[kind] = name
    annotations = feather.read_table(args.annotations).to_pylist()
    by_id = {int(row["bodyId"]): row for row in annotations}
    base = load_graph(args.base)
    roles = {str(body): str(role) for body, role in zip(base["ids"], base["roles"], strict=True)}
    # Old PFL visual-boundary injection is removed, rather than retained as a shortcut.
    replaced = set()
    for body in roles:
        if by_id[int(body)].get("type") in selection["replace_boundary_types"]:
            if roles[body] != "input":
                raise ValueError("Expected the declared old boundary to be an input")
            roles[body] = "internal"
            replaced.add(by_id[int(body)]["type"])
    if not replaced or replaced != set(selection["replace_boundary_types"]):
        raise ValueError("Every declared old boundary type must be replaced")
    found = set()
    for row in annotations:
        kind = row.get("type")
        if row.get("status") == "Traced" and kind in declared:
            found.add(kind)
            body, role = str(row["bodyId"]), groups[declared[kind]]["role"]
            if body in roles and roles[body] != role:
                raise ValueError("New type selection conflicts with base role")
            roles[body] = role
    if found != set(declared):
        raise ValueError("Some exact types have no Traced neurons")
    with load_array(args.graph, allow_pickle=False) as raw:
        graph, audit = extract_id_subgraph(raw, annotations, roles)
    membership = [declared.get(str(t), "base_body_steering") for t in graph["types"]]
    pair_counts = {}
    adjacency = [[] for _ in graph["ids"]]
    for pre, post, count in zip(graph["pre"], graph["post"], graph["weights"], strict=True):
        adjacency[pre].append(int(post))
        pair = membership[pre] + " -> " + membership[post]
        row = pair_counts.setdefault(pair, {"neuron_pairs": 0, "synapses": 0.})
        row["neuron_pairs"] += 1
        row["synapses"] += float(count)
    # Reverse BFS gives every source's shortest actual directed path to an output.
    reverse = [[] for _ in graph["ids"]]
    for pre, targets in enumerate(adjacency):
        for post in targets:
            reverse[post].append(pre)
    distance = {int(i): 0 for i in np.flatnonzero(graph["roles"] == "output")}
    queue = deque(distance)
    while queue:
        post = queue.popleft()
        for pre in reverse[post]:
            if pre not in distance:
                distance[pre] = distance[post] + 1
                queue.append(pre)
    chain = []
    stages = selection.get("ordered_path_stages", [])
    allowed_next = None
    for stage in reversed(stages):
        indices = set(map(int, np.flatnonzero(np.isin(graph["types"], stage["types"]))))
        if not indices:
            raise ValueError("Ordered path stage has no selected neurons")
        allowed = (indices if allowed_next is None else
                   {i for i in indices if any(post in allowed_next for post in adjacency[i])})
        chain.append({"stage": stage["name"], "neurons": len(indices),
                      "reaches_output_via_every_later_stage": len(allowed)})
        allowed_next = allowed
    if stages and not np.isin(graph["types"][graph["roles"] == "output"],
                              stages[-1]["types"]).all():
        raise ValueError("Ordered path must end at every declared output")
    boundary = {row["body_id"]: row for row in audit["boundary"]}
    summary = {}
    for name in [*groups, "base_body_steering"]:
        indices = [i for i, group in enumerate(membership) if group == name]
        rows = [boundary[str(graph["ids"][i])] for i in indices]
        total = sum(r["incoming_synapses_total"] for r in rows)
        summary[name] = {
            "neurons": len(indices), "reaches_output": sum(i in distance for i in indices),
            "incoming_fraction_retained": (sum(r["incoming_synapses_retained"] for r in rows)
                                            / total if total else None),
            "path_lengths_to_output": sorted({distance[i] for i in indices if i in distance}),
        }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "visual-cx-body.npz", **graph)
    metadata = [{k: by_id[int(body)].get(k) for k in (
        "bodyId", "type", "instance", "somaSide", "rootSide", "assignedOlHex1", "assignedOlHex2")}
        for body in graph["ids"]]
    sources = {name: sha256(getattr(args, name)) for name in
               ("graph", "annotations", "base", "groups")}
    sources.update(script=sha256(__file__))
    import flybrain_drive.connectome_selection as implementation
    sources["extraction_implementation"] = sha256(implementation.__file__)
    result = {"scope": "measured structural candidate; no functional/retinal registration claim",
              "source_sha256": sources, "graph_sha256": sha256(out / "visual-cx-body.npz"),
              "nodes": len(graph["ids"]), "edge_records": len(graph["pre"]),
              "groups": summary, "group_pairs": pair_counts,
              "ordered_direct_chain": list(reversed(chain)),
              "old_PFL_input_injection_removed": True,
              "retinal_to_neuron_registration": "unassigned",
              "synaptic_signs_and_dynamics": "unassigned",
              "physiological_compass_and_goal_computation_tested": False}
    for name, value in (("audit", audit), ("annotations", metadata), ("summary", result)):
        (out / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: result[k] for k in ("nodes", "edge_records", "groups")}, indent=2))


if __name__ == "__main__":
    main()
