"""Extend a structural fly candidate toward thermal/load and longitudinal control.

This extracts measured edges and reports boundaries. It does not train or assign
physiological temperature/load tuning, transmitter signs or vehicle pedal outputs.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from flybrain_drive.bounded_routes import shortest_route_union
from flybrain_drive.connectome_selection import extract_id_subgraph
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import load_graph, save_graph, sha256


def matches(row, selector):
    return all(row.get(key) in value if isinstance(value, list) else row.get(key) == value
               for key, value in selector.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vehicle-circuit-extension.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    registry = json.loads(Path(config["source_registry"]).read_text(encoding="utf8"))
    run = begin_run(args.output, {"selection": config, "seed": None,
                                 "selection_algorithm": "all bounded unweighted shortest routes"})
    try:
        import pyarrow.feather as feather
        from scipy.sparse import csr_matrix

        source_hashes = {"annotations": sha256(registry["annotation_path"]),
                         "raw_graph": sha256(registry["graph_path"]),
                         "base_graph": sha256(config["base_graph"])}
        expected = {"annotations": registry["annotation_sha256"],
                    "raw_graph": registry["graph_sha256"],
                    "base_graph": config["base_graph_sha256"]}
        if source_hashes != expected:
            raise ValueError("Input source hash differs from fixed registry")
        rows = feather.read_table(registry["annotation_path"]).to_pylist()
        base = load_graph(config["base_graph"])
        with load_array(registry["graph_path"], allow_pickle=False) as archive:
            raw = {key: archive[key] for key in ("crow", "col", "counts", "body_ids")}
        ids = raw["body_ids"]
        index = {int(body): i for i, body in enumerate(ids)}
        memberships = {}
        for kind in ("source_groups", "target_groups", "internal_groups"):
            for name, selector in config[kind].items():
                if name in memberships or selector.get("status") != "Traced":
                    raise ValueError("Unique groups and explicit Traced status required")
                selected = [r for r in rows if matches(r, selector)]
                if not selected or any(int(r["bodyId"]) not in index for r in selected):
                    raise ValueError("Selected group absent from graph: " + name)
                memberships[name] = [int(r["bodyId"]) for r in selected]
        sources = sorted({b for name in config["source_groups"] for b in memberships[name]})
        targets = sorted({b for name in config["target_groups"] for b in memberships[name]})
        if len(sources) > 256 or len(targets) > 32 or set(sources) & set(targets):
            raise ValueError("Distinct bounded source/target populations required")
        incoming = csr_matrix((raw["counts"], raw["col"], raw["crow"]),
                              shape=(len(ids), len(ids)))
        print(f"analysing sources={len(sources)} targets={len(targets)}", flush=True)
        selected, pairs = shortest_route_union(
            incoming, [index[b] for b in sources], [index[b] for b in targets],
            max_hops=config["max_hops"], max_nodes=config["max_route_union_nodes"])
        base_roles = {str(body): str(role)
                      for body, role in zip(base["ids"], base["roles"], strict=True)}
        roles = base_roles.copy()
        for i in selected:
            roles.setdefault(str(ids[i]), "internal")
        for name in config["internal_groups"]:
            for body in memberships[name]:
                roles.setdefault(str(body), "internal")
        new_inputs = [b for name in config["new_input_groups"] for b in memberships[name]]
        if len(new_inputs) != len(set(new_inputs)) or set(new_inputs) & set(targets):
            raise ValueError("Disjoint scalar input and output groups required")
        for body in new_inputs:
            if str(body) in base_roles and base_roles[str(body)] != "input":
                raise ValueError("New input would replace an internal/output role")
            roles[str(body)] = "input"
        for body in targets:
            if str(body) in roles and roles[str(body)] not in ("output", "internal"):
                raise ValueError("Output would replace an input role")
            roles[str(body)] = "output"
        graph, boundary = extract_id_subgraph(raw, rows, roles)
        new_index = {str(body): i for i, body in enumerate(graph["ids"])}
        edges = {(str(graph["ids"][a]), str(graph["ids"][b])): float(w)
                 for a, b, w in zip(graph["pre"], graph["post"], graph["weights"], strict=True)}
        for a, b, w in zip(base["pre"], base["post"], base["weights"], strict=True):
            if edges.get((str(base["ids"][a]), str(base["ids"][b]))) != float(w):
                raise ValueError("A base structural edge was removed or changed")
        full_input = np.array([r["incoming_synapses_total"] for r in boundary["boundary"]])
        if (full_input.shape != (len(graph["ids"]),) or np.any(full_input < 0)):
            raise ValueError("Full incoming normalization does not match node order")
        np.save(run / "full-incoming-counts.npy", full_input, allow_pickle=False)
        pairs = [{"source_body_id": str(ids[p["source_index"]]),
                  "target_body_id": str(ids[p["target_index"]]),
                  "hops": p["hops"], "route_union_nodes": p["route_union_nodes"]} for p in pairs]
        coverage = {}
        for source in config["source_groups"]:
            for target in config["target_groups"]:
                source_ids = set(map(str, memberships[source]))
                target_ids = set(map(str, memberships[target]))
                found = [p for p in pairs if p["source_body_id"] in source_ids
                         and p["target_body_id"] in target_ids]
                coverage[source + " -> " + target] = {
                    "source_neurons": len(source_ids), "target_neurons": len(target_ids),
                    "reachable_pairs_within_hop_bound": len(found),
                    "sources_reaching_any": len({p["source_body_id"] for p in found}),
                    "targets_reached": len({p["target_body_id"] for p in found}),
                    "path_lengths": sorted({p["hops"] for p in found})}
        ports = {name: {"body_ids": list(map(str, members)),
                        "indices": [new_index[str(b)] for b in members]}
                 for name, members in memberships.items()}
        save_graph(run / "vehicle-candidate.npz", graph, {"scope": config["scope"],
                    "source_sha256": source_hashes, "selection": config})
        write_json(run / "ports.json", ports)
        write_json(run / "boundary.json", boundary)
        write_json(run / "paths.json", pairs)
        write_json(run / "annotations.json", [r for r in rows if str(r["bodyId"]) in roles])
        result = {"status": "completed", "nodes": len(graph["ids"]),
                  "edges": len(graph["pre"]), "base_nodes": len(base["ids"]),
                  "all_base_edges_preserved": True,
                  "all_selected_internal_edges_preserved": True,
                  "route_union_nodes": len(selected), "new_scalar_inputs": len(new_inputs),
                  "output_neurons": len(targets), "source_sha256": source_hashes,
                  "coverage": coverage, "graph_sha256": sha256(run / "vehicle-candidate.npz"),
                  "full_incoming_sha256": sha256(run / "full-incoming-counts.npy"),
                  "versions": {name: importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "pyarrow")},
                  "physiological_tuning_verified": False, "signed_dynamics_assigned": False,
                  "thermal_load_adapter_trained": False, "vehicle_actuator_mapping_assigned": False,
                  "driving_evaluated": False, "goal_achieved": False}
        write_json(run / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        write_json(run / "result.json", {"status": "failed", "error": type(error).__name__,
                                         "message": str(error), "goal_achieved": False})
        raise


if __name__ == "__main__":
    main()
