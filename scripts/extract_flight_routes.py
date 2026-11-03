"""Extract bounded measured flight-route candidates, preserving cut boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy.sparse import csr_matrix

from flybrain_drive.bounded_routes import shortest_route_union
from flybrain_drive.connectome_selection import extract_id_subgraph
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fly-flight-route-extraction-20261029.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.config).read_text(encoding="utf8"))
    registry = json.loads(Path(plan["graph_registry"]).read_text(encoding="utf8"))
    inventory = json.loads(Path(plan["inventory"]).read_text(encoding="utf8"))
    graph_path, annotation_path = Path(registry["graph_path"]), Path(registry["annotation_path"])
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            plan=plan,
            seed=None,
            config_sha256=sha256(args.config),
            registry_sha256=sha256(plan["graph_registry"]),
            inventory_sha256=sha256(plan["inventory"]),
            actuator_messages_sent=0,
        ),
    )
    result = dict(
        status="failed",
        actuator_messages_sent=0,
        functional_validation=False,
        vehicle_mapping_validated=False,
        goal_achieved=False,
    )
    try:
        if sha256(graph_path) != registry["graph_sha256"]:
            raise ValueError("Measured graph hash mismatch")
        if sha256(annotation_path) != registry["annotation_sha256"]:
            raise ValueError("Annotation hash mismatch")
        columns = [
            "bodyId",
            "type",
            "instance",
            "status",
            "somaSide",
            "rootSide",
            "superclass",
            "class",
            "subclass",
            "entryNerve",
            "exitNerve",
            "somaNeuromere",
            "receptorType",
        ]
        annotations = feather.read_table(annotation_path, columns=columns).to_pylist()
        annotated = {int(row["bodyId"]): row for row in annotations}
        with load_array(graph_path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in ("crow", "col", "counts", "body_ids")}
        bodies = raw["body_ids"]
        lookup = {int(body): index for index, body in enumerate(bodies)}
        incoming = csr_matrix(
            (raw["counts"], raw["col"], raw["crow"]), shape=(len(bodies), len(bodies))
        )

        def group_ids(groups):
            return sorted(
                {int(row["bodyId"]) for group in groups for row in inventory["candidates"][group]}
            )

        selected, endpoint_roles, paths = set(), {}, []
        for path_index, declaration in enumerate(plan["paths"]):
            source_ids = group_ids(declaration["source_groups"])
            target_ids = group_ids(declaration["target_groups"])
            for body in source_ids + target_ids:
                if body not in lookup or annotated[body]["status"] != "Traced":
                    raise ValueError("Declared endpoint is absent or not Traced")
            nodes, pairs = shortest_route_union(
                incoming,
                np.array([lookup[body] for body in source_ids], dtype=np.int64),
                np.array([lookup[body] for body in target_ids], dtype=np.int64),
                max_hops=plan["max_hops"],
                max_nodes=plan["max_nodes"],
            )
            selected.update(nodes)
            selected.update(lookup[body] for body in source_ids + target_ids)
            if len(selected) > plan["max_nodes"]:
                raise ValueError("Combined graph exceeds budget; no truncation")
            reached_pairs = {
                (int(bodies[pair["source_index"]]), int(bodies[pair["target_index"]]))
                for pair in pairs
            }
            for pair in pairs:
                pair["source_body_id"] = str(bodies[pair["source_index"]])
                pair["target_body_id"] = str(bodies[pair["target_index"]])
            missing_pairs = [
                dict(source_body_id=str(source), target_body_id=str(target))
                for source in source_ids
                for target in target_ids
                if (source, target) not in reached_pairs
            ]
            paths.append(
                dict(
                    declaration=declaration,
                    source_count=len(source_ids),
                    target_count=len(target_ids),
                    reached_pairs=pairs,
                    missing_pairs=missing_pairs,
                    nodes_on_shortest_routes=len(nodes),
                )
            )
            write_json(
                run / "progress.json",
                dict(completed_path_groups=path_index + 1, selected_nodes=len(selected)),
            )
        # Endpoint roles are structural I/O declarations, not physiological signs.
        for group in ("HS", "VS", "haltere_annotation_candidates"):
            for body in group_ids([group]):
                endpoint_roles[body] = "input"
        for body in group_ids(["wing_motor_annotation_candidates"]):
            endpoint_roles[body] = "output"
        roles = {
            str(int(bodies[index])): endpoint_roles.get(int(bodies[index]), "internal")
            for index in sorted(selected)
        }
        write_json(run / "paths.json", paths)
        write_json(
            run / "selected-annotations.json",
            [annotated[int(bodies[index])] for index in sorted(selected)],
        )
        graph, boundary = extract_id_subgraph(raw, annotations, roles)
        np.savez_compressed(run / "graph.npz", **graph)
        write_json(run / "boundary.json", boundary)
        side_summary = {}
        for group in ("DNg02", "wing_motor_annotation_candidates", "haltere_annotation_candidates"):
            rows = [annotated[body] for body in group_ids([group])]
            side_summary[group] = {
                field: {
                    str(side): sum(row[field] == side for row in rows)
                    for side in sorted({row[field] for row in rows}, key=str)
                }
                for field in ("somaSide", "rootSide")
            }
        result.update(
            status="completed_structural_candidate",
            termination_reason="all induced measured edges exported with cut-boundary audit",
            nodes=len(graph["ids"]),
            edges=len(graph["pre"]),
            graph_sha256=sha256(run / "graph.npz"),
            sides=side_summary,
            path_summaries=[
                dict(
                    declaration=path["declaration"],
                    reached_pairs=len(path["reached_pairs"]),
                    missing_pairs=len(path["missing_pairs"]),
                    nodes_on_shortest_routes=path["nodes_on_shortest_routes"],
                )
                for path in paths
            ],
            graph_source_sha256=registry["graph_sha256"],
            annotation_sha256=registry["annotation_sha256"],
            signs="unsigned counts only; neurotransmitter and functional signs unknown",
            missing_receptor_subtypes=("haltere receptorType absent; "
                                       "no pitch/roll/yaw ports inferred"),
        )
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    write_json(run / "result.json", result)
    if result["status"] == "failed":
        raise RuntimeError(result["reason"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
