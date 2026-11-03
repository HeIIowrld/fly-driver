"""Build an exact-ID body/steering structural candidate, with separate unimplemented ports."""

import argparse
import json
from pathlib import Path

from flybrain_drive.connectome_selection import extract_id_subgraph
from flybrain_drive.episode_storage import load_array
from flybrain_drive.graph import save_graph, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "annotations", "routes", "paths", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    import pyarrow.feather as feather

    routes = json.loads(Path(args.routes).read_text(encoding="utf8"))
    paths = json.loads(Path(args.paths).read_text(encoding="utf8"))
    assert sha256(args.routes) == paths["source_sha256"]["routes"]
    assert sha256(args.graph) == routes["source_sha256"]["graph"]
    assert sha256(args.annotations) == routes["source_sha256"]["annotations"]
    rows = feather.read_table(args.annotations).to_pylist()
    roles, visual_boundary = {}, []
    for row in rows:
        if row.get("status") == "Traced" and row.get("type") in ("PFL2", "PFL3", "DNa02", "DNa03"):
            body = str(row["bodyId"])
            role = "output" if row["type"] == "DNa02" else "internal"
            if row["type"] in ("PFL2", "PFL3"):
                role = "input"
                visual_boundary.append(body)
            roles[body] = role
    groups = {name: group["source_body_ids"] for name, group in routes["groups"].items()}
    for members in groups.values():
        for body in members:
            if body in roles:
                raise ValueError("Organ group overlaps declared steering types")
            roles[body] = "input"
    for route in paths["routes"]:
        for path in route["paths"]:
            roles.setdefault(path["intermediate_body_id"], "internal")
    with load_array(args.graph, allow_pickle=False) as raw:
        graph, audit = extract_id_subgraph(raw, rows, roles)
    import flybrain_drive.connectome_selection as selection
    metadata = {
        "scope": "body-afferent and two-hop steering structural candidate; not full fly controller",
        "source_sha256": {key: sha256(getattr(args, key)) for key in
                          ("graph", "annotations", "routes", "paths")},
        "selection_sha256": sha256(selection.__file__), "script_sha256": sha256(__file__),
        "selection": {"organ_inputs": sum(len(ids) for ids in groups.values()),
                      "steering_types": ["PFL2", "PFL3", "DNa02", "DNa03"],
                      "relays": "all measured two-hop intermediates in supplied path audit"},
        "not_policy_ready": True,
        "limitations": ["PFL is a cut visual/CX boundary, upstream visual circuit absent",
                       "organ tuning/subtype and vehicle transduction are not assigned",
                       "separate visual/body adapters absent; fused injection is prohibited",
                       "signs, receptors, time constants and vehicle output sign unverified"],
        "license_source": "configs/malecns-recovered-source.json; MaleCNS CC-BY",
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    save_graph(out / "body-steering.npz", graph, metadata)
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
    ports = {"status": "structural port groups only, functional adapters not implemented",
             "visual_CX_cut_boundary_ids": sorted(visual_boundary, key=int),
             "organ_input_groups": groups,
             "output_ids": graph["ids"][graph["roles"] == "output"].tolist(),
             "prohibited": "Do not send a fused visual/body embedding into every input port"}
    (out / "ports.json").write_text(json.dumps(ports, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"nodes": len(graph["ids"]), "edges": len(graph["pre"]),
                      "inputs": len(graph["ids"][graph["roles"] == "input"]),
                      "outputs": audit["output_minimum_hops"]}))


if __name__ == "__main__":
    main()
