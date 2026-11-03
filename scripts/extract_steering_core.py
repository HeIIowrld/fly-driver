"""Extract a literature-scoped PFL/DNa structural candidate, not a full fly brain."""

import argparse
import hashlib
import json
from pathlib import Path

from flybrain_drive.connectome_selection import extract_type_subgraph
from flybrain_drive.episode_storage import load_array
from flybrain_drive.graph import save_graph, sha256


def extract(graph_path, annotations_path, output):
    import pyarrow.feather as feather

    roles = {"PFL2": "input", "PFL3": "input", "DNa03": "internal", "DNa02": "output"}
    table = feather.read_table(annotations_path,
                                columns=["bodyId", "type", "status", "somaSide", "instance"])
    with load_array(graph_path, allow_pickle=False) as raw:
        graph, audit = extract_type_subgraph(raw, table.to_pylist(), roles)
    source_module = Path(__file__).resolve().parents[1] / "src/flybrain_drive"
    metadata = {
        "dataset": "male-cns:v1.0",
        "source_graph_sha256": sha256(graph_path),
        "source_annotations_sha256": sha256(annotations_path),
        "selection_source_sha256": sha256(source_module / "connectome_selection.py"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "structural_candidate_scope": "PFL2/PFL3 boundary to DNa02 through retained DNa03",
        "functional_reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10881397/",
        "reference_limit": (
            "Type-level steering pathways only; not specific signs, organ ports or calibration"
        ),
        "license_source": "https://male-cns.janelia.org/download/ (CC-BY)",
        "input_ports": "PFL2/PFL3 are a cut boundary; no retinal/CX upstream computation supplied",
        "outputs": "DNa02 soma sides, no established vehicle turn-sign mapping",
        "scope": "candidate circuit, not complete vision/body driving agent",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_graph(output / "steering-core.npz", graph, metadata)
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return graph, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    graph, audit = extract(args.graph, args.annotations, args.output)
    print(json.dumps({"nodes": len(graph["ids"]), "edges": len(graph["pre"]),
                       "type_pair_connections": audit["type_pair_connections"],
                       "output_minimum_hops": audit["output_minimum_hops"]}, indent=2))


if __name__ == "__main__":
    main()
