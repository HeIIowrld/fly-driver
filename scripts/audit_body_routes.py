"""Export every recovered outgoing edge of explicitly selected body-sensory groups."""

import argparse
import json
from pathlib import Path

from flybrain_drive.connectome_routes import audit_direct_routes
from flybrain_drive.episode_storage import load_array
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "annotations", "selectors", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    import numpy as np
    import pyarrow.feather as feather

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    selectors = json.loads(Path(args.selectors).read_text(encoding="utf8"))
    with load_array(args.graph, allow_pickle=False) as raw:
        reports, edges = audit_direct_routes(raw, feather.read_table(args.annotations).to_pylist(),
                                             selectors)
    np.savez_compressed(out / "direct-edges.npz", **edges)
    sources = {"graph": sha256(args.graph), "annotations": sha256(args.annotations),
               "selectors": sha256(args.selectors), "script": sha256(__file__)}
    import flybrain_drive.connectome_routes as implementation
    sources["implementation"] = sha256(implementation.__file__)
    result = {"scope": "direct measured structural routes, not a signed functional model",
              "source_sha256": sources, "edges_sha256": sha256(out / "direct-edges.npz"),
              "direction": "incoming CSR row=post col=pre; exported pre->post",
              "all_selected_source_outgoing_edges_in_recovered_graph_exported": True,
              "limitations": ["Traced-only graph excludes other source-data neurons",
                              "counts do not supply transmitter/receptor/dynamics",
                              "organ functional subtype and vehicle side/sign not inferred",
                              "direct target superclass is not a complete functional pathway"],
              "groups": reports}
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({name: {k: r[k] for k in ("source_neurons", "direct_targets", "synapses",
                                             "target_superclass_synapses")}
                      for name, r in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
