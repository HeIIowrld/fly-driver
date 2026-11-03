"""List measured two-edge body-group paths to exact DNa02/DNa03 types, without sign claims."""

import argparse
import json
from pathlib import Path

from flybrain_drive.episode_storage import load_array
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "annotations", "routes", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    import pyarrow.feather as feather

    route = json.loads(Path(args.routes).read_text(encoding="utf8"))
    hashes = {k: sha256(getattr(args, k)) for k in ("graph", "annotations", "routes")}
    assert hashes["graph"] == route["source_sha256"]["graph"], "Different source graph"
    assert hashes["annotations"] == route["source_sha256"]["annotations"], "Different annotations"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    annotations = feather.read_table(args.annotations).to_pylist()
    targets = [r for r in annotations if r.get("status") == "Traced"
               and r.get("type") in ("DNa02", "DNa03")]
    rows_by_id = {int(r["bodyId"]): r for r in annotations}
    summaries = []
    with load_array(args.graph, allow_pickle=False) as raw:
        ids, crow, col, counts = (raw[k] for k in ("body_ids", "crow", "col", "counts"))
        index = {int(body): i for i, body in enumerate(ids)}
        for name, group in route["groups"].items():
            first = {int(r["body_id"]): r for r in group["targets"]}
            for target in targets:
                body = int(target["bodyId"])
                i = index[body]
                paths = []
                for edge in range(int(crow[i]), int(crow[i + 1])):
                    intermediate = int(ids[col[edge]])
                    if intermediate not in first:
                        continue
                    row = rows_by_id[intermediate]
                    paths.append({
                        "intermediate_body_id": str(intermediate), "type": row.get("type"),
                        "superclass": row.get("superclass"), "soma_side": row.get("somaSide"),
                        "group_to_intermediate_synapses": (
                            first[intermediate]["synapses_from_group"]),
                        "intermediate_to_target_synapses": float(counts[edge]),
                    })
                paths.sort(key=lambda r: (-r["intermediate_to_target_synapses"],
                                          int(r["intermediate_body_id"])))
                direct = first.get(body, {}).get("synapses_from_group", 0)
                summaries.append({"source_group": name, "target_body_id": str(body),
                                  "target_type": target["type"],
                                  "target_soma_side": target["somaSide"],
                                  "direct_synapses": direct, "two_hop_intermediates": len(paths),
                                  "paths": paths})
    hashes["script"] = sha256(__file__)
    result = {"scope": "structural direct and two-edge paths only; no transmission/dynamics proof",
              "source_sha256": hashes, "target_selection": "exact DNa02 or DNa03, status Traced",
              "no_edge_count_threshold": True, "functional_ports_assigned": False,
              "limitations": ["first-edge counts pool an entire annotated sensory group",
                              "two positive structural edges do not imply excitatory transmission",
                              "weak paths retained; significance and functional subtype unverified",
                              "soma side is not the vehicle turn sign"], "routes": summaries}
    (out / "paths.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps([{k: r[k] for k in ("source_group", "target_body_id", "direct_synapses",
                                         "two_hop_intermediates")} for r in summaries], indent=2))


if __name__ == "__main__":
    main()
