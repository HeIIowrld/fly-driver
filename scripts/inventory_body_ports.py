"""Inventory measured body-sensory annotations without assigning unreviewed circuit ports."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from flybrain_drive.episode_storage import load_array


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import pyarrow.feather as feather

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    with load_array(args.graph, allow_pickle=False) as graph:
        ids = set(map(int, graph["body_ids"]))
    columns = ["bodyId", "type", "instance", "status", "class", "subclass",
               "entryNerve", "rootSide", "somaSide", "mancType", "mancBodyid"]
    rows = [r for r in feather.read_table(args.annotations, columns=columns).to_pylist()
            if r["status"] == "Traced" and r["class"] == "mechanosensory_proprioceptive"]
    for row in rows:
        row["present_in_recovered_graph"] = row["bodyId"] in ids
    counts = collections.Counter(str(r["subclass"]) for r in rows)
    chordotonal = [r for r in rows if r["subclass"] == "chordotonal organ"]
    groups = collections.Counter((str(r["entryNerve"]), str(r["rootSide"]), str(r["type"]))
                                 for r in chordotonal)
    inventory = {
        "scope": "annotation inventory only; no functional role or vehicle-port assignment",
        "selection": {"status": "Traced", "class": "mechanosensory_proprioceptive"},
        "source_sha256": {"annotations": digest(args.annotations), "graph": digest(args.graph),
                           "script": digest(__file__)},
        "rows": len(rows), "subclass_counts": dict(counts),
        "present_in_graph": sum(r["present_in_recovered_graph"] for r in rows),
        "chordotonal_groups": [{"entryNerve": k[0], "rootSide": k[1], "type": k[2], "count": n}
                                for k, n in sorted(groups.items())],
        "functional_role_verified": False,
        "limitations": [
            "chordotonal organ annotation does not identify FeCO claw/hook/club subtype",
            "entryNerve/rootSide require biological review before a vehicle side/sign mapping",
            "comma-separated type labels remain ambiguous and are preserved literally",
            "FANC/FlyWire physiological subtype evidence is not a MaleCNS body-ID mapping",
        ],
        "neurons": rows,
    }
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: inventory[k] for k in ("rows", "subclass_counts", "present_in_graph")}))


if __name__ == "__main__":
    main()
