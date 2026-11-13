"""Attribute the missed independent left turns to the first failed stage."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
sys.path.insert(0,str(RUNS/"vision-corner-left-right-audit-20261113-01"))
BASE = RUNS / "vision-corner-left-risk-reallocation-20261113-01"
FROZEN = RUNS / "vision-corner-phase-veto-frontier-20261113-01/frozen.json"
sys.path.insert(0,str(RUNS/"vision-corner-added-left-phase-veto-20261113-01"))
from veto_added_left import available, dependencies, describe, digests, load_sensor, old
from audit_and_refit import audit, build


def main():
    proof = old.read(OUT/"archive-replay/result.json")
    paths = [OUT/"archive-replay"/name for name in ("sensors.npz","recovered-sensors.npz")]
    raw = load_sensor(paths[0],proof["inputs_sha256"])
    recovered = load_sensor(paths[1],proof["recovered_input_sha256"])
    observer,settings,structured,_ = dependencies()
    baseline,offers = describe(raw,recovered,observer,settings,structured)
    policy = old.read(FROZEN)["chosen"]
    threshold = old.read(BASE/"frozen.json")["chosen"]["add"]
    raw_offers = {o[0]:o for o in offers if o[1]==0}
    gate_pass = {o[0] for o in available(baseline,offers,policy["predicate"],policy["threshold"]) if o[1]==0}
    truth = OUT/"labels.jsonl"
    labels = [json.loads(line) for line in truth.read_bytes().splitlines()]
    indexes = {(row["run"],row["frame"]):i for i,row in enumerate(baseline)}
    audit_rows = audit(build(recovered,settings),baseline,labels,observer,settings)
    counts = Counter()
    for row in audit_rows:
        if row["side"]!=1:
            continue
        i = indexes[row["run"],row["frame"]]
        offer = raw_offers.get(i)
        if offer is None:
            result = "no_eligible_raw_selected_left"
        elif not np.isfinite(offer[3]) or offer[3] < threshold:
            result = "structured_confirmation_below_cut"
        elif i not in gate_pass:
            result = "inside_probability_exceeds_approach"
        else:
            result = "additional_left_admitted"
        counts[result] += 1
        row["supplement_status"] = result
        if offer is not None:
            row.update(add_distance_m=offer[2],add_probability=offer[3],
                       raw_inside_probability=float(offer[-1][4]),
                       raw_approach_probability=float(offer[-1][3]))
    rows = OUT/"approach-audit.jsonl"
    old.jsonl(rows,audit_rows)
    baseline_failures = Counter(row["reason"] for row in audit_rows if row["side"]==1)
    old.write(OUT/"approach-audit.json",dict(
        left_baseline_first_failure=dict(baseline_failures),left_supplement_first_failure=dict(counts),
        rows_sha256=old.digest(rows),sources=digests((*paths,truth,FROZEN,Path(__file__),
            Path(audit.__code__.co_filename))),seed=None,termination="archive_end"))
    print(baseline_failures,counts,flush=True)


if __name__ == "__main__":
    main()
