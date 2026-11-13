"""Freeze the smallest change that meets the already computed development reserve."""
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "vision-corner-added-left-phase-veto-20261113-01"
sys.path.insert(0, str(PRIOR))
import veto_added_left as experiment


def main():
    old = experiment.old
    source = PRIOR / "frozen.json"
    trial_record = old.read(source)
    for path, digest in trial_record["sources"].items():
        if old.digest(Path(path)) != digest:
            raise ValueError("Development source changed")
    eligible = [t for t in trial_record["trials"] if t["left_lost"] == 0
                and t["predicate"] == "inside_probability"
                and max(t["inside_delta"].values()) <= trial_record["development_inside_cap"]
                and sum(t["inside_delta"].values()) <= 0]
    if not eligible:
        raise ValueError("No single-probability threshold meets the development reserve")
    # Minimize changes to a previously frozen detector, then choose the stricter
    # threshold among values with the same number of removals.
    chosen = min(eligible, key=lambda t: (
        sum(sum(r.values()) for r in t["removed"].values()), t["threshold"]))
    receipt = OUT / "frozen.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    sources = {**trial_record["sources"], **experiment.digests((source, Path(__file__)))}
    old.write(receipt, dict(chosen=chosen, sources=sources, development=trial_record["development"],
              development_inside_cap=trial_record["development_inside_cap"],
              deployment_risk_caps=trial_record["deployment_risk_caps"],
              source_revision=old.revision(), seed=None, termination="frozen_before_comparison"))
    print("Frozen", chosen, flush=True)
    experiment.OUT = OUT
    experiment.compare()


if __name__ == "__main__":
    main()
