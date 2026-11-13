"""Test frozen confirmation on the same frame with an offline surface candidate."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from surface_replay import OUT, TRACK, read, reference
from kn5_surfaces import digest

MODES = ("old", "image", "mesh", "adjacent_invalid_only")
CUT = .25


def main():
    entry = reference.context()
    runner = entry.import_replay()
    hashes = {Path(__file__), Path(runner.__file__), Path(reference.signals.__code__.co_filename)}
    rows, old, _, _, profiles, training = runner.setup(TRACK, hashes)
    path = OUT / "mesh-oracle.jsonl"
    receipt = OUT / "surface-labels.json"
    assert digest(path) == read(receipt)["mesh_oracle_sha256"]
    hashes.update((path, receipt))
    predicted = {r["frame"]: r for r in map(json.loads, path.read_text(encoding="utf8").splitlines())}
    assert len(predicted) == len(rows)
    timelines = {f"{name}_{mode}": defaultdict(list) for name in runner.MODELS for mode in MODES}
    paired = Counter()
    details = []
    for row, prior in zip(rows, old, strict=True):
        key = runner.key(row)
        diagnostic = predicted[key[1]]
        assert (diagnostic["truth"], diagnostic["label"], diagnostic["truth_endpoints_m"]) == (
            row["truth"], row["label"], row["truth_endpoints_m"])
        original = {name: prior["proposals"][spec[1]] for name, spec in runner.MODELS.items()}
        finals, scores = {}, {}
        for mode in MODES:
            candidate = diagnostic.get(mode)
            for name in runner.MODELS:
                value = reference.signals(profiles[name].get(key), *candidate)["local"] if candidate else None
                accepted = candidate if value is not None and value >= CUT else None
                decision = original[name] or accepted
                method = f"{name}_{mode}"
                finals[method] = decision
                scores[method] = value
                timelines[method][key[0]].append(reference.publish(row, decision,
                     prior["baseline_observable"] or decision is not None))
            if mode != "old" and not any(original.values()):
                a, b = [finals[f"{name}_{mode}"] for name in runner.MODELS]
                if a != b:
                    paired[mode, "t4t5" if b else "edge", reference.category(row, a or b)] += 1
        if row["truth"] == "approach":
            details.append(dict(frame=key[1], truth_endpoints_m=row["truth_endpoints_m"],
                                label=row["label"], candidates={mode: diagnostic.get(mode) for mode in MODES[1:]},
                                final=finals, confirmation=scores))
    report = {name: runner.totals(timeline) for name, timeline in timelines.items()}
    historical = read(reference.OLD / "result.json")["tracks"][TRACK]["metrics"]
    assert report["edge_image"] == historical["edge_far_wide"]
    assert report["t4t5_image"] == historical["t4t5_far_wide"]
    data = dict(status="privileged_candidate_diagnostic", seed=None, source_revision=reference.revision(),
                cutoff=CUT, frames=len(rows), training=training, scores=report,
                paired=[dict(mode=k[0], accepted_by=k[1], category=k[2], frames=v) for k,v in sorted(paired.items())],
                approaches=details, source_sha256={str(p): digest(p) for p in sorted(hashes)},
                termination="frozen_confirmation_and_image_control_parity_complete")
    reference.write(OUT / "frozen-confirmation.json", data)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
