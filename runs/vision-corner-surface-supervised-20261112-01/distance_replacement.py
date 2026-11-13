"""Diagnose same-direction distance replacement using two learned readouts."""
from collections import Counter, defaultdict
from pathlib import Path
import json

from prepare import OUT, SILVER, old

MIN_SEPARATION = 3.
MAX_DISAGREEMENT = 3.
CUT = .25


def refine(current, direct, learned, profile, *, require_higher_score):
    if current is None or direct is None or learned is None:
        return current
    direction, distance = current
    if direct[0] != direction or learned[0] != direction:
        return current
    low, high = sorted([direct[1], learned[1]])
    if high-low > MAX_DISAGREEMENT or low-MIN_SEPARATION < distance < high+MIN_SEPARATION:
        return current
    candidates = [(old.signals(profile, *p)["local"], p) for p in (direct,learned)]
    supported = [(score,p) for score,p in candidates if score is not None and score >= CUT]
    if not supported:
        return current
    score, candidate = max(supported, key=lambda x:x[0])
    if require_higher_score:
        base_score = old.signals(profile, *current)["local"]
        if base_score is None or score <= base_score:
            return current
    return candidate


def main():
    entry = old.context()
    runner = entry.import_replay()
    sources = {Path(__file__), Path(runner.__file__), Path(old.signals.__code__.co_filename)}
    truth, priors, _, _, profiles, _ = runner.setup(SILVER, sources)
    frames = OUT / "silverstone-frames.jsonl"
    receipt = OUT / "result.json"
    previous = old.read(receipt)
    assert old.digest(frames) == previous["rows_sha256"]
    sources.update((frames, receipt))
    rows = [json.loads(line) for line in frames.read_text(encoding="utf8").splitlines()]
    summary, changed = defaultdict(lambda: defaultdict(list)), []
    count = Counter()
    for row, reference, prior in zip(rows, truth, priors, strict=True):
        key = runner.key(row)
        assert key == runner.key(reference) == runner.key(prior)
        offers = row["offers"]
        for name in runner.MODELS:
            base = row["decisions"][f"{name}_far_wide"]
            profile = profiles[name].get(key)
            for mode, strict in (("agreement", False), ("must_beat_existing", True)):
                candidate = refine(base, offers.get("direct_replace"), offers.get("learned_row"), profile,
                                   require_higher_score=strict)
                assert (candidate is None) == (base is None)
                if candidate:
                    assert candidate[0] == base[0]
                method = f"{name}_{mode}"
                summary[method][key[0]].append(old.publish(reference, candidate,
                             prior["baseline_observable"] or candidate is not None))
                if candidate != base:
                    before, after = [old.category(reference, p) for p in (base, candidate)]
                    count[method, before, after] += 1
                    changed.append(dict(method=method, frame=key[1], truth=reference["truth"],
                                        distance=reference["truth_endpoints_m"], before=base, after=candidate,
                                        scores=[old.signals(profile,*p)["local"] for p in (base,candidate)],
                                        category_before=before, category_after=after))
    metrics = {mode: runner.totals(by_run) for mode,by_run in summary.items()}
    for mode, values in metrics.items():
        name = mode.split("_",1)[0]
        for invariant in ("left", "right", "outside", "in_corner", "frames", "no_output"):
            assert values[invariant] == previous["metrics"][f"{name}_far_wide"][invariant]
    data = dict(status="posthoc_distance_diagnostic", seed=None, source_revision=old.revision(),
                threshold=CUT, disagreement_m=MAX_DISAGREEMENT, separation_m=MIN_SEPARATION,
                metrics=metrics, changed=changed,
                transitions=[dict(method=k[0], before=k[1], after=k[2], count=v) for k,v in sorted(count.items())],
                source_sha256={str(p):old.digest(p) for p in sorted(sources)},
                termination="all_frames_scored_direction_and_alert_presence_preserved")
    old.write(OUT / "distance-replacement.json", data)
    print(json.dumps(dict(metrics=metrics, transitions=data["transitions"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
