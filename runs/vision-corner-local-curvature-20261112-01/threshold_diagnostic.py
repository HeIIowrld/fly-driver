"""Diagnose stricter confirmation thresholds on the recorded paired candidates."""

from collections import defaultdict
import json
from pathlib import Path

from replay_local import OUT, MODES, baseline_data, digest, frame_key, publish, read, summarize, write
from summarize_paired import count

CUTS = (.25, .30, .33, .35, .40, .50, .60, .70, .80)


def replay(rows, truth, cut, channel):
    timelines = defaultdict(list)
    direct = None
    name = "boundary" if channel == 0 else "t4t5"
    for row in rows:
        key = frame_key(row)
        original = row["proposals"]["baseline"]
        proposal = row["proposals"]["ranked_local_"+name]
        local = row["local_scores"][channel]
        accepted = original is None and local is not None and local >= cut and proposal is not None
        value = original or (proposal if accepted else None)
        if value is None and direct == (key[0], key[1]-1):
            score = row["held_scores"][channel]
            if score is not None and score >= cut:
                value = row["proposals"]["ranked_local_"+name+"_hold"]
        direct = key if accepted else None
        timelines[key[0]].append(publish(truth[key], value, row["baseline_observable"] or value is not None))
    return {tag: summarize(timeline) for tag, timeline in timelines.items()}


def main():
    source = OUT/"result.json"
    proof = read(source)
    hashes = {Path(__file__), source, Path(count.__code__.co_filename)}
    results = {}
    for track, report in proof["tracks"].items():
        path = OUT/f"{track}-frames.jsonl"
        assert digest(path) == report["rows_sha256"]
        hashes.add(path)
        rows = [json.loads(line) for line in path.open()]
        group, _, _, _ = baseline_data(track, hashes)
        truth = {frame_key(r): r for r in group["rows"]}
        results[track] = {}
        for channel, mode in enumerate(("boundary", "t4t5")):
            cuts = {}
            for cut in CUTS:
                summaries = replay(rows, truth, cut, channel)
                if cut == .25:
                    assert summaries == report["methods"]["ranked_local_"+mode+"_hold"]
                cuts[str(cut)] = dict(metrics=count(summaries),
                    stable=[dict(run=tag, **event) for tag, summary in summaries.items()
                            for event in summary["events"] if event["first_stable_frame"] is not None])
            results[track][mode] = cuts
    write(OUT/"threshold-diagnostic.json", dict(status="evaluated", posthoc=True, tracks=results,
          source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
