"""Separate re-detection after a held warning from re-detection after a direct one."""

from pathlib import Path
from evaluate_fused_corner import read_rows
from replay import CONFIRM, LOCK, ORIGINAL, key, totals
from replay_learned import baseline_data, publish
from score_corner_captures import summarize
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent


def main():
    continuity = OUT / "continuity-result.json"
    baseline = ORIGINAL / "result.json"
    hashes = {Path(__file__), continuity, baseline, Path(publish.__code__.co_filename)}
    results = {}
    for track, proof in read(continuity)["tracks"].items():
        new = OUT / f"{track}-continuity.jsonl"
        old = ORIGINAL / f"{track}-frames.jsonl"
        assert digest(new) == proof["rows_sha256"]
        assert digest(old) == read(baseline)["tracks"][track]["rows_sha256"]
        hashes.update((new, old))
        group, _, _, _ = baseline_data(track, hashes)
        timelines = {}
        records = []
        previous = None
        for row, source, truth in zip(read_rows(new), read_rows(old), group["rows"], strict=True):
            assert key(row) == key(source) == key(truth)
            alert = {}
            for mode, (_, old_hold) in CONFIRM.items():
                direct = old_hold.removesuffix("_hold")
                base = row["final"][mode]
                offer = row["final"][f"{mode}_4m"]
                held = (row["local_travel_m"] is not None and previous is not None
                        and previous["proposals"][old_hold] is not None
                        and previous["proposals"][direct] is None)
                alert[mode] = base
                alert[f"{mode}_4m"] = offer
                alert[f"{mode}_held_4m"] = offer if held else base
            for mode, proposal in alert.items():
                scored = publish(truth, proposal, source["baseline_observable"] or proposal is not None)
                timelines.setdefault(mode, {}).setdefault(truth["run"], []).append(scored)
            if any(alert[f"{m}_4m"] != alert[m] for m in CONFIRM):
                records.append(dict(run=row["run"],frame=row["frame"],truth=row["truth"],
                                    label=row["label"], decisions=alert))
            previous = source
        path = OUT / f"{track}-bridge-changes.jsonl"
        jsonl(path, records)
        results[track] = dict(rows_sha256=digest(path),
            metrics={mode: totals({run:summarize(rows) for run,rows in runs.items()})
                     for mode,runs in timelines.items()})
        print(track,results[track]["metrics"],flush=True)
    write(OUT / "bridge-result.json",dict(status="evaluated",seed=None,posthoc=True,tracks=results,
          termination="full_timeline_scored",source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
