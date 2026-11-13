"""Require a transported previous warning to corroborate a new boundary-only offer."""

import json
from pathlib import Path

import numpy as np

from corner_fusion_boundaries import read_motion, transform
from flybrain_drive.ac_camera_archive import CameraArchive
from replay import CONFIRM, key, totals
from replay_learned import baseline_data, publish
from score_corner_captures import adjacent, summarize
from train_vision_teacher_line import digest, jsonl, read, write

OUT = Path(__file__).parent
BOUNDED = OUT.parent / "vision-corner-bounded-boundary-20261112-01"
CUTS = (2., 4., 5.)


def corroborates(old, offer, travel, tolerance):
    if old is None or offer is None or travel is None or old[0] != offer[0]:
        return False
    distance = old[1] - travel
    return distance > 0. and abs(offer[1] - distance) <= tolerance


def run(track, hashes):
    group, _, mapping, bundles = baseline_data(track, hashes)
    record = BOUNDED / f"{track}-frames.jsonl"
    original_receipt = BOUNDED / "result.json"
    assert digest(record) == read(original_receipt)["tracks"][track]["rows_sha256"]
    hashes.update((record, original_receipt))
    rows = {key(r): r for r in map(json.loads, record.open())}
    truth = {key(r): r for r in group["rows"]}
    assert list(rows) == list(truth)
    saved, all_reports = [], {}
    for tag, folder in read(mapping).items():
        methods = {name: [] for name in CONFIRM}
        methods.update({f"{name}_{cut:g}m": [] for name in CONFIRM for cut in CUTS})
        previous = None
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[tag]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                k = tag, item["capture_id"]
                if k not in rows:
                    previous = None
                    continue
                row, t = rows[k], truth[k]
                name = Path(item["file"]).name
                assert archive.by_name[name]["sha256"] == item["file_sha256"]
                meta = archive.json(name)
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, k[1], item["generation"])
                motion = read_motion(meta)
                step = None
                if motion is not None and previous is not None and previous[2] is not None and adjacent(previous[0], t):
                    step = transform(motion, previous[2])
                travel = float(np.linalg.norm(step[1])) if step is not None else None
                final = {}
                for mode in CONFIRM:
                    old, offer = (row["final"][m] for m in (mode, "bounded_normal_" + mode))
                    final[mode] = old
                    prior = previous[1]["final"][mode] if previous is not None else None
                    for cut in CUTS:
                        final[f"{mode}_{cut:g}m"] = old or (offer if corroborates(prior, offer, travel, cut) else None)
                for mode, proposal in final.items():
                    methods[mode].append(publish(t, proposal, proposal is not None))
                saved.append(dict(run=tag, frame=k[1], truth=t["truth"], label=t["label"],
                                  truth_endpoints_m=t["truth_endpoints_m"], local_travel_m=travel, final=final))
                previous = t, row, motion
        all_reports[tag] = {mode: summarize(values) for mode, values in methods.items()}
    assert [key(r) for r in saved] == list(truth)
    path = OUT / f"{track}-continuity.jsonl"
    jsonl(path, saved)
    metrics = {mode: totals({tag: r[mode] for tag, r in all_reports.items()})
               for mode in next(iter(all_reports.values()))}
    return dict(rows_sha256=digest(path), metrics=metrics)


def main():
    hashes = {Path(__file__), Path(read_motion.__code__.co_filename), Path(publish.__code__.co_filename),
              Path(adjacent.__code__.co_filename)}
    results = {}
    for track in ("ks_silverstone1967", "spa"):
        results[track] = run(track, hashes)
        print(track, results[track]["metrics"], flush=True)
    write(OUT / "continuity-result.json", dict(status="evaluated", posthoc=True, seed=None,
          tracks=results, tested_distance_tolerances_m=CUTS, termination="full_timeline_scored",
          source_sha256={str(p): digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
