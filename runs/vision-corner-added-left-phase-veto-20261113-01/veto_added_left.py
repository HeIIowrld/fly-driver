"""Calibrate a veto on added left alerts while retaining the frozen thresholds."""
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "vision-corner-left-risk-reallocation-20261113-01"))
from left_policy import (
    CLI, DEVELOP, EVAL, METHOD, PER_RECORDING, RISK, SensorArrays, apply_policy,
    archive_mapping, dependencies, describe, digests, image, keys, label,
    load_sensor, old, read_development, recover_sections, reference, score, stream,
)

PREVIOUS = OUT.parent / "vision-corner-left-risk-reallocation-20261113-01" / "frozen.json"
HIGHLANDS = OUT.parent / "vision-highlands-short-fixed-corner-evaluation-20261113-01"
INSIDE_CAP = 1
PREDICATES = {
    "inside_probability": (4, (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)),
    "inside_minus_approach_probability": (None, (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
}


def value(offer, name):
    vector = offer[-1]
    index, _ = PREDICATES[name]
    return float(vector[index] if index is not None else vector[4] - vector[3])


def available(base, offers, name, cap):
    return [offer for offer in offers
            if base[offer[0]]["forecasts"][METHOD]["state"] == "left" or value(offer, name) <= cap]


def counts(old_counts, new_counts, fields):
    return {key: new_counts.get(key, 0) - old_counts.get(key, 0) for key in fields}


def calibrate():
    previous = old.read(PREVIOUS)
    observer, settings, structured, sources = dependencies()
    if previous["settings"] != settings:
        raise ValueError("Previous observer settings changed")
    mapping, map_code = archive_mapping()
    sources.update((Path(__file__), map_code, PREVIOUS))
    candidates = {}
    totals = {}
    for track in DEVELOP:
        arrays, truth, proof = read_development(track, sources, settings)
        raw = SensorArrays(arrays)
        inventory = mapping[track]
        folders = {name: Path(path) for name, path in old.read(inventory).items()}
        corrected, _, provenance = recover_sections(
            raw, folders, proof["camera_bundle_sha256_by_run"], stream, image, old)
        sources.update((*provenance, inventory))
        base, offers = describe(raw, SensorArrays(dict(raw, sections=corrected)), observer, settings, structured)
        if keys(base) != keys(truth):
            raise ValueError("Development sensor and truth timeline differs")
        current, changes = apply_policy(base, offers, previous, None)
        totals[track] = counts(score(base, truth)["counts"], score(current, truth)["counts"], (*RISK, "hit_1"))
        selected = {offer[0]: offer for offer in offers if offer[1] == 0}
        candidates[track] = [(selected[i], label(truth[i], 1, change["distance_m"]))
                             for i, change in changes.items() if change["action"] == "add"]
        print(track, totals[track], flush=True)
    trials = []
    for name, (_, thresholds) in PREDICATES.items():
        for cap in thresholds:
            removed = {track: dict.fromkeys((*RISK, "hit_1"), 0) for track in DEVELOP}
            for track, selected in candidates.items():
                for offer, category in selected:
                    if value(offer, name) > cap and category in removed[track]:
                        removed[track][category] += 1
            inside = [totals[t]["inside"] - removed[t]["inside"] for t in DEVELOP]
            lost_hits = sum(r["hit_1"] for r in removed.values())
            trial = dict(predicate=name, threshold=cap, left_lost=lost_hits,
                         inside_delta=dict(zip(DEVELOP, inside, strict=True)), removed=removed)
            trials.append(trial)
    eligible = [t for t in trials if t["left_lost"] == 0
                and max(t["inside_delta"].values()) <= INSIDE_CAP
                and sum(t["inside_delta"].values()) <= 0]
    receipt = OUT / "frozen.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    if not eligible:
        old.write(receipt, dict(development=list(DEVELOP), trials=trials,
                  sources=digests(sources), source_revision=old.revision(), seed=None,
                  termination="no_point_preserves_development_hits_and_inside_budget"))
        return False
    chosen = min(eligible, key=lambda t: (sum(t["inside_delta"].values()),
                 -sum(sum(v.values()) for v in t["removed"].values()), -t["threshold"], t["predicate"]))
    old.write(receipt, dict(chosen=chosen, trials=trials, development=list(DEVELOP),
              sources=digests(sources), original_policy_sha256=old.digest(PREVIOUS),
              development_previous_deltas=totals, development_inside_cap=INSIDE_CAP,
              deployment_risk_caps=PER_RECORDING, source_revision=old.revision(), seed=None,
              termination="frozen_before_comparison"))
    print("Frozen", chosen, flush=True)
    return True


def compare():
    import json

    receipt = OUT / "frozen.json"
    frozen = old.read(receipt)
    previous = old.read(PREVIOUS)
    for path, sha in frozen["sources"].items():
        if old.digest(Path(path)) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    observer, settings, structured, _ = dependencies()
    reports = {}
    for track, name in (*EVAL, ("ks_highlands", "highlands")):
        directory = HIGHLANDS / "archive-replay" if name == "highlands" else CLI / (
            "vallelunga-retry" if name == "vallelunga" else name)
        proof = old.read(directory / "result.json")
        raw, recovered = (directory / p for p in ("sensors.npz", "recovered-sensors.npz"))
        base, offers = describe(load_sensor(raw, proof["inputs_sha256"]),
                                load_sensor(recovered, proof["recovered_input_sha256"]),
                                observer, settings, structured)
        point = frozen["chosen"]
        prior, original_edits = apply_policy(base, offers, previous, None)
        filtered = available(base, offers, point["predicate"], point["threshold"])
        revised, new_edits = apply_policy(base, filtered, previous, None)
        if any(original_edits.get(i) != change for i, change in new_edits.items()):
            raise ValueError("Veto added or modified an edit")
        if name == "highlands":
            path = HIGHLANDS / "labels.jsonl"
            truth = [json.loads(line) for line in path.read_bytes().splitlines()]
        else:
            truth, path = reference(track)
        before, current, after = (score(rows, truth) for rows in (base, prior, revised))
        delta = counts(before["counts"], after["counts"], (*RISK, "hit_1", "hit_2"))
        forecasts = OUT / f"{track}-forecasts.jsonl"
        old.jsonl(forecasts, revised)
        reports[track] = dict(before=before, previous=current, after=after, delta=delta,
            caps_pass=all(delta[k] <= cap for k, cap in PER_RECORDING.items()),
            sources=digests((raw, recovered, path, directory / "result.json")),
            forecast_sha256=old.digest(forecasts))
        print(track, delta, reports[track]["caps_pass"], flush=True)
    old.write(OUT / "comparison.json", dict(tracks=reports, freeze_sha256=old.digest(receipt),
              highlands_role="previously evaluated transfer archive", termination="archive_end"))


if __name__ == "__main__" and calibrate():
    compare()
