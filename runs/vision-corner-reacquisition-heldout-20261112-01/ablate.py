"""Swap the saved prior and current confirmation independently on common offers."""

from collections import Counter
from pathlib import Path

import joblib

from evaluate_fused_corner import read_rows
from paired_motion import inputs
import profile_replay as profile
from replay import CONFIRM, CUTOFF, PRIOR, key, within
from replay_learned import signals
from summarize_paired import category
from train_vision_teacher_line import digest, read, write

OUT = Path(__file__).parent


def count(rows):
    results = Counter()
    eligible = 0
    for row in rows:
        if row["candidate"] is None or any(row["old"].values()):
            continue
        eligible += 1
        for prior in CONFIRM:
            for current in CONFIRM:
                held = row["previous_held"].get(prior)
                score = row["scores"][current]
                accepted = (held is not None and row["travel_m"] is not None
                            and within(row["candidate"], held, row["travel_m"])
                            and score is not None and score >= CUTOFF)
                if accepted:
                    results[prior, current, category(row, row["candidate"])] += 1
    return dict(eligible=eligible, confirmed=[dict(prior=k[0],current=k[1],category=k[2],count=n)
                                               for k,n in sorted(results.items())])


def silverstone_623(hashes):
    track = "ks_silverstone1967"
    lock = OUT.parent / "vision-vallelunga-local-corner-protocol-20261112-01"
    model = lock / "confirmation-models.joblib"
    frozen = lock / "lock.json"
    assert {Path(k).resolve(): v for k,v in read(frozen)["hashes"].items()}[model.resolve()] == digest(model)
    continuity = PRIOR / f"{track}-continuity.jsonl"
    proof = PRIOR / "continuity-result.json"
    assert digest(continuity) == read(proof)["tracks"][track]["rows_sha256"]
    hashes.update((model, frozen, continuity, proof))
    rows = read_rows(continuity)
    data = inputs(track, hashes)
    weights = joblib.load(model)
    row, previous = rows[623], rows[622]
    candidate = row["final"]["t4t5_4m"]
    assert key(row) == (track,623) and candidate == [2,6.]
    scores = {}
    for name, (mode, _) in CONFIRM.items():
        features = data[mode]
        selected = [key(r) for r in features["selected"]]
        assert len(set(selected)) == len(selected)
        values = profile.predict(features,weights[mode])[selected.index(key(row))]
        scores[name] = signals(values,*candidate)["local"]
    offer = dict(row,candidate=candidate, old={k:row["final"][k] for k in CONFIRM}, scores=scores,
                 previous_held={k:previous["final"][k] for k in CONFIRM},
                 travel_m=row["local_travel_m"])
    return dict(offer=offer, swap=count([offer]))


def main():
    hashes = {Path(__file__),Path(within.__code__.co_filename),Path(signals.__code__.co_filename)}
    result = silverstone_623(hashes)
    print(result,flush=True)
    write(OUT/"silverstone-623-ablation.json",dict(result,source_sha256={str(p):digest(p) for p in sorted(hashes)}))


if __name__ == "__main__":
    main()
