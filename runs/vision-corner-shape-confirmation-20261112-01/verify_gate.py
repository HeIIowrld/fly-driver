"""Verify data separation, candidate parity, and matched false-alert score order."""
from collections import Counter
from hashlib import file_digest
from pathlib import Path
import json
import math

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TEST = ("ks_silverstone1967","ks_vallelunga")
MODES = ("edge","t4t5")
STATES = {"straight","left","right","abstain"}


def digest(path):
    with path.open("rb") as file:
        return file_digest(file,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def outcome(row,candidate):
    if candidate is None:
        return "quiet"
    if row["truth"]!="approach":
        return f'{row["truth"]}_alert'
    direction, distance = candidate
    if str(direction)!=row["label"]:
        return "wrong_direction"
    return "joint_hit" if max(abs(distance-d) for d in row["truth_endpoints_m"])<=5. else "distance_error"


def audit_scores(rows,mode):
    supports = [r for r in rows if r["scores"][mode] is not None]
    outside = [r["scores"][mode] for r in supports if r["truth"] in ("straight","out_of_range")]
    actual = sum(r["truth"] in ("straight","out_of_range") for r in rows)
    errors = {}
    for cap in (math.floor(actual*.025),math.floor(actual*.05),math.floor(actual*.10)):
        thresholds = sorted({math.inf,*[r["scores"][mode] for r in supports]},reverse=True)
        choices = []
        for cutoff in thresholds:
            if sum(p>=cutoff for p in outside)>cap:
                continue
            hits = [r for r in supports if r["scores"][mode]>=cutoff and outcome(r,r["candidate"])=="joint_hit"]
            choices.append((len(hits),sum(r["label"]=="2" for r in hits)))
        errors[str(cap)] = dict(max_joint=max(n for n,_ in choices),max_right_joint=max(n for _,n in choices))
    correct = []
    for r in supports:
        if outcome(r,r["candidate"])=="joint_hit":
            score = r["scores"][mode]
            correct.append(dict(frame=r["frame"],label=r["label"],score=score,candidate=r["candidate"],
                                outside_rank=sum(p>=score for p in outside)))
    return dict(caps=errors,correct_candidate_scores=correct)


def main():
    trained_path = OUT/"training.json"
    trained = read(trained_path)
    assert not set(TEST)&set(trained["development"])
    assert digest(OUT/"confirmation.joblib")==trained["checkpoint_sha256"]
    files = dict(trained["source_sha256"])
    for track in trained["development"]:
        receipt = read(OUT/f"{track}-candidates.json")
        assert track not in receipt["shape_training"] and not set(TEST)&set(receipt["shape_training"])
        assert digest(OUT/f"{track}-candidates.npz")==receipt["arrays_sha256"]
        files.update(receipt["source_sha256"])
    for name,sha in files.items():
        assert not any(t in name for t in TEST)
        assert digest(ROOT/name)==sha, name
    results = {}
    for title, name in (("silverstone","result.json"),("vallelunga","vallelunga-result.json")):
        path = OUT/name
        result = read(path)
        assert result["thresholds"]==trained["threshold_by_model"]
        for filename,sha in result["source_sha256"].items():
            assert digest(ROOT/filename)==sha, filename
        records = OUT/f"{title}-frames.jsonl"
        assert digest(records)==result["rows_sha256"]
        rows = [json.loads(line) for line in records.read_text(encoding="utf8").splitlines()]
        assert len(rows)==len({(r["run"],r["frame"]) for r in rows})==1200
        paired = 0
        raw_right = Counter()
        outputs = {mode:Counter() for mode in MODES}
        for r in rows:
            candidate, base = r["candidate"],r["common_edge_warning"]
            assert (r["scores"]["edge"] is None)==(r["scores"]["t4t5"] is None)
            if r["scores"]["edge"] is not None:
                paired += 1
                assert candidate is not None
            if r["truth"]=="approach" and r["label"]=="2":
                raw_right[outcome(r,candidate)] += 1
            for mode in MODES:
                score = r["scores"][mode]
                expected = candidate if score is not None and score>=trained["threshold_by_model"][mode] else None
                candidate_mode = f"{mode}_candidate_only"
                assert r["final"][candidate_mode]==expected
                assert r["final"][f"{mode}_shared_edge"]==(base or expected)
                assert r["states"][candidate_mode] in STATES
                outputs[mode][r["states"][candidate_mode]] += 1
        assert paired==result["paired_available"]
        results[title] = dict(result_sha256=digest(path),raw_right=dict(raw_right),
                              candidate_only_states={k:dict(v) for k,v in outputs.items()},
                              score_audit={m:audit_scores(rows,m) for m in MODES})
    report = dict(status="verified_paired_replays",source_revision=trained["source_revision"],seed=trained["seed"],
                  verified_training_sources=len(files),training_sha256=digest(trained_path),test=results,
                  verifier_sha256=digest(Path(__file__)),termination="four_development_circuits_two_excluded_replays")
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps({t:{"right":v["raw_right"],
                        "audit":{m:r["caps"] for m,r in v["score_audit"].items()}}
                      for t,v in results.items()},indent=2))


if __name__ == "__main__":
    main()
