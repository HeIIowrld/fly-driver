"""Compare all prequalified development vetoes before a new independent capture."""
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
EXPERIMENT = OUT.parent / "vision-corner-added-left-phase-veto-20261113-01"
sys.path.insert(0, str(EXPERIMENT))
from veto_added_left import (
    CLI, EVAL, HIGHLANDS, METHOD, PER_RECORDING, PREVIOUS, RISK, apply_policy,
    available, counts, dependencies, describe, digests, keys, load_sensor, old, reference, score,
)
import json


def prequalified():
    receipt = EXPERIMENT / "frozen.json"
    original = old.read(receipt)
    for source, checksum in original["sources"].items():
        if old.digest(Path(source)) != checksum:
            raise ValueError(f"Development source changed: {source}")
    eligible = [trial for trial in original["trials"] if trial["left_lost"] == 0
                and max(trial["inside_delta"].values()) <= original["development_inside_cap"]
                and sum(trial["inside_delta"].values()) <= 0]
    candidates = {f"v{index:02}": trial for index,trial in enumerate(eligible)}
    if not candidates:
        raise ValueError("No prequalified thresholds")
    calibration = OUT / "prequalified.json"
    old.write(calibration, dict(
        candidates=candidates, development=original["development"], risks=PER_RECORDING,
        previous_receipt_sha256=old.digest(PREVIOUS), development_receipt_sha256=old.digest(receipt),
        sources=digests((Path(__file__),receipt,PREVIOUS)), source_revision=old.revision(),
        seed=None, termination="all_thresholds_listed_before_comparison"))
    return calibration, candidates


def compare(candidates):
    original = old.read(PREVIOUS)
    observer, settings, structured, _ = dependencies()
    if settings != original["settings"]:
        raise ValueError("Observer settings changed")
    reports = {candidate:{} for candidate in candidates}
    sources = {PREVIOUS}
    for track,name in (*EVAL,("ks_highlands","highlands")):
        replay = HIGHLANDS / "archive-replay" if name=="highlands" else CLI / (
            "vallelunga-retry" if name=="vallelunga" else name)
        receipt = replay / "result.json"
        proof = old.read(receipt)
        raw, recovered = (replay / p for p in ("sensors.npz","recovered-sensors.npz"))
        base, offers = describe(load_sensor(raw,proof["inputs_sha256"]),
                                load_sensor(recovered,proof["recovered_input_sha256"]),
                                observer,settings,structured)
        if name=="highlands":
            path = HIGHLANDS / "labels.jsonl"
            truth = [json.loads(row) for row in path.read_bytes().splitlines()]
        else:
            truth,path = reference(track)
        prediction_keys = keys(base)
        label_keys = keys(truth)
        if len(set(label_keys)) != len(label_keys) or not set(label_keys).issubset(prediction_keys):
            raise ValueError("Comparison labels differ from the forecast timeline")
        if name != "monza" and prediction_keys != label_keys:
            raise ValueError("Full-timeline comparison labels differ from the forecast timeline")
        before = score(base,truth)["counts"]
        sources.update((raw,recovered,receipt,path))
        print(track,flush=True)
        for candidate, setting in candidates.items():
            filtered = available(base,offers,setting["predicate"],setting["threshold"])
            output, _ = apply_policy(base,filtered,original,None)
            if any(a!=b for a,b in zip(base,output,strict=True)
                   if a["forecasts"][METHOD]["state"]=="right"):
                raise ValueError("Right forecast changed")
            after = score(output,truth)
            delta = counts(before,after["counts"],(*RISK,"hit_1","hit_2"))
            limits = all(delta[field] <= cap for field,cap in PER_RECORDING.items())
            reports[candidate][track] = dict(delta=delta,score=after,caps_pass=limits)
            print(candidate,delta,limits,flush=True)
    return reports,sources


def main():
    calibration,candidates = prequalified()
    reports,sources = compare(candidates)
    must_improve = ("ks_silverstone1967","ks_vallelunga")
    eligible = [c for c,rows in reports.items()
                if all(rows[t]["delta"]["hit_1"] > 0 for t in must_improve)
                and all(r["caps_pass"] and r["delta"]["hit_2"] >= 0 for r in rows.values())]
    records = OUT / "comparison.json"
    old.write(records,dict(
        eligible=eligible,candidates=candidates,results=reports,
        prequalified_sha256=old.digest(calibration),sources=digests(sources),
        selection_recordings=list(reports[next(iter(reports))]), seed=None,
        source_revision=old.revision(),termination="archive_end"))
    if not eligible:
        print("No candidate met all comparison requirements",flush=True)
        return
    chosen = max(eligible,key=lambda c: (
        reports[c]["ks_highlands"]["delta"]["hit_1"],
        reports[c]["monza"]["delta"]["hit_1"],
        sum(r["delta"]["hit_1"] for r in reports[c].values()),
        -sum(sum(r["delta"][k] for k in RISK) for r in reports[c].values()),
        c))
    setting = candidates[chosen]
    old.write(OUT/"frozen.json",dict(
        chosen=setting,candidate=chosen,prequalified_sha256=old.digest(calibration),
        comparison_sha256=old.digest(records),base_policy_sha256=old.digest(PREVIOUS),
        sources=digests((Path(__file__),calibration,records,PREVIOUS)),
        selection_recordings=list(reports[chosen]),deployment_risk_caps=PER_RECORDING,
        source_revision=old.revision(),seed=None,termination="frozen_before_new_capture"))
    print("Frozen",chosen,setting,flush=True)


if __name__ == "__main__":
    main()
