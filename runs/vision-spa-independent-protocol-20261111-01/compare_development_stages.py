from pathlib import Path
from statistics import median
from audit import INPUTS, MUGELLO, RECOVERY, RUNS
from evaluate_fused_corner import load_frozen, load_score
from train_vision_teacher_line import digest, read, write
from transfer import BASES, compare_tracks

out=Path(__file__).parent
reports={}
hashes={Path(__file__)}
for track in (*INPUTS,"imola"):
    if track=="imola":
        evaluation=RUNS/"vision-imola-approach-diagnostic-20261111-01"
        base=evaluation/"base"
        cfg=RUNS/"vision-imola-approach-capture-protocol-20261111-01/diagnostic-recovery.json"
    else:
        evaluation=MUGELLO if track=="mugello" else RECOVERY/track
        base=BASES[track]
        cfg=RECOVERY/INPUTS[track][0]
    protocol=load_frozen(cfg)
    rules_path=Path(protocol["base_protocol"])
    rules=load_frozen(rules_path,Path(protocol["archived_base_entrypoint"]) if protocol["archived_base_entrypoint"] else None)["comparison"]
    result_path=evaluation/"result.json"
    base_path=base/"result.json"
    r=read(result_path)
    b=read(base_path)
    assert digest(base_path)==r["base_receipt_sha256"] and digest(cfg)==r["protocol_sha256"]
    parent=load_score(base/"parent/score",b["score_receipt_sha256"]["parent"])
    mixed=load_score(base/"candidate/score",b["score_receipt_sha256"]["candidate"])
    recovered=load_score(evaluation/"score",r["score_receipt_sha256"])
    mixed_report=compare_tracks(parent,mixed,rules)
    recovered_report=compare_tracks(parent,recovered,rules)
    assert recovered_report["metrics"]==r["comparison"]["metrics"]
    reports[track]=dict(revision=r["revision"],mixed=mixed_report,recovered=recovered_report)
    hashes.update((cfg,rules_path,result_path,base_path))
    print(track, [(key,rep["metrics"]["candidate"],len(rep["stable_leads_m"]))
           for key,rep in (("mixed",mixed_report),("recovered",recovered_report))])
write(out/"development-stages.json",dict(tracks=reports,seed=None,source_sha256={str(p):digest(p) for p in sorted(hashes)},
      termination="stored_stage_comparisons_verified"))
