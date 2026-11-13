"""Identify which frozen confirmation rule suppresses physically correct Silverstone candidates."""
from pathlib import Path
from collections import Counter
import sys

import numpy as np

from geometry_intervention import OUT,TEST,METHODS,old,load_observer
sys.path.insert(0,str(OUT.parent/"vision-corner-paired-risk-20261112-01"))
from paired_risk import archive_scores


def main():
    path=OUT/"geometry-intervention.json"
    parent=old.read(path)
    for name,sha in parent["source_sha256"].items():
        assert old.digest(Path(name))==sha
    models,settings,sources=load_observer()
    report={}
    for mode,intervention in parent["tracks"].items():
        sensor=OUT/f"oracle-{mode}-sensors.npz"
        assert old.digest(sensor)==intervention["sensor_sha256"]
        metadata,selected,raw_score,phase=archive_scores(sensor,settings,models)
        index={r["frame"]:i for i,r in enumerate(metadata)}
        data=[]
        for row in intervention["details"]:
            i=index[row["frame"]]
            offer=selected.get(i)
            assert (list(offer) if offer else None)==row["candidate"]
            if row["selection"]!="joint_hit":
                continue
            state={}
            for j,method in enumerate(METHODS):
                score=float(raw_score[i][j])
                q=float(phase[i])
                cut=settings["cuts"][method]
                reasons=[]
                if not np.isfinite(score) or not np.isfinite(q):
                    reasons.append("unavailable")
                if np.isfinite(score) and score<cut:
                    reasons.append("confidence")
                if np.isfinite(q) and q>settings["phase_cut"]:
                    reasons.append("inside_veto")
                state[method]=dict(raw_score=score if np.isfinite(score) else None,
                                   phase_inside=q if np.isfinite(q) else None,
                                   reasons=reasons,accepted=not reasons)
                assert state[method]["accepted"]==(row["confirmed"][method]=="joint_hit")
            data.append(dict(frame=row["frame"],truth_m=row["truth_m"],candidate=row["candidate"],methods=state))
        report[mode]=dict(candidates=len(data),frames=data,
                          reasons={m:dict(Counter("+".join(r["methods"][m]["reasons"]) or "accepted"
                                                 for r in data)) for m in METHODS})
        sources.add(sensor)
    sources.update((Path(__file__),path,Path(archive_scores.__code__.co_filename)))
    old.write(OUT/"geometry-confirmation-audit.json",dict(status="diagnostic_only",tracks=report,
              thresholds=settings["cuts"],phase_cut=settings["phase_cut"],
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="actual_confirmation_misses_audited"))
    print({n:r["reasons"] for n,r in report.items()},flush=True)


if __name__=="__main__":
    main()
