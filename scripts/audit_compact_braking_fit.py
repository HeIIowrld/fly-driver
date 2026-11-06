"""Compare frozen phase and compact outputs without evaluating future frames as inputs."""
import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.audit_flight_braking_phase_errors import phase_masks, statistics
from scripts.fit_current_flight_launch_pedals import datasets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compact",action="append",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=begin_run(a.output,dict(args=vars(a),program_sha256=sha256(__file__),seed=None,
        used_for_optimization=False, vehicle_control_sent=False))
    result=dict(status="failed",fits={})
    try:
        for compact in map(Path,a.compact):
            cm=read(compact/"manifest.json")["config"]
            phase=Path(cm["arguments"]["phase_fit"])
            pm=read(phase/"manifest.json")["config"]
            r=read(compact/"result.json")
            for folder in (phase,compact):
                if sha256(folder/"predictions.npz")!=read(folder/"result.json")["predictions_sha256"]:
                    raise ValueError("Exact fit predictions required")
            fit=dict(fit_status=r["status"],iterations=r["iterations"],episodes=[])
            result["fits"][str(compact)]=fit
            with np.load(phase/"predictions.npz") as op,np.load(compact/"predictions.npz") as cp:
                for i,item in enumerate(datasets(Path(pm["evaluation"]))):
                    old,new=(v[f"prediction_{i}"] for v in (op,cp))
                    mask,_,_=phase_masks(item)
                    mask["low"]=item["low"]
                    label=item["data"]["labels"][:,1:]
                    if not np.array_equal(old[:,0],new[:,0]):
                        raise ValueError("Pedal-only fit changed original steering")
                    row=dict(role=item["receipt"]["role"],groups={},
                        low_prediction_max_difference=float(abs(old[item["low"]]-new[item["low"]]).max()))
                    fit["episodes"].append(row)
                    if row["role"]=="train" and row["low_prediction_max_difference"]!=0:
                        raise ValueError("Original TRAIN launch predictions changed")
                    for k,m in mask.items():
                        if not m.any():continue
                        row["groups"][k]=dict(labels=int(m.sum()),
                            original=statistics(item["original"][m,1:],label[m]),
                            phase=statistics(old[m,1:],label[m]),compact=statistics(new[m,1:],label[m]))
        result["status"]="completed"
    except Exception as exc:
        result["reason"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
