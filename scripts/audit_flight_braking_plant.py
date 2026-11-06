"""Use independently replayed original SDK/native records to grade student braking."""
import argparse
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.learned_high_speed_braking_evidence import evaluate_braking
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--source-audit",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    root=Path(a.run); audit=Path(a.source_audit)
    digest=sha256(root/"result.json")
    out=begin_run(a.output,dict(args=vars(a),source_result_sha256=digest,
        source_audit_sha256=sha256(audit/"result.json"),program_sha256=sha256(__file__),
        minimum_speed_kmh=100.,minimum_drop_kmh=5.,seed=None,vehicle_control_sent=False))
    result=dict(status="failed",passed=False,vehicle_control_sent=False)
    try:
        m=read(audit/"manifest.json")["config"]
        evidence=read(audit/"result.json")
        source=read(root/"result.json")
        if (m["source_result_sha256"]!=digest
                or evidence["status"]!="completed_independent_student_audit"
                or not evidence["actuator_feedback_raw_bytes_present"]
                or evidence["maximum_prediction_difference"]>2e-6
                or evidence["sdk_requests_verified"]!=source["student_requests"]):
            raise ValueError("Independently verified original control and native bytes required")
        for name in ("actuator/actuator-terminal.json","actuator/actuator-feedback.json"):
            if sha256(root/name)!=source["artifact_sha256"][name]:
                raise ValueError("Original terminal or plant evidence changed")
        terminal=read(root/"actuator/actuator-terminal.json")
        feedback=read(root/"actuator/actuator-feedback.json")
        result.update(evaluate_braking(terminal["events"],feedback),
                      status="completed_braking_evidence_audit",
                      original_trial_status=source["status"],
                      independent_stop_proven=evidence["native_neutral_stop_verified"])
    except Exception as error:
        result["reason"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
