"""Run a fixed launch comparison serially and require cleanup between cases."""
import argparse
import subprocess
import sys
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.preflight_current_flight_control import inspect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--previous", default="mailbox-baseline80-01")
    parser.add_argument("--phase-followup", action="store_true",
                        help="Compare the phase-constrained lineages at 80C after the main matrix")
    args = parser.parse_args()
    suffix = f"{args.attempt:02}"
    base = Path("runs/lease-balanced-learning-20261105")
    temperatures = (80,) if args.phase_followup else (80, 40)
    cases = [(temp, lineage) for temp in temperatures for lineage in ("parent", "dual_guard")]
    prefix, fit = ("phase-", "phase") if args.phase_followup else ("balanced-", "joint")
    out = begin_run(base / f"{'phase-' if args.phase_followup else ''}comparison-owned-{suffix}", dict(
        cases=cases, seed=None, program_sha256=sha256(__file__), planned_seconds=.35,
        previous=args.previous, phase_followup=args.phase_followup, automatic_retry=False, ui_actions_sent=0))
    previous = base / args.previous / "run"
    rows = []
    status = "blocked"
    error = None

    def check(name):
        proof = inspect()
        write_json(out / (name + ".state.json"), proof)
        session = proof["session"]
        if (session["paused"] is not True or session["ai_controlled"] is not False
                or session["camera_active"] is not False or session.get("pending_receipt_id")):
            raise RuntimeError("Paused, camera idle, AI off and no pending receipt required")

    try:
        check("before")
        if args.phase_followup:
            main_matrix = read(base / "comparison-owned-02/result.json")
            if len(main_matrix["runs"]) != 4 or main_matrix["status"] not in (
                    "completed", "completed_with_failures"):
                raise ValueError("Require completion of the predeclared main matrix first")
            audit = read(base / "phase-joint-audit-01/result.json")
            for lineage in ("parent", "dual_guard"):
                path = base / f"{lineage}-phase-fit-01"
                digest = sha256(path / "pedals.pt")
                if (audit["status"] != "completed"
                        or not any(row["checkpoint_sha256"] == digest for row in audit["fits"].values())):
                    raise ValueError("Require independently audited phase fit")
        for temp, lineage in cases:
            label = "parent" if lineage == "parent" else "candidate"
            name = f"{prefix}{label}{temp}-{suffix}"
            case = base / name
            checkpoint = base / f"{lineage}-{fit}-fit-01/pedals.pt"
            command = [sys.executable, "-m", "scripts.run_mailbox_flight_sequence",
                       "--actor-kind", "balanced_" + lineage, "--checkpoint", str(checkpoint),
                       "--temperature-c", str(temp), "--mode", "student",
                       "--planned-student-seconds", ".35", "--completed-metadata-evidence",
                       str(previous), "--output", str(case)]
            with (out / (name + ".log")).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                write_json(out / "progress.json", dict(case=name, pid=child.pid, command=command))
                code = child.wait()
            row = dict(case=name, pid=child.pid, exitcode=code, original_child_handle=True)
            rows.append(row)
            write_json(out / (name + ".exit.json"), row)
            check(name)
            if (case / "run/result.json").exists():
                result = read(case / "run/result.json")
                row.update(run_status=result["status"], student_requests=result.get("student_requests"))
                previous = case / "run"
        status = "completed" if all(row["exitcode"] == 0 for row in rows) else "completed_with_failures"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        write_json(out / "result.json", dict(status=status, error=error, runs=rows, ui_actions_sent=0))
    if status == "blocked":
        raise SystemExit(error)


if __name__ == "__main__":
    main()
