"""Finite read-only watcher: audit completed TRAIN/development leases, never evaluation."""

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--first-raw-audit", required=True)
    parser.add_argument("--first-targets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    collection = Path(args.collection)
    plan = read(collection / "collection-plan.json")
    plan_hash = sha256(collection / "collection-plan.json")
    assert plan["schema"] == "flybrain.flight.fresh_teacher_corpus.v3"
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            plan_sha256=plan_hash,
            program_sha256=sha256(__file__),
            maximum_wait_seconds=2400,
        ),
    )
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    parent = kernel.OpenProcess(0x100000, False, args.parent_pid)  # SYNCHRONIZE only
    assert parent or (collection / "result.json").exists(), (
        "Confirmed collection process handle or actual terminal result required"
    )
    deadline = time.monotonic() + 2400
    results = []

    def execute(command, directory):
        with (run / (directory + ".log")).open("x", encoding="utf8") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            write_json(
                run / "progress.json", dict(stage=directory, child_pid=child.pid, completed=results)
            )
            code = child.wait()
        write_json(run / (directory + "-exit.json"), dict(exit_code=code, command=command))
        assert code == 0, f"Child failed: {directory}, exit{code}"

    try:
        for entry in plan["episodes"]:
            if entry["role"] == "evaluation":
                continue
            while True:
                assert sha256(collection / "collection-plan.json") == plan_hash
                partial = collection / "partial-result.json"
                complete = read(partial)["episodes"] if partial.exists() else []
                selected = [r for r in complete if r["index"] == entry["index"]]
                if selected:
                    assert selected[0]["role"] == entry["role"]
                    break
                terminal = collection / "result.json"
                if terminal.exists():
                    write_json(
                        run / "result.json",
                        dict(
                            status="completed_available_terminal_prefix_audits",
                            episodes=results,
                            collection_status=read(terminal)["status"],
                            collection_result_sha256=sha256(terminal),
                            first_uncompleted_index=entry["index"],
                            whole_collection_completed=False,
                            evaluation_arrays_or_receipts_opened=False,
                            training_dataset_sealed=False,
                            actuator_messages_sent=0,
                            ui_actions_sent=0,
                            goal_achieved=False,
                        ),
                    )
                    return
                assert kernel.WaitForSingleObject(parent, 0) == 258, "Collection process exited"
                assert time.monotonic() < deadline, "Finite audit watcher deadline"
                time.sleep(1)
            recording = Path(selected[0]["path"])
            assert sha256(recording / "result.json") == selected[0]["result_sha256"]
            name = f"episode-{entry['index']:02d}-{entry['role']}"
            raw = Path(args.first_raw_audit) if entry["index"] == 0 else run / name / "raw-audit"
            targets = Path(args.first_targets) if entry["index"] == 0 else run / name / "targets"
            if entry["index"] != 0:
                execute(
                    [
                        sys.executable,
                        "-m",
                        "scripts.audit_ac_engine_ai_probe",
                        "--recording",
                        str(recording),
                        "--require-aid-off",
                        "--output",
                        str(raw),
                    ],
                    name + "-raw",
                )
                execute(
                    [
                        sys.executable,
                        "-m",
                        "scripts.label_ac_engine_ai_receipts",
                        "--recording",
                        str(recording),
                        "--raw-audit",
                        str(raw),
                        "--calibration-run",
                        "runs/ac-shared-lateral-recovery-prepare-20261029-t933/calibration",
                        "--output",
                        str(targets),
                    ],
                    name + "-targets",
                )
            assert (
                read(raw / "result.json")["status"]
                == read(targets / "result.json")["status"]
                == "completed"
            )
            assert read(raw / "manifest.json")["config"]["events_sha256"] == sha256(
                recording / "events.jsonl"
            )
            assert read(targets / "manifest.json")["config"]["events_sha256"] == sha256(
                recording / "events.jsonl"
            )
            results.append(
                dict(
                    **entry,
                    recording=str(recording),
                    raw_audit=str(raw),
                    targets=str(targets),
                    raw_result_sha256=sha256(raw / "result.json"),
                    target_result_sha256=sha256(targets / "result.json"),
                )
            )
            write_json(
                run / "partial-result.json", dict(status="auditing_known_prefix", episodes=results)
            )
            print(json.dumps(dict(audited=name, completed=len(results))), flush=True)
        write_json(
            run / "result.json",
            dict(
                status="completed_preassigned_train_development_audits",
                episodes=results,
                evaluation_arrays_or_receipts_opened=False,
                training_dataset_sealed=False,
                actuator_messages_sent=0,
                ui_actions_sent=0,
                goal_achieved=False,
            ),
        )
    except Exception as error:
        write_json(
            run / "result.json",
            dict(
                status="stopped_with_original_prefix",
                episodes=results,
                error=type(error).__name__,
                reason=str(error),
                goal_achieved=False,
            ),
        )
        raise
    finally:
        if parent:
            kernel.CloseHandle(parent)


if __name__ == "__main__":
    main()
