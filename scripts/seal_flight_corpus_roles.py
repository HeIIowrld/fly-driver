"""Seal every audited completed TRAIN/development lease in its original fixed role."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--audits", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    collection, audits = Path(args.collection), Path(args.audits)
    terminal, plan = read(collection / "result.json"), read(collection / "collection-plan.json")
    assert terminal["status"] in ("failed", "completed_fixed_teacher_collection_batch")
    completed = terminal["episodes"]
    expected = [e for e in completed if e["role"] in ("train", "development")]
    report = read(audits / "result.json")
    assert report["status"] in (
        "completed_available_terminal_prefix_audits",
        "completed_preassigned_train_development_audits",
    )
    lookup = {e["index"]: e for e in report["episodes"]}
    assert len(lookup) == len(report["episodes"]) == len(expected)
    assert set(lookup) == {e["index"] for e in expected}
    run = begin_run(
        args.output,
        dict(
            scope=__doc__,
            program_sha256=sha256(__file__),
            collection_terminal_sha256=sha256(collection / "result.json"),
            collection_plan_sha256=sha256(collection / "collection-plan.json"),
            audit_terminal_sha256=sha256(audits / "result.json"),
            original_collection_status=terminal["status"],
            evaluation_arrays_opened=False,
        ),
    )
    shutil.copyfile(__file__, run / "executed-program.py")
    result = dict(
        status="running",
        roles={},
        optimizer_updates=0,
        actuator_messages_sent=0,
        ui_actions_sent=0,
        goal_achieved=False,
    )
    try:
        for role in ("train", "development"):
            sources = []
            for entry in expected:
                index = entry["index"]
                assert plan["episodes"][index] == dict(index=index, role=entry["role"])
                assert lookup[index]["role"] == entry["role"]
                assert Path(lookup[index]["recording"]) == Path(entry["path"])
                assert entry["child_exit_code"] == 0
                assert sha256(Path(entry["path"]) / "result.json") == entry["result_sha256"]
                if entry["role"] == role:
                    sources.append(
                        dict(
                            recording=entry["path"],
                            raw_audit=lookup[index]["raw_audit"],
                            targets=lookup[index]["targets"],
                            collection_index=index,
                            preassigned_role=role,
                        )
                    )
            assert sources, f"No complete original {role} sources"
            config = run / f"{role}-config.json"
            write_json(
                config,
                dict(
                    seed=863426,
                    require_aid_off=True,
                    dataset_role=role,
                    collection_plan=str(collection / "collection-plan.json"),
                    sources=sources,
                    scope=(
                        "All complete audited fixed-role histories; "
                        "preserve failed/unexecuted outcomes"
                    ),
                ),
            )
            dataset, coverage = run / f"{role}-dataset", run / f"{role}-coverage"
            commands = [
                [
                    "scripts.seal_ac_engine_ai_receipt_dataset",
                    "--config",
                    str(config),
                    "--output",
                    str(dataset),
                ],
                [
                    "scripts.audit_flight_teacher_coverage",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(coverage),
                ],
            ]
            for step, command in enumerate(commands):
                with (run / f"{role}-{step}.log").open("x", encoding="utf8") as log:
                    child = subprocess.Popen(
                        [sys.executable, "-m", *command],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW
                        | subprocess.BELOW_NORMAL_PRIORITY_CLASS,
                    )
                    write_json(
                        run / "progress.json",
                        dict(role=role, module=command[0], child_pid=child.pid),
                    )
                    code = child.wait()
                write_json(run / f"{role}-{step}-exit.json", dict(exit_code=code, command=command))
                assert code == 0
            sealed, counts = read(dataset / "result.json"), read(coverage / "result.json")
            result["roles"][role] = dict(
                dataset=str(dataset),
                coverage=str(coverage),
                indices=[s["collection_index"] for s in sources],
                dataset_manifest_sha256=sha256(dataset / "dataset.json"),
                episodes=sealed["episodes"],
                observations=sealed["decisions"],
                supervised_heads=sealed["supervised_heads"],
                counts=counts["counts"],
            )
            write_json(run / "role-provenance.json", result["roles"])
            print(json.dumps(dict(role=role, **result["roles"][role])), flush=True)
        result.update(
            status="completed_fixed_role_sealing",
            roles_changed=False,
            original_collection_status=terminal["status"],
            whole_planned_collection_completed=terminal["status"] != "failed",
            evaluation_opened=False,
            failed_or_partial_episodes_used=False,
        )
    except Exception as error:
        result.update(status="failed", error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run / "result.json", result)


if __name__ == "__main__":
    main()
