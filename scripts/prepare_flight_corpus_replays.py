"""Pipeline complete audited sensory histories without fitting, role changes or controls."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.win_process_receipt import ProcessReceipt


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    collection, audits = Path(plan["collection"]), Path(plan["audits"])
    original_plan = read(collection / "collection-plan.json")
    original_plan_hash = sha256(collection / "collection-plan.json")
    assert original_plan["revision"] == 4
    entries = [e for e in original_plan["episodes"] if e["role"] in ("train", "development")]
    assert len(entries) == 20 and len({e["index"] for e in entries}) == 20
    snapshot = Path(plan["source_snapshot"])
    for name in (
        "scripts/replay_virtual_body.py",
        "scripts/replay_flyvis_cameras.py",
        "scripts/audit_flyvis_camera_replay.py",
        "src/flybrain_drive/virtual_body.py",
        "src/flybrain_drive/flyvis_camera.py",
    ):
        assert sha256(name) == sha256(snapshot / name)
    handles = {}
    for role, pid, directory in (
        ("collection", plan["collection_pid"], collection),
        ("audit", plan["audit_pid"], audits),
    ):
        try:
            handles[role] = ProcessReceipt(pid)
        except OSError:
            assert (directory / "result.json").exists(), (
                f"{role}: missing process and terminal result"
            )
    run = begin_run(
        args.output,
        dict(
            plan=plan,
            plan_sha256=sha256(args.plan),
            collection_plan_sha256=original_plan_hash,
            program_sha256=sha256(__file__),
            author_checkpoint_sha256=sha256(Path(plan["model"]) / "best_chkpt"),
            original_role_entries=entries,
            evaluation_data_opened=False,
        ),
    )
    shutil.copyfile(__file__, run / "executed-program.py")
    shutil.copyfile(args.plan, run / "replay-plan.json")
    result = dict(
        status="running",
        body_episodes=[],
        episodes=[],
        optimizer_updates=0,
        actuator_messages_sent=0,
        ui_actions_sent=0,
        goal_achieved=False,
    )
    deadline = time.monotonic() + 3600

    def active(role, directory):
        if (directory / "result.json").exists():
            return False
        handle = handles.get(role)
        assert handle is not None and handle.snapshot()["active"], f"{role} exited without result"
        assert time.monotonic() < deadline, "Finite preparation dependency wait expired"
        return True

    def call(module, arguments, log_path, gpu=False):
        if gpu:
            bootstrap = (
                f"import sys,site,runpy; p={plan['gpu_site']!r}; site.addsitedir(p); "
                f'sys.path.insert(0,p); runpy.run_module({module!r},run_name="__main__")'
            )
            command = [plan["base_python"], "-c", bootstrap, *arguments]
        else:
            command = [sys.executable, "-m", module, *arguments]
        with log_path.open("x", encoding="utf8") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            write_json(
                run / "progress.json",
                dict(
                    module=module,
                    child_pid=child.pid,
                    prepared_body_episodes=len(result["body_episodes"]),
                    completed_neural_episodes=len(result["episodes"]),
                    log=str(log_path),
                ),
            )
            print(json.dumps(dict(module=module, child_pid=child.pid)), flush=True)
            code = child.wait()  # Observe the same original process; no timeout resubmission.
        write_json(log_path.with_suffix(".exit.json"), dict(command=command, exit_code=code))
        assert code == 0, f"{module}: child exited{code}; no automatic retry"

    try:
        for entry in entries:
            while True:
                assert sha256(collection / "collection-plan.json") == original_plan_hash
                partial = audits / "partial-result.json"
                available = read(partial)["episodes"] if partial.exists() else []
                selected = [e for e in available if e["index"] == entry["index"]]
                if selected:
                    break
                if not active("audit", audits):
                    break
                write_json(
                    run / "progress.json",
                    dict(
                        stage="waiting_for_audited_body_source",
                        index=entry["index"],
                        prepared=len(result["body_episodes"]),
                        dependency_handle=handles["audit"].snapshot(),
                    ),
                )
                time.sleep(1)
            if not selected:
                break
            source = selected[0]
            assert source["role"] == entry["role"]
            recording = Path(source["recording"])
            assert read(recording / "result.json")["status"] == "completed"
            assert sha256(Path(source["raw_audit"]) / "result.json") == source["raw_result_sha256"]
            assert sha256(Path(source["targets"]) / "result.json") == source["target_result_sha256"]
            episode = run / f"episode-{entry['index']:02d}-{entry['role']}"
            episode.mkdir()
            body = episode / "body"
            call(
                "scripts.replay_virtual_body",
                [
                    "--recording",
                    str(recording),
                    "--contract",
                    plan["body_contract"],
                    "--output",
                    str(body),
                ],
                episode / "body.log",
            )
            assert read(body / "result.json")["status"] == "completed"
            result["body_episodes"].append(
                dict(
                    **source, body_replay=str(body), body_result_sha256=sha256(body / "result.json")
                )
            )
            write_json(run / "partial-result.json", result)
        # No GPU competition with this task's live capture; CPU body preparation above is read-only.
        while active("collection", collection):
            write_json(
                run / "progress.json",
                dict(
                    stage="waiting_for_collection_before_gpu",
                    prepared=len(result["body_episodes"]),
                    dependency_handle=handles["collection"].snapshot(),
                ),
            )
            time.sleep(1)
        terminal = read(collection / "result.json")
        result["original_collection_status"] = terminal["status"]
        result["collection_terminal_sha256"] = sha256(collection / "result.json")
        for source in result["body_episodes"]:
            recording, body = Path(source["recording"]), Path(source["body_replay"])
            episode = body.parent
            vision, audit = episode / "vision", episode / "full-vision-audit"
            call(
                "scripts.replay_flyvis_cameras",
                [
                    "--recording",
                    str(recording),
                    "--body-replay",
                    str(body),
                    "--model-dir",
                    plan["model"],
                    "--cache-root",
                    plan["cache"],
                    "--camera-config",
                    plan["camera_config"],
                    "--output",
                    str(vision),
                ],
                episode / "vision.log",
                gpu=True,
            )
            call(
                "scripts.audit_flyvis_camera_replay",
                [
                    "--replay",
                    str(vision),
                    "--body-replay",
                    str(body),
                    "--model-dir",
                    plan["model"],
                    "--cache-root",
                    plan["cache"],
                    "--full-chronological",
                    "--output",
                    str(audit),
                ],
                episode / "audit.log",
                gpu=True,
            )
            assert (
                read(vision / "result.json")["status"]
                == read(audit / "result.json")["status"]
                == "completed"
            )
            result["episodes"].append(
                dict(
                    **source,
                    scene_replay=str(vision),
                    full_audit=str(audit),
                    vision_result_sha256=sha256(vision / "result.json"),
                    audit_result_sha256=sha256(audit / "result.json"),
                )
            )
            write_json(run / "partial-result.json", result)
        whole = (
            len(result["episodes"]) == 20
            and terminal["status"] == "completed_fixed_teacher_collection_batch"
        )
        result.update(
            status="completed_complete_roles_replay"
            if whole
            else "completed_audited_prefix_replay",
            whole_preassigned_collection_prepared=whole,
            evaluation_data_opened=False,
            original_times_and_neural_state_preserved=True,
        )
    except Exception as error:
        result.update(status="failed", error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run / "result.json", result)
        for handle in handles.values():
            handle.close()


if __name__ == "__main__":
    main()
