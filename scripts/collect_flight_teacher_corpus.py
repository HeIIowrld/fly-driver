"""Run preassigned finite teacher leases; preserve failures and never use desktop input."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = Path(args.plan)
    plan, digest = read(config), sha256(config)
    assert plan["schema"] == "flybrain.flight.fresh_teacher_corpus.v3"
    assert (plan["duration_seconds"], plan["speed_hint_kmh"], plan["ai_steer_multiplier"]) == (
        30,
        30,
        0.9,
    )
    assert [e["index"] for e in plan["episodes"]] == list(range(24))
    assert all(e["role"] in ("train", "development", "evaluation") for e in plan["episodes"])
    assert plan["session_restarts_between_records"] == 0
    prior_root = Path(plan["prerequisite_moving_diagnostic"])
    audit_root = Path(plan["prerequisite_raw_audit"])
    prior, audit = read(prior_root / "result.json"), read(audit_root / "result.json")
    assert prior["status"] == audit["status"] == "completed"
    assert prior["camera_wait_misses"] == 0
    assert 70 <= prior["decisions"] == prior["all_four_views_valid_decisions"] <= 76
    assert audit["decisions"] == prior["decisions"] and audit["intervals_above_80ms"] == 0
    assert not any(audit["actor_replay_max_errors"].values())
    assert audit["camera_aid_state_verified"]
    for file in (
        "src/flybrain_drive/ac_csp_vision.py",
        "src/flybrain_drive/ac_csp_geometry.py",
        "src/flybrain_drive/ac_csp_retina.py",
        "src/flybrain_drive/ac_native_physics.py",
        "scripts/record_ac_engine_ai_probe.py",
    ):
        assert sha256(file) == sha256(Path(plan["source_snapshot"]) / file)
    assert shutil.disk_usage(os.environ["LOCALAPPDATA"]).free >= plan["minimum_free_local_bytes"]
    axis = read("configs/ac-axis-probe.json")
    sdk = load_vjoy_sdk(axis["dll_path"], axis["dll_sha256"])
    window_api = WindowsACWindow()
    generation = None

    def idle():
        nonlocal generation
        state, modified = read_published_json(
            Path(os.environ["LOCALAPPDATA"]) / "flybrain/csp-session/status.json"
        )
        assert 0 <= time.time() - modified <= 2
        assert state["paused"] and not state["ai_controlled"] and not state["camera_active"]
        assert state["ideal_line_shown"] is False and not state.get("pending_receipt_id")
        assert not state.get("error") and not state["in_menu"] and not state["is_online"]
        if generation is None:
            generation = state["generation"]
        assert generation == state["generation"]
        windows = window_api.discover()
        assert len(windows) == 1
        assert window_api.user.GetForegroundWindow() != windows[0].hwnd
        assert sdk.GetVJDStatus(axis["device_id"]) == 1
        with NativePhysicsReader(timeout=0.15) as reader:
            sample = reader.receive()
            assert abs(sample.fields["speedKmh"]) < 0.1
            assert sample.fields["gas"] < 0.02 and sample.fields["brake"] > 0.98
        teacher = NativeTeacherStatusReader()
        try:
            ownership = teacher.snapshot()
        finally:
            teacher.close()
        return dict(
            session=state,
            native=dict(sample.fields),
            native_teacher=ownership,
            native_owner_qualification=False,
            engine_pid=windows[0].pid,
            foreground=False,
            vjoy_free=True,
        )

    run = begin_run(
        args.output,
        dict(
            plan=plan,
            plan_sha256=digest,
            program_sha256=sha256(__file__),
            preflight_result_sha256=sha256(prior_root / "result.json"),
            preflight_audit_sha256=sha256(audit_root / "result.json"),
        ),
    )
    shutil.copyfile(__file__, run / "collection-program.py")
    shutil.copyfile(config, run / "collection-plan.json")
    result = dict(
        status="failed",
        goal_achieved=False,
        optimizer_updates=0,
        student_policy_applied=False,
        ui_actions_sent=0,
        episodes=[],
    )
    try:
        for episode in plan["episodes"]:
            assert sha256(config) == digest
            assert (
                shutil.disk_usage(os.environ["LOCALAPPDATA"]).free
                >= plan["minimum_free_local_bytes"]
            )
            before = idle()
            name = f"episode-{episode['index']:02d}-{episode['role']}"
            output = run / name
            command = [
                sys.executable,
                "-m",
                "scripts.record_ac_engine_ai_probe",
                "--config",
                plan["retina_config"],
                "--duration-seconds",
                "30",
                "--speed-hint-kmh",
                "30",
                "--ai-steer-multiplier",
                "0.9",
                "--shared-transport",
                "--image-format",
                "bmp",
                "--local-event-spool",
                "--output",
                str(output),
            ]
            if plan.get("async_pixel_precompute", False):
                command.append("--async-pixel-precompute")
            write_json(run / (name + "-before.json"), before)
            with (run / (name + ".log")).open("x", encoding="utf8") as log:
                child = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                write_json(
                    run / "progress.json",
                    dict(
                        stage=name,
                        pid=child.pid,
                        completed_episodes=result["episodes"],
                        role_policy_fixed=True,
                    ),
                )
                print(json.dumps(dict(stage=name, pid=child.pid)), flush=True)
                code = child.wait()
            receipt = read(output / "result.json")
            # Keep actual exit/outcome even if admission below fails.
            result["last_child"] = dict(path=str(output), exit_code=code, status=receipt["status"])
            assert code == 0 and receipt["status"] == "completed"
            assert receipt["ui_actions_sent"] == receipt["vjoy_messages_sent"] == 0
            assert not receipt["vision"]["worker_alive"] and not receipt["vision"]["error"]
            after = idle()
            write_json(run / (name + "-after.json"), after)
            result["episodes"].append(
                dict(
                    **episode,
                    path=str(output),
                    result_sha256=sha256(output / "result.json"),
                    decisions=receipt["decisions"],
                    child_exit_code=code,
                    camera_wait_misses=receipt["camera_wait_misses"],
                )
            )
            write_json(run / "partial-result.json", result)
        result.update(
            status="completed_fixed_teacher_collection_batch",
            all24_original_leases_completed=True,
            role_assignments_changed=False,
            observations=sum(e["decisions"] for e in result["episodes"]),
            labels_audited_or_sealed=False,
        )
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / "result.json", result)
        raise
    write_json(run / "result.json", result)


if __name__ == "__main__":
    main()
