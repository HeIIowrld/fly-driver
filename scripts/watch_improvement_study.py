"""Verify local WSL results and GPU preflight before submitting bounded main training."""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from scripts.verify_remote_preflight import verify_preflight
from scripts.watch_local_screening import collect_tree


def launcher_alive(pid):
    if pid is None:
        return True
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    # os.kill(pid, 0) is not a read-only liveness probe on Windows.
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def check_source(root, frozen):
    root = Path(root)
    for name, digest in frozen["files_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError("Study source changed before promotion: " + name)
    status = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root),
         "status", "--porcelain"], check=True, capture_output=True, text=True,
    )
    if status.stdout.strip():
        raise ValueError("Commit study changes before remote submission")


def launch(root, config, output, wheels, mode, time_limit):
    subprocess.run(
        [sys.executable, "-m", "scripts.remote_gpu", "launch", "--config", str(config),
         "--output", str(output), "--wheels", str(wheels), "--partition", "partition1",
         "--node", "auto", "--mode", mode, "--time", time_limit,
         "--array-concurrency", "2"], cwd=root, check=True,
    )
    return json.loads((output / "submission.json").read_text(encoding="utf-8"))


def watch(source, output, root, wheels, frozen, launcher_pid=None):
    source, output, root = Path(source), Path(output), Path(root)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 12 * 3600
    collected, receipts = set(), {}
    try:
        while time.monotonic() < deadline:
            for path in source.rglob("*.json"):
                if path.name not in ("live.json", "progress.json", "result.json",
                                     "bc-progress.json", "device.json", "validation.json"):
                    continue
                destination = output / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    write_json(destination, value)
                except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                    continue
            summary = source / "summary.json"
            if summary.exists():
                rows = json.loads(summary.read_text(encoding="utf-8"))["runs"]
                for row in rows:
                    # Pipeline outputs are immediate children with validated plain names.
                    name = Path(row["trial_output"]).name
                    if name in collected:
                        continue
                    if name in ("", ".", "..") or "/" in name or "\\" in name:
                        raise ValueError("Unsafe local study output name")
                    receipts[name] = collect_tree(source / name, output / name)
                    log = source / (name + ".log")
                    if log.exists():
                        (output / log.name).write_bytes(log.read_bytes())
                    collected.add(name)
                    write_json(output / "collection-verified.json", receipts)
            heartbeat = {"state": "local_running",
                         "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                         "collected_trials": sorted(collected)}
            if (source / "live.json").exists():
                heartbeat["local"] = json.loads((source / "live.json").read_text(encoding="utf-8"))
            write_json(output / "heartbeat.json", heartbeat)
            terminal = source / "result.json"
            if terminal.exists():
                result = json.loads(terminal.read_text(encoding="utf-8"))
                # Preserve every source/log/result/failed-run file before using a proposal.
                receipts["complete-study"] = collect_tree(source, output / "complete-study")
                write_json(output / "collection-verified.json", receipts)
                if result["status"] != "completed":
                    raise RuntimeError("Local improvement study failed: " + str(result))
                if not result["remote_proposal"]:
                    write_json(output / "heartbeat.json", {
                        "state": "no_replicated_candidate", "result": result,
                        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    return
                check_source(root, frozen)
                proposal = output / "complete-study/remote-proposal.json"
                config = json.loads(proposal.read_text(encoding="utf-8"))
                if len(config["campaign"]["arms"]) * len(config["campaign"]["seeds"]) > 9:
                    raise ValueError("Main-training proposal exceeds nine tasks")
                probe_output = output / "remote-preflight"
                launch(root, proposal, probe_output, wheels, "probe", "00:10:00")
                while time.monotonic() < deadline:
                    proof = verify_preflight(probe_output, root)
                    if proof is not None:
                        break
                    write_json(output / "heartbeat.json", {
                        "state": "waiting_for_remote_preflight",
                        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    time.sleep(20)
                else:
                    raise TimeoutError("Remote GPU preflight deadline")
                check_source(root, frozen)
                write_json(output / "preflight-verified.json", proof)
                main_output = output / "remote-main"
                receipt = launch(root, proposal, main_output, wheels, "campaign-array", "03:00:00")
                write_json(output / "heartbeat.json", {
                    "state": "remote_submitted", "selected": result["selected"],
                    "jobs": receipt["jobs"], "remote_output": str(main_output),
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "local_verified": True, "gpu_preflight_verified": True,
                })
                return
            if not launcher_alive(launcher_pid):
                if source.exists():
                    receipts["failed-local-study"] = collect_tree(
                        source, output / "failed-local-study"
                    )
                    write_json(output / "collection-verified.json", receipts)
                raise RuntimeError("Local WSL launcher exited without a terminal study result")
            time.sleep(20)
        raise TimeoutError("Improvement-study supervisor exceeded 12 hours")
    except Exception as error:
        write_json(output / "heartbeat.json", {
            "state": "failed", "reason": type(error).__name__, "message": str(error),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_source.read_text(encoding="utf-8"))
    watch(args.source, args.output, args.root, args.wheels, frozen, args.launcher_pid)


if __name__ == "__main__":
    main()
