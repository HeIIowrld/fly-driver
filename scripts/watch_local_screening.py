"""Copy local WSL evidence, verify terminal files, and submit eligible additional trials."""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256


def collect_tree(source, target):
    verified = {}
    for original in sorted(source.rglob("*")):
        if not original.is_file() or original.suffix == ".partial":
            continue
        relative = original.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        digest = sha256(original)
        if sha256(destination) != digest:
            raise RuntimeError("Local collection hash mismatch: " + str(relative))
        verified[relative.as_posix()] = digest
    return verified


def verify_collected(output):
    receipt = json.loads((output / "collection-verified.json").read_text(encoding="utf-8"))
    required = {"manifest.json", "result.json", "remote-addition-config.json"}
    if not required.issubset(receipt.get("rdna4-pipeline", {})):
        raise ValueError("Completed collection lacks required pipeline evidence")
    count = 0
    for directory, files in receipt.items():
        for relative, digest in files.items():
            name = directory + "/" + relative
            path = PurePosixPath(name)
            if (not directory.startswith("rdna4-") or "/" in directory or ":" in name
                    or "\\" in name or path.is_absolute() or ".." in path.parts):
                raise ValueError("Unsafe collected evidence path")
            if sha256(output / name) != digest:
                raise RuntimeError("Collected evidence changed: " + name)
            count += 1
    return {"verified_files": count, "receipt_sha256": sha256(output / "collection-verified.json")}


def watch(source, output, root, wheels, auto_submit, promotion_policy=None, resume=False,
          preflight_output=None, collected_only=False):
    if collected_only and not resume:
        raise ValueError("Collected-only mode requires an existing verified collection")
    if resume:
        if not output.is_dir() or (output / "remote-addition-plan.json").exists():
            raise ValueError("Resume only an existing collector before remote promotion planning")
    else:
        output.mkdir(parents=True, exist_ok=False)
    collected = set()
    if resume and (output / "heartbeat.json").exists():
        collected.update(json.loads((output / "heartbeat.json").read_text(encoding="utf-8")).get(
            "collected_runs", []
        ))
    deadline = time.monotonic() + 12 * 3600
    runs = output if collected_only else source / "runs"
    pipeline = runs / "rdna4-pipeline"
    while time.monotonic() < deadline:
        try:
            if collected_only:
                write_json(output / "collection-rechecked.json", verify_collected(output))
            # Small live status files; large traces/checkpoints are copied when terminal.
            for original in (() if collected_only else runs.rglob("*.json")):
                if original.name not in ("progress.json", "lifecycle.json", "result.json",
                                          "device.json", "summary.json", "live.json",
                                          "bc-progress.json"):
                    continue
                destination = output / original.relative_to(runs)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
            progress = runs / "rdna4-screening-stage1/progress.json"
            if not collected_only and progress.exists():
                for row in json.loads(progress.read_text(encoding="utf-8"))["completed"]:
                    key = f"{row['arm']}/seed-{row['seed']}"
                    if key not in collected:
                        src = runs / "rdna4-screening-stage1" / key
                        receipt = collect_tree(src, output / "rdna4-screening-stage1" / key)
                        write_json(output / f"verified-{row['arm']}-{row['seed']}.json", receipt)
                        collected.add(key)
            terminal = pipeline / "result.json"
            if terminal.exists():
                if not collected_only:
                    receipt = {}
                    for directory in runs.glob("rdna4-*"):
                        if directory.is_dir():
                            receipt[directory.name] = collect_tree(
                                directory, output / directory.name
                            )
                    write_json(output / "collection-verified.json", receipt)
                result = json.loads(terminal.read_text(encoding="utf-8"))
                remote_config = output / "rdna4-pipeline/remote-addition-config.json"
                if auto_submit and result.get("candidates") and remote_config.exists():
                    if preflight_output is not None:
                        from scripts.verify_remote_preflight import verify_preflight

                        proof = verify_preflight(preflight_output, root)
                        while proof is None:
                            if time.monotonic() >= deadline:
                                raise RuntimeError("GPU preflight exceeded collector deadline")
                            write_json(output / "heartbeat.json", {
                                "state": "waiting_remote_preflight",
                                "updated_utc": datetime.now(timezone.utc).isoformat(),
                                "verified": True, "candidates": result["candidates"],
                                "preflight_output": str(preflight_output),
                            })
                            time.sleep(30)
                            proof = verify_preflight(preflight_output, root)
                        write_json(output / "remote-preflight-verified.json", proof)
                    launch_settings = {
                        "partition": "partition1", "node": "auto", "time_limit": "00:30:00",
                        "array_concurrency": 1,
                    }
                    selected = result["candidates"]
                    policy_hash = None
                    if promotion_policy is not None:
                        from scripts.prepare_remote_main import prepare_main

                        policy = json.loads(promotion_policy.read_text(encoding="utf-8"))
                        main = prepare_main(
                            json.loads(remote_config.read_text(encoding="utf-8")), policy
                        )
                        remote_config = output / "main-training-config.json"
                        write_json(remote_config, main)
                        launch_settings = policy["launch"]
                        selected = main["main_training"]["selected"]
                        policy_hash = sha256(promotion_policy)
                    # Verify actual actor source against the completed local manifest.
                    manifest = json.loads(
                        (output / "rdna4-pipeline/manifest.json").read_text(encoding="utf-8")
                    )
                    for name, digest in manifest["source_sha256"].items():
                        if name.startswith("src/") and sha256(root / name) != digest:
                            raise RuntimeError("Actor source changed after screening: " + name)
                    git = ["git", "-c", "safe.directory=" + root.as_posix(), "-C", str(root)]
                    status = subprocess.check_output(git + ["status", "--porcelain"], text=True)
                    if status.strip():
                        raise RuntimeError("Commit reviewed source before automatic remote sealing")
                    revision = subprocess.check_output(
                        git + ["rev-parse", "HEAD"], text=True
                    ).strip()
                    remote_output = root / "runs" / ("remote-rdna4-added-" +
                                    datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
                    write_json(output / "remote-addition-plan.json", {
                        "source_revision": revision, "output": str(remote_output),
                        "candidates": result["candidates"],
                        "main_selected": selected, "promotion_policy_sha256": policy_hash,
                        "launch_settings": launch_settings,
                        "existing_jobs_unchanged": [265295, 265296],
                    })
                    command = [sys.executable, "-m", "scripts.remote_gpu", "launch",
                               "--config", str(remote_config), "--wheels", str(wheels),
                               "--output", str(remote_output), "--partition",
                               launch_settings["partition"], "--node", launch_settings["node"],
                               "--time", launch_settings["time_limit"], "--array-concurrency",
                               str(launch_settings["array_concurrency"]),
                               "--mode", "campaign-array"]
                    with (output / "remote-addition-launch.log").open("w") as stream:
                        subprocess.run(command, cwd=root, stdout=stream,
                                       stderr=subprocess.STDOUT, check=True)
                write_json(output / "heartbeat.json", {
                    "state": "completed" if result["status"] == "completed" else "failed",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "verified": True, "candidates": result.get("candidates", []),
                    "remote_existing_jobs_unchanged": [265295, 265296],
                })
                return
            write_json(output / "heartbeat.json", {
                "state": "watching_local_screening", "updated_utc":
                datetime.now(timezone.utc).isoformat(), "collected_runs": sorted(collected),
                "remote_existing_jobs_unchanged": [265295, 265296],
                "promotion_policy": str(promotion_policy) if promotion_policy else None,
            })
        except Exception as error:
            write_json(output / "heartbeat.json", {
                "state": "failed", "updated_utc": datetime.now(timezone.utc).isoformat(),
                "reason": type(error).__name__, "message": str(error),
            })
            raise
        time.sleep(30)
    raise RuntimeError("Local watcher exceeded twelve-hour limit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--auto-submit", action="store_true")
    parser.add_argument("--promotion-policy", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-output", type=Path,
                        help="Require verified collected GPU probe evidence before submission")
    parser.add_argument("--collected-only", action="store_true",
                        help="Recheck existing collection hashes and resume without WSL access")
    args = parser.parse_args()
    watch(args.source, args.output, args.root, args.wheels, args.auto_submit,
          args.promotion_policy, args.resume, args.preflight_output, args.collected_only)
