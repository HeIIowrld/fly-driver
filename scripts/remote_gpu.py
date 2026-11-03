"""Seal, upload and submit one GPU job together with its receipt-gated local watcher."""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import uuid
import zipfile
from datetime import UTC, datetime
from email.parser import Parser
from pathlib import Path, PurePosixPath

import paramiko

from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from scripts.slurm_credentials import read_setting

ROOT = Path(__file__).resolve().parents[1]
BASE_ID = "64e6ad82752ad7b2cc05b3a8e5bbb2231f68b84c7341abab22fa8451da3c1233"
BASE_PYTHON = "/mnt/data1/dsl01/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"


def remote_path(value):
    path = PurePosixPath(value)
    if path.parent != PurePosixPath("/data1/dsl01/campaigns") or not re.fullmatch(
        r"\.mecar-flybrain-[a-z0-9-]+", path.name
    ):
        raise ValueError("Expected one owned campaigns/.mecar-flybrain-* directory")
    return str(path)


def connect():
    password = read_setting("MECAR_SSH_PASSWORD")
    if not password:
        raise RuntimeError("Configured SSH credential is unavailable")
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys(str(Path.home() / ".ssh/known_hosts"))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    ssh.connect(
        read_setting("MECAR_SSH_HOST", "afsd.iptime.org"),
        port=int(read_setting("MECAR_SSH_MASTER_PORT", "222")),
        username=read_setting("MECAR_SSH_USERNAME", "dsl01"),
        password=password,
        allow_agent=False,
        look_for_keys=False,
        timeout=20,
        auth_timeout=30,
    )
    ssh.get_transport().set_keepalive(30)
    return ssh


def execute(ssh, command):
    _, stdout, stderr = ssh.exec_command(command)
    output, error = stdout.read().decode(), stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"Remote operation failed ({code}): {error.strip()}")
    return output.strip()


def seal(output, config, wheels, include_wheels=False):
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    config = {**config, "source_revision": revision}
    entries = {}
    for directory in ("src", "scripts", "configs", "docs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".sh", ".json", ".md"):
                if "__pycache__" not in path.parts:
                    entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ("README.md", "AGENTS.md", "pyproject.toml", "references.json"):
        entries[name] = (ROOT / name).read_bytes()
    entries["input/train.json"] = json.dumps(config, indent=2).encode()
    wheel_paths = sorted(wheels.glob("*.whl"))
    if not wheel_paths:
        raise ValueError("Offline Linux wheel directory is empty")
    wheel_manifest = []
    for path in wheel_paths:
        if path.is_symlink():
            raise ValueError("Wheel symlink rejected")
        with zipfile.ZipFile(path) as wheel:
            metadata_name = next(
                name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(wheel.read(metadata_name).decode())
        wheel_manifest.append(
            {
                "filename": path.name,
                "distribution": metadata["Name"],
                "version": metadata["Version"],
                "sha256": sha256(path),
            }
        )
        if include_wheels:
            entries["input/wheels/" + path.name] = path.read_bytes()
    entries["input/wheel-manifest.json"] = json.dumps(wheel_manifest, indent=2).encode()
    manifest = {
        "source_revision": revision,
        "files_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in entries.items()},
        "created_utc": datetime.now(UTC).isoformat(),
        "base_runtime": BASE_ID,
        "config": config,
    }
    entries["stage-manifest.json"] = json.dumps(manifest, indent=2).encode()
    archive = output / "campaign.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for key, value in entries.items():
            if PurePosixPath(key).is_absolute() or ".." in PurePosixPath(key).parts:
                raise ValueError("Unsafe stage member")
            bundle.writestr(key, value)
    manifest["archive_sha256"] = sha256(archive)
    write_json(output / "stage-manifest.json", manifest)
    return manifest


def start_watcher(output, remote, jobs):
    plan = {
        "schema": "flybrain/ephemeral-v1",
        "wait_for_jobs": jobs,
        "batches": [{"port": 222, "paths": [remote]}],
        "cleanup_policy": "collect source/inputs/logs/results, verify, delete terminal owned run",
    }
    plan_path = output / "collection-plan.json"
    write_json(plan_path, plan)
    command = [
        sys.executable,
        "-m",
        "scripts.manage_slurm_ephemeral",
        "--plan",
        str(plan_path),
        "--output",
        str(output / "collection"),
        "--watch",
        "--detach",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def slurm_options(args, config, compute, exports, mode=None, after_job=None):
    mode = mode or args.mode
    dependency = after_job or args.after_job
    if dependency and not re.fullmatch(r"[0-9]+", dependency):
        raise ValueError("Invalid dependency job ID")
    merge = mode == "campaign-merge"
    array = mode == "campaign-array"
    options = [
        "sbatch",
        "--parsable",
        f"--partition={args.partition}",
        "--account=ug",
        "--qos=normal",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={1 if merge else max(4, int(config.get('num_envs', 1)))}",
        "--mem=2G" if merge else "--mem=24G",
        "--time=00:05:00" if merge else f"--time={args.time}",
        "--job-name=mecar-flybrain-" + mode,
        "--chdir=" + compute,
        "--output=" + compute + ("/slurm-%A_%a.out" if array else "/slurm-%j.out"),
        "--error=" + compute + ("/slurm-%A_%a.err" if array else "/slurm-%j.err"),
        "--export=" + exports,
        "--wrap=exec " + compute + "/scripts/run_gpu.sh",
    ]
    if args.node != "auto":
        options.append("--nodelist=" + args.node)
    if not merge:
        options.append("--gres=gpu:1")
    if array:
        count = len(config["campaign"]["seeds"]) * len(config["campaign"]["arms"])
        if not count:
            raise ValueError("Empty campaign array")
        concurrency = int(getattr(args, "array_concurrency", 1))
        if concurrency < 1:
            raise ValueError("Array concurrency must be positive")
        options.append(f"--array=0-{count - 1}%{min(concurrency, count)}")
    if dependency:
        options.append("--dependency=afterany:" + dependency)
    return options


def launch(args):
    output = args.output.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = seal(output, config, args.wheels, args.include_wheels)
    remote = remote_path(
        "/data1/dsl01/campaigns/.mecar-flybrain-"
        + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-")
        + uuid.uuid4().hex[:8]
    )
    compute = remote.replace("/data1/", "/mnt/data1/", 1)
    receipt = {
        "remote_run": remote,
        "compute_run": compute,
        "archive_sha256": manifest["archive_sha256"],
        "jobs": [],
        "mode": args.mode,
    }
    created = False
    try:
        with connect() as ssh:
            execute(
                ssh,
                "test ! -e " + shlex.quote(remote) + " && mkdir -m 700 -- " + shlex.quote(remote),
            )
            created = True
            receipt["state"] = "remote_tree_created"
            write_json(output / "submission.json", receipt)
            with ssh.open_sftp() as sftp:
                sftp.put(
                    str(output / "campaign.zip"), remote + "/campaign.zip.partial", confirm=True
                )
                sftp.rename(remote + "/campaign.zip.partial", remote + "/campaign.zip")
            extract_code = (
                "import zipfile; z=zipfile.ZipFile('campaign.zip'); "
                "assert all(not n.startswith('/') and '..' not in n.split('/') "
                "for n in z.namelist()); z.extractall('.')"
            )
            execute(
                ssh,
                f"cd {shlex.quote(remote)} && "
                f"test \"$(sha256sum campaign.zip | cut -d' ' -f1)\" = "
                f"{shlex.quote(manifest['archive_sha256'])} && python3 -c "
                + shlex.quote(extract_code)
                + " && chmod 700 scripts/run_gpu.sh",
            )
            exports = (
                f"ALL,FLYBRAIN_ROOT={compute},FLYBRAIN_BASE_PYTHON={BASE_PYTHON},"
                f"FLYBRAIN_BASE_ENV=/mnt/data1/dsl01/opt/mecar-runtime-cache/environments/{BASE_ID},"
                "FLYBRAIN_CACHE=/mnt/data1/dsl01/opt/mecar-runtime-cache,"
                f"FLYBRAIN_MODE={args.mode}"
            )
            options = slurm_options(args, config, compute, exports)
            result = execute(ssh, " ".join(shlex.quote(value) for value in options))
            job = result.split(";", 1)[0]
            if not re.fullmatch(r"[0-9]+", job):
                raise RuntimeError("Unexpected Slurm submission receipt")
            receipt.update(
                state="submitted",
                jobs=[job],
                partition=args.partition,
                node=args.node,
                after_job=args.after_job,
                time_limit=args.time,
            )
            write_json(output / "submission.json", receipt)
            if args.mode == "campaign-array":
                merge_options = slurm_options(
                    args,
                    config,
                    compute,
                    exports.replace("FLYBRAIN_MODE=campaign-array", "FLYBRAIN_MODE=campaign-merge"),
                    mode="campaign-merge",
                    after_job=job,
                )
                merged = execute(ssh, " ".join(shlex.quote(value) for value in merge_options))
                merge_job = merged.split(";", 1)[0]
                if not re.fullmatch(r"[0-9]+", merge_job):
                    raise RuntimeError("Unexpected Slurm merge receipt")
                receipt["jobs"].append(merge_job)
                receipt["array_tasks"] = [
                    {"index": index, "seed": seed, "arm": arm["name"]}
                    for index, (seed, arm) in enumerate(
                        (seed, arm)
                        for seed in config["campaign"]["seeds"]
                        for arm in config["campaign"]["arms"]
                    )
                ]
                receipt["merge_job"] = merge_job
                receipt["array_concurrency"] = min(
                    int(getattr(args, "array_concurrency", 1)), len(receipt["array_tasks"])
                )
                write_json(output / "submission.json", receipt)
    except Exception as error:
        receipt.update(state="submission_failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "submission.json", receipt)
        raise
    finally:
        if created:
            start_watcher(output, remote, receipt["jobs"])
    print(json.dumps(receipt, indent=2), flush=True)


def status(args):
    receipt = json.loads((args.output / "submission.json").read_text())
    job = receipt["jobs"][0]
    if any(not re.fullmatch(r"[0-9]+", value) for value in receipt["jobs"]):
        raise ValueError("Invalid job ID")
    with connect() as ssh:
        queue = execute(
            ssh, "squeue --array -h -j " + ",".join(receipt["jobs"]) + " -o '%i|%j|%T|%R'"
        )
        print("scheduler:", queue or "terminal/not in queue", flush=True)
        with ssh.open_sftp() as sftp:
            root = receipt["remote_run"]
            names = [
                "output/gpu-probe.json",
                "output/training/device.json",
                "output/training/progress.json",
                "output/training/result.json",
                "output/job-exit.json",
                "output/campaign/progress.json",
                "output/campaign/result.json",
                "output/merged/result.json",
            ]
            for task in receipt.get("array_tasks", []):
                prefix = f"output/task-{task['index']}"
                names.extend(
                    [
                        prefix + "/gpu-probe.json",
                        prefix + "/job-exit.json",
                        prefix + "/campaign/progress.json",
                        prefix
                        + f"/campaign/{task['arm']}/seed-{task['seed']}/training/progress.json",
                    ]
                )
            for name in names:
                try:
                    with sftp.open(root + "/" + name) as stream:
                        payload = stream.read().decode()
                        if name.endswith("/campaign/progress.json"):
                            progress = json.loads(payload)
                            payload = json.dumps(
                                {
                                    "current": progress["current"],
                                    "completed": [
                                        {
                                            key: row.get(key)
                                            for key in (
                                                "arm",
                                                "seed",
                                                "status",
                                                "mean_progress_gain",
                                                "reason",
                                            )
                                        }
                                        for row in progress["completed"]
                                    ],
                                },
                                indent=2,
                            )
                        elif name.endswith("/training/progress.json"):
                            progress = json.loads(payload)
                            payload = json.dumps(
                                {
                                    key: progress.get(key)
                                    for key in (
                                        "update",
                                        "interactions",
                                        "loss",
                                        "device",
                                        "wall_seconds",
                                        "transitions_per_second",
                                        "approx_kl",
                                    )
                                },
                                indent=2,
                            )
                        print(name, payload, flush=True)
                except OSError:
                    pass
            for suffix in ("out", "err"):
                try:
                    with sftp.open(root + f"/slurm-{job}.{suffix}") as stream:
                        size = stream.stat().st_size
                        stream.seek(max(0, size - 4000))
                        print(f"log.{suffix}", stream.read().decode(errors="replace"), flush=True)
                except OSError:
                    pass


def adjust_array(args):
    """Change only an owned array's throttle; preserve its sealed inputs and age."""
    output = args.output.resolve()
    receipt = json.loads((output / "submission.json").read_text())
    if receipt.get("mode") != "campaign-array" or not receipt.get("array_tasks"):
        raise ValueError("Expected an existing campaign array receipt")
    job = receipt["jobs"][0]
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("Invalid array job ID")
    requested = int(args.array_concurrency)
    if requested < 1:
        raise ValueError("Array concurrency must be positive")
    expected = remote_path(receipt["remote_run"]).replace("/data1/", "/mnt/data1/", 1)
    username = read_setting("MECAR_SSH_USERNAME", "dsl01")
    with connect() as ssh:
        before = execute(ssh, "scontrol show job " + job)
        fields = dict(re.findall(r"(?:^|\s)([A-Za-z]+)=([^\s]+)", before))
        if (fields.get("ArrayJobId") != job or fields.get("WorkDir") != expected
                or fields.get("JobName") != "mecar-flybrain-campaign-array"
                or not fields.get("UserId", "").startswith(username + "(")
                or fields.get("QOS") != "normal" or fields.get("Account") != "ug"):
            raise ValueError("Array ownership / source directory / account check failed")
        quota = execute(ssh, "sacctmgr -nP show qos normal format=Name,MaxTRESPU")
        limit = re.search(r"(?:^|[|,])gres/gpu=(\d+)", quota)
        if limit is None or requested > int(limit[1]):
            raise ValueError("Concurrency exceeds the verified normal QoS GPU limit")
        concurrency = min(requested, len(receipt["array_tasks"]))
        execute(ssh, f"scontrol update JobId={job} ArrayTaskThrottle={concurrency}")
        after = execute(ssh, "scontrol show job " + job)
        if f"ArrayTaskThrottle={concurrency}" not in after:
            raise RuntimeError("Scheduler did not confirm the requested array throttle")
    path = output / "scheduler-adjustments.json"
    history = json.loads(path.read_text()) if path.exists() else []
    history.append({
        "utc": datetime.now(UTC).isoformat(), "job": job,
        "requested_array_concurrency": concurrency, "verified_qos": quota,
        "before": before, "after": after,
        "sealed_archive_sha256": receipt["archive_sha256"],
        "scope": "only task throttle changed; task count, inputs, source and budgets preserved",
    })
    write_json(path, history)
    print(json.dumps({"job": job, "array_concurrency": concurrency,
                      "receipt": str(path)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("launch")
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--wheels", type=Path, required=True)
    submit.add_argument("--output", type=Path, required=True)
    submit.add_argument("--partition", default="partition1")
    submit.add_argument("--node", default="hpc-stat1", help="node name, or auto for no node pin")
    submit.add_argument("--time", default="03:00:00")
    submit.add_argument("--array-concurrency", type=int, default=1)
    submit.add_argument(
        "--mode", choices=["probe", "training", "campaign", "campaign-array"], default="training"
    )
    submit.add_argument("--after-job", help="start after this Slurm job terminates (afterany)")
    submit.add_argument(
        "--include-wheels", action="store_true", help="upload all wheel bytes for offline setup"
    )
    inspect = commands.add_parser("status")
    inspect.add_argument("--output", type=Path, required=True)
    adjust = commands.add_parser("adjust-array")
    adjust.add_argument("--output", type=Path, required=True)
    adjust.add_argument("--array-concurrency", type=int, required=True)
    args = parser.parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "adjust-array":
        adjust_array(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
