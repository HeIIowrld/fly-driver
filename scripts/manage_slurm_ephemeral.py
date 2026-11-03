# Adapted from the reviewed MECar-Simulation operational utility (2026-10-05).
"""Local SSH lifecycle manager; credential values never enter plans or receipts."""

import argparse
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import paramiko

from scripts.slurm_credentials import read_setting, ssh_password
from scripts.slurm_ephemeral_remote import allowed_name, normalize

REMOTE_SOURCE = Path(__file__).with_name("slurm_ephemeral_remote.py").read_text(encoding="utf-8")
PORTS = (222, 223, 224)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    part = path.with_suffix(path.suffix + ".partial")
    part.write_text(json.dumps(value, indent=2), encoding="utf-8")
    part.replace(path)


class Session:
    def __init__(self, host, username, password):
        self.host, self.username, self.password = host, username, password

    def connect(self, port):
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys(str(Path.home() / ".ssh/known_hosts"))
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        ssh.connect(
            self.host,
            port=port,
            username=self.username,
            password=self.password,
            allow_agent=False,
            look_for_keys=False,
            timeout=20,
            auth_timeout=30,
        )
        ssh.get_transport().set_keepalive(30)
        return ssh

    def call(self, port, request):
        command = "python3 -c " + shlex.quote(REMOTE_SOURCE)
        if port != 222:
            partition, node = {223: ("jobs", "hpc"), 224: ("partition1", "hpc-stat1")}[port]
            command = (
                f"srun --partition={partition} --nodelist={node} --account=ug --qos=normal "
                "--nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 "
                "--chdir=/tmp --job-name=mecar-housekeep " + command
            )
        with self.connect(222) as ssh:
            stdin, stdout, stderr = ssh.exec_command(command)
            stdin.write(json.dumps(request))
            stdin.channel.shutdown_write()
            output, error = stdout.read(), stderr.read()
            code = stdout.channel.recv_exit_status()
            if code:
                raise RuntimeError(
                    f"remote operation {request['operation']} on port {port}: "
                    + error.decode(errors="replace")
                )
            return json.loads(output)

    def inventory(self):
        result = {"222": self.call(222, {"operation": "inventory"})}
        for port in (223, 224):
            with self.connect(port) as ssh, ssh.open_sftp() as sftp:
                homes = []
                for home in ("/data1/dsl01", "/mnt/data1/dsl01", "/home/dsl01"):
                    try:
                        sftp.stat(home)
                        homes.append(home)
                    except OSError:
                        continue
                if not homes:
                    raise ValueError("cannot establish owned home on compute node")
                uid = sftp.stat(homes[0]).st_uid
                candidates = []
                parents = [
                    home + suffix
                    for home in homes
                    for suffix in (
                        "",
                        "/campaigns",
                        "/uploads",
                        "/tmp",
                        "/logs",
                        "/backups",
                        "/releases",
                    )
                ]
                for parent in parents + ["/tmp", "/var/tmp"]:
                    try:
                        for entry in sftp.listdir_attr(parent):
                            if (
                                allowed_name(entry.filename)
                                and entry.st_uid == uid
                                and not stat.S_ISLNK(entry.st_mode)
                                and not entry.filename.startswith(".mecar-transfer-")
                            ):
                                candidates.append(parent + "/" + entry.filename)
                    except OSError:
                        continue
                result[str(port)] = {
                    "host": {223: "hpc", 224: "hpc-stat1"}[port],
                    "uid": uid,
                    "candidates": sorted(set(candidates)),
                    "jobs": result["222"]["jobs"],
                    "inspection": "SFTP metadata only",
                }
        return result

    def download(self, port, remote, local, progress):
        # Bounded transfers resume from the last flushed byte after disconnects.
        # Limit prefetch requests rather than queuing a multi-GB file all at once.
        while True:
            offset = local.stat().st_size if local.exists() else 0
            with self.connect(port) as ssh, ssh.open_sftp() as sftp:
                total = sftp.stat(remote).st_size
                if offset > total:
                    raise ValueError("partial download is larger than the remote archive")
                if offset == total:
                    progress(total, total)
                    return
                end = min(total, offset + 256 * 1024 * 1024)
                with sftp.open(remote, "rb") as source, local.open("ab") as destination:
                    source.seek(offset)
                    source.prefetch(file_size=end, max_concurrent_requests=32)
                    while offset < end:
                        block = source.read(min(1024 * 1024, end - offset))
                        if not block:
                            raise EOFError("incomplete remote archive")
                        destination.write(block)
                        offset += len(block)
                        progress(offset, total)
                    destination.flush()
                    os.fsync(destination.fileno())

    def cross_node_guard(self, paths):
        normalized = {p.replace("/mnt/data1/", "/data1/") for p in paths}
        for port, info in self.inventory().items():
            matching = [
                p for p in info["candidates"] if p.replace("/mnt/data1/", "/data1/") in normalized
            ]
            if matching and int(port) == 222:
                check = self.call(int(port), {"operation": "check", "paths": matching})
                if check["blockers"]:
                    raise RuntimeError(f"active references on {info['host']}: {check['blockers']}")
            elif matching:
                wanted = [normalize(p) for p in matching]
                busy = [
                    j["id"]
                    for j in info["jobs"]
                    if any(p in normalize(j["detail"]) for p in wanted)
                ]
                with self.connect(int(port)) as ssh, ssh.open_sftp() as sftp:
                    for proc in sftp.listdir_attr("/proc"):
                        if not proc.filename.isdigit() or proc.st_uid != info["uid"]:
                            continue
                        data = ""
                        for suffix in ("cmdline", "environ"):
                            try:
                                with sftp.open(f"/proc/{proc.filename}/{suffix}", "rb") as f:
                                    data += f.read().decode(errors="replace")
                            except OSError:
                                pass
                        try:
                            data += sftp.readlink(f"/proc/{proc.filename}/cwd")
                        except OSError:
                            pass
                        try:
                            descriptors = sftp.listdir(f"/proc/{proc.filename}/fd")
                            for descriptor in descriptors:
                                try:
                                    data += sftp.readlink(f"/proc/{proc.filename}/fd/{descriptor}")
                                except OSError:
                                    pass
                        except OSError:
                            pass
                        if any(p in normalize(data) for p in wanted):
                            busy.append("process " + proc.filename)
                if busy:
                    raise RuntimeError(f"active references on {info['host']}: {busy}")


def clean_batch(session, batch, output, index):
    paths, port = batch["paths"], batch["port"]
    receipt_path = output / f"batch-{index:02d}-transfer.json"
    archive = output / f"batch-{index:02d}-results.tar.gz"
    deletion = output / f"batch-{index:02d}-deleted.json"
    if deletion.exists():
        return
    session.cross_node_guard(paths)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["paths"] != paths:
            raise ValueError("saved receipt belongs to a different plan")
    else:
        print(f"Archiving batch {index}: {len(paths)} terminal MECar targets", flush=True)
        receipt = session.call(port, {"operation": "archive", "paths": paths})
        write_json(receipt_path, receipt)
    if not archive.exists() or digest(archive) != receipt["archive_sha256"]:
        last = [0.0]

        def progress(done, total):
            if time.monotonic() - last[0] > 30 or done == total:
                print(f"Batch {index}: downloaded {done}/{total} bytes", flush=True)
                last[0] = time.monotonic()

        part = archive.with_suffix(".partial")
        session.download(port, receipt["archive"], part, progress)
        if digest(part) != receipt["archive_sha256"]:
            part.unlink()
            raise ValueError("download SHA-256 mismatch; remote data retained")
        part.replace(archive)
    # The compressed stream and embedded source manifest must both be readable.
    with tarfile.open(archive, "r:gz") as tar:
        manifest = json.load(tar.extractfile("transfer-manifest.json"))
        if manifest["paths"] != paths or manifest["tree_sha256"] != receipt["tree_sha256"]:
            raise ValueError("embedded manifest mismatch")
    verified = digest(archive)
    write_json(
        output / f"batch-{index:02d}-verified.json",
        {
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "sha256": verified,
            "archive_bytes": archive.stat().st_size,
            "paths": paths,
        },
    )
    session.cross_node_guard(paths)
    print(f"Batch {index}: local archive verified; deleting exact remote targets", flush=True)
    result = session.call(
        port,
        {
            "operation": "delete_verified",
            "paths": paths,
            "receipt": receipt,
            "local_verified_sha256": verified,
        },
    )
    if not result["verified_absent"]:
        raise RuntimeError("remote deletion verification failed")
    write_json(deletion, result)
    print(f"Batch {index}: deleted and verified absent", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=read_setting("MECAR_SSH_HOST", "afsd.iptime.org"))
    parser.add_argument("--username", default=read_setting("MECAR_SSH_USERNAME", "dsl01"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--detach", action="store_true", help="start a hidden local watcher")
    parser.add_argument("--credential-stdin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    password = sys.stdin.readline().rstrip("\r\n") if args.credential_stdin else ssh_password()
    if not password:
        raise ValueError("empty credential")
    if args.detach:
        if not args.watch or not args.plan or args.credential_stdin:
            parser.error("--detach requires --watch and --plan")
        command = [
            sys.executable,
            "-m",
            "scripts.manage_slurm_ephemeral",
            "--host",
            args.host,
            "--username",
            args.username,
            "--output",
            str(output),
            "--plan",
            str(args.plan.resolve()),
            "--watch",
            "--credential-stdin",
        ]
        options = (
            {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        with (output / "watcher.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                **options,
            )
            child.stdin.write(password + "\n")
            child.stdin.close()
        write_json(
            output / "watcher-process.json",
            {
                "pid": child.pid,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "command": command,
                "credential_storage": "anonymous pipe then process memory only",
            },
        )
        print(
            f"Hidden watcher started: PID {child.pid}; status {output / 'lifecycle-status.json'}",
            flush=True,
        )
        return
    session = Session(args.host, args.username, password)
    if args.plan is None:
        inventory = session.inventory()
        write_json(output / "inventory.json", inventory)
        print(
            json.dumps(
                {
                    port: {"host": r["host"], "candidates": r["candidates"]}
                    for port, r in inventory.items()
                },
                indent=2,
            ),
            flush=True,
        )
        return
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    status_path = output / "lifecycle-status.json"
    watched = set(map(str, plan.get("wait_for_jobs", [])))
    while True:
        try:
            if watched:
                status = session.call(222, {"operation": "status"})
                active = [j for j in status["jobs"] if j["id"].split("_")[0] in watched]
                impossible = [
                    j["id"] for j in active if j["reason"].strip("()") == "DependencyNeverSatisfied"
                ]
                if impossible:
                    session.call(222, {"operation": "cancel_failed_dependency", "jobs": impossible})
                if active:
                    if not args.watch:
                        raise RuntimeError("jobs still active; use --watch")
                    write_json(
                        status_path,
                        {
                            "state": "waiting_for_jobs",
                            "checked_at_utc": datetime.now(UTC).isoformat(),
                            "active": [
                                {k: v for k, v in j.items() if k != "detail"} for j in active
                            ],
                        },
                    )
                    print(
                        f"Waiting for {len(active)} Slurm tasks; no active files deleted",
                        flush=True,
                    )
                    time.sleep(30)
                    continue
            write_json(
                status_path,
                {
                    "state": "collecting_and_cleaning",
                    "started_at_utc": datetime.now(UTC).isoformat(),
                },
            )
            for index, batch in enumerate(plan["batches"]):
                clean_batch(session, batch, output, index)
            write_json(output / "inventory-after.json", session.inventory())
            write_json(
                status_path,
                {
                    "state": "complete",
                    "finished_at_utc": datetime.now(UTC).isoformat(),
                    "batches": len(plan["batches"]),
                },
            )
            print("Results collected, verified, and remote campaign files removed.", flush=True)
            return
        except Exception as error:
            write_json(
                status_path,
                {
                    "state": "retrying" if args.watch else "failed",
                    "checked_at_utc": datetime.now(UTC).isoformat(),
                    "error": str(error),
                },
            )
            if not args.watch:
                raise
            print(
                f"Collection/cleanup paused: {type(error).__name__}; retrying in 30 seconds",
                flush=True,
            )
            time.sleep(30)


if __name__ == "__main__":
    main()
