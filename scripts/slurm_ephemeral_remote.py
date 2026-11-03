# Adapted from the reviewed MECar-Simulation operational utility (2026-10-05).
"""Private JSON-over-SSH helper: inventory, archive, then receipt-gated deletion."""

import getpass
import hashlib
import io
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

DISPOSABLE = {".venv", "venv", "__pycache__", ".cache", "jax-cache", ".pytest_cache", ".ruff_cache"}


def bases():
    homes = [Path(p) for p in ("/data1/dsl01", "/mnt/data1/dsl01", str(Path.home()))]
    values = []
    for home in homes:
        if home.exists():
            values.extend(
                home / suffix
                for suffix in ("", "campaigns", "uploads", "tmp", "logs", "backups", "releases")
            )
    values.extend([Path("/tmp"), Path("/var/tmp")])
    return {p.resolve() for p in values if p.is_dir()}


def allowed_name(name):
    return bool(re.match(r"(?i)(?:\.?mecar[-_]|20260828_gpu_lap_v\d+\.zip$)", name))


def validate(path):
    p = Path(path)
    if not p.is_absolute() or p.parent.resolve() not in bases() or not allowed_name(p.name):
        raise ValueError(f"outside explicit MECar scope: {p}")
    if p.is_symlink() or p.resolve() != p or os.path.ismount(p):
        raise ValueError(f"symlink, alias or mount target rejected: {p}")
    if p.lstat().st_uid != os.getuid():
        raise ValueError(f"target is not owned by current user: {p}")
    return p


def normalize(value):
    return value.replace("/mnt/data1/", "/data1/")


def writable_retry(function, path, error):
    """Retry deleting a verified owned member copied from a read-only wheel."""

    item = Path(path)
    if item.is_symlink():
        raise error[1]
    parent = item.parent
    os.chmod(parent, os.stat(parent).st_mode | stat.S_IWUSR | stat.S_IXUSR)
    os.chmod(item, os.stat(item).st_mode | stat.S_IWUSR)
    function(path)


def jobs():
    lines = subprocess.check_output(
        ["squeue", "--array", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%R"],
        text=True,
    ).splitlines()
    result = []
    for line in lines:
        job_id, state, reason = line.split("|", 2)
        reason = reason.strip("()")
        detail = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
        result.append({"id": job_id, "state": state, "reason": reason, "detail": detail})
    return result


def blockers(paths):
    normalized = [normalize(str(p)) for p in paths]
    result = []
    for job in jobs():
        detail = normalize(job["detail"])
        if any(p in detail for p in normalized):
            result.append("Slurm job " + job["id"])
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.stat().st_uid != os.getuid() or int(proc.name) == os.getpid():
                continue
            # Match privately; process arguments and environment never leave this helper.
            data = (proc / "cmdline").read_bytes().decode(errors="replace")
            data += (proc / "environ").read_bytes().decode(errors="replace")
            data += os.readlink(proc / "cwd")
            for fd in (proc / "fd").iterdir():
                try:
                    data += os.readlink(fd)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
            if any(p in normalize(data) for p in normalized):
                result.append("process " + proc.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(set(result))


def disposable(path, root):
    return path.name in DISPOSABLE or (
        path.parent == root and (path.name == "deps" or path.name.startswith("mpl-cache-"))
    )


def scan(paths):
    entries, excluded = [], []

    def visit(path, root):
        s = path.lstat()
        row = [str(path), s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        entries.append(row)
        if path != root and disposable(path, root):
            excluded.append(str(path))
            return
        if stat.S_ISDIR(s.st_mode):
            if path != root and os.path.ismount(path):
                raise ValueError(f"nested mount rejected: {path}")
            for child in sorted(path.iterdir()):
                visit(child, root)
        elif not (stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode)):
            raise ValueError(f"special file rejected: {path}")

    for root in paths:
        visit(root, root)
    digest = hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()
    return {"entries": entries, "excluded_runtime_paths": excluded, "tree_sha256": digest}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = json.load(sys.stdin)
    operation = args["operation"]
    if operation == "inventory":
        candidates = []
        for base in sorted(bases()):
            for p in sorted(base.iterdir()):
                if allowed_name(p.name) and not p.name.startswith(".mecar-transfer-"):
                    try:
                        validate(p)
                        candidates.append(str(p))
                    except (ValueError, FileNotFoundError):
                        continue
        print(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "uid": os.getuid(),
                    "candidates": candidates,
                    "jobs": jobs(),
                }
            )
        )
        return
    if operation == "status":
        print(json.dumps({"jobs": jobs()}))
        return
    if operation == "check":
        paths = [validate(p) for p in args["paths"]]
        print(json.dumps({"blockers": blockers(paths)}))
        return
    if operation == "cancel_failed_dependency":
        wanted = set(args["jobs"])
        cancelled = []
        for job in jobs():
            if (
                job["id"] in wanted
                and job["state"] == "PENDING"
                and job["reason"] == "DependencyNeverSatisfied"
            ):
                subprocess.run(["scancel", job["id"]], check=True)
                cancelled.append(job["id"])
        print(json.dumps({"cancelled": cancelled}))
        return
    paths = [validate(p) for p in args["paths"]]
    if not paths:
        raise ValueError("empty target list")
    if len(set(paths)) != len(paths) or any(
        a != b and a in b.parents for a in paths for b in paths
    ):
        raise ValueError("duplicate or overlapping targets")
    busy = blockers(paths)
    if busy:
        raise ValueError("active references: " + ", ".join(busy))
    if operation == "archive":
        before = scan(paths)
        home = next(
            p for p in (Path("/data1/dsl01"), Path("/mnt/data1/dsl01"), Path.home()) if p.exists()
        )
        archive = home / (".mecar-transfer-" + uuid.uuid4().hex + ".tar.gz")
        manifest = {"host": socket.gethostname(), "paths": list(map(str, paths)), **before}
        excluded = set(before["excluded_runtime_paths"])
        with tarfile.open(archive, "x:gz", compresslevel=1, dereference=False) as tar:
            for row in before["entries"]:
                p = Path(row[0])
                if str(p) not in excluded:
                    tar.add(p, arcname="files/" + str(p).lstrip("/"), recursive=False)
            data = json.dumps(manifest, indent=2).encode()
            info = tarfile.TarInfo("transfer-manifest.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if scan(paths)["tree_sha256"] != before["tree_sha256"]:
            archive.unlink()
            raise ValueError("source changed during archiving")
        receipt = {
            "archive": str(archive),
            "archive_sha256": file_hash(archive),
            "archive_bytes": archive.stat().st_size,
            "paths": list(map(str, paths)),
            "tree_sha256": before["tree_sha256"],
            "excluded_runtime_paths": before["excluded_runtime_paths"],
            "archived_entries": len(before["entries"]) - len(excluded),
        }
        archive.with_suffix(".receipt.json").write_text(json.dumps(receipt))
        print(json.dumps(receipt))
    elif operation == "delete_verified":
        archive = Path(args["receipt"]["archive"])
        if (
            not re.fullmatch(r"\.mecar-transfer-[0-9a-f]{32}\.tar\.gz", archive.name)
            or archive.parent.resolve() not in bases()
        ):
            raise ValueError("invalid transfer receipt path")
        receipt_path = archive.with_suffix(".receipt.json")
        receipt = json.loads(receipt_path.read_text())
        if args["receipt"] != receipt or list(map(str, paths)) != receipt["paths"]:
            raise ValueError("receipt mismatch")
        if (
            args["local_verified_sha256"] != receipt["archive_sha256"]
            or file_hash(archive) != receipt["archive_sha256"]
        ):
            raise ValueError("archive digest mismatch")
        if scan(paths)["tree_sha256"] != receipt["tree_sha256"]:
            raise ValueError("source changed since transfer")
        for p in paths:
            validate(p)
            if p.is_dir():
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise RuntimeError("symlink-safe deletion unavailable")
                shutil.rmtree(p, onerror=writable_retry)
            else:
                p.unlink()
        archive.unlink()
        receipt_path.unlink()
        print(
            json.dumps(
                {
                    "deleted": list(map(str, paths)),
                    "verified_absent": all(not p.exists() for p in paths),
                }
            )
        )
    else:
        raise ValueError("unknown operation")


if __name__ == "__main__":
    main()
