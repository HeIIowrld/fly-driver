"""Run evidence, no overwritten runs and no secrets captured."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .graph import sha256


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    # Windows/SMB readers may briefly deny delete-sharing during atomic replacement.
    # Retry only the rename, for a bounded interval; never alter permissions or drop data.
    for attempt in range(7):
        try:
            temporary.replace(path)
            break
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 6:
                raise
            time.sleep(min(.01 * 2**attempt, .1))


def source_hashes(root):
    """Keep the exact source set and sorted order while bounding parallel NAS reads."""
    root = Path(root)
    files = [p
        for directory in ("src", "configs", "scripts", "integrations/ac", "integrations/ac_csp")
        for p in sorted((root / directory).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
        and p.suffix in (".py", ".json", ".lua", ".ini")]
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='run-provenance') as pool:
        hashes = list(pool.map(sha256, files))
    return {str(p.relative_to(root)).replace('\\', '/'): value
            for p, value in zip(files, hashes, strict=True)}


def begin_run(path, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = source_hashes(root)
    versions = {}
    for name in ("numpy", "gymnasium", "mujoco", "torch", "mecar-flybrain", "mss",
                 "windows-capture", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    try:
        revision = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "-C",
                str(root),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root.as_posix()}",
                "-C",
                str(root),
                "status",
                "--porcelain",
            ],
            capture_output=True,
            text=True,
        )
        revision_value = revision.stdout.strip() if revision.returncode == 0 else "uncommitted"
        dirty_value = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except FileNotFoundError:
        revision_value, dirty_value = "git_unavailable", None
    write_json(
        path / "manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "source_sha256": sources,
            "source_root": str(root),
            "working_directory": str(Path.cwd()),
            "versions": versions,
            "git_revision": revision_value,
            "git_dirty": dirty_value,
            "runtime_environment": {
                key: os.environ.get(key) for key in (
                    "MUJOCO_GL", "GALLIUM_DRIVER", "MESA_D3D12_DEFAULT_ADAPTER_NAME",
                    "MIOPEN_FIND_MODE", "OMP_NUM_THREADS", "LP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
            "staged_source_provenance": (
                json.loads((root / "source-provenance.json").read_text(encoding='utf8'))
                if (root / "source-provenance.json").is_file() else None
            ),
            "evidence_scope": "prototype smoke / diagnostic, not a scientific benchmark",
        },
    )
    return path
