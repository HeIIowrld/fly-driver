"""Verify every staged source/config/wheel before use."""

import hashlib
import json
from pathlib import Path, PurePosixPath

root = Path.cwd()
manifest = json.loads((root / "stage-manifest.json").read_text())
for name, expected in manifest["files_sha256"].items():
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe stage member")
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Stage member SHA-256 mismatch: {name}")
print("Verified", len(manifest["files_sha256"]), "stage members", flush=True)
