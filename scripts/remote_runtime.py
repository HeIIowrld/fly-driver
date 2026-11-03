"""Prepare a new immutable dependency overlay; existing shared caches are read-only."""

import hashlib
import json
import os
import ssl
import subprocess
import sys
import urllib.request
import uuid
import venv
from pathlib import Path


def acquire_wheels(root, cache):
    """Download exact official files into an owned package cache and verify every hash."""
    manifest = json.loads((root / "input/wheel-manifest.json").read_text())
    result = []
    packages = cache / "packages/flybrain-wheels"
    packages.mkdir(parents=True, exist_ok=True)
    certificates = next(
        (
            str(path)
            for path in (
                Path("/etc/ssl/certs/ca-certificates.crt"),
                Path("/etc/pki/tls/certs/ca-bundle.crt"),
            )
            if path.exists()
        ),
        None,
    )
    context = ssl.create_default_context(cafile=certificates)
    for entry in manifest:
        filename = entry["filename"]
        if Path(filename).name != filename or not filename.endswith(".whl"):
            raise ValueError("Invalid wheel filename")
        expected = entry["sha256"]
        if len(expected) != 64:
            raise ValueError("Invalid wheel SHA-256")
        directory = packages / expected
        directory.mkdir(exist_ok=True)
        target = directory / filename
        if target.is_symlink():
            raise ValueError("Package cache symlink rejected")
        if not target.exists():
            uploaded = root / "input/wheels" / filename
            partial = directory / (filename + "." + uuid.uuid4().hex + ".partial")
            if uploaded.exists():
                partial.write_bytes(uploaded.read_bytes())
            else:
                url = f"https://pypi.org/pypi/{entry['distribution']}/{entry['version']}/json"
                with urllib.request.urlopen(url, timeout=60, context=context) as response:
                    project = json.load(response)
                official = next(item for item in project["urls"] if item["filename"] == filename)
                if official["digests"]["sha256"] != expected:
                    raise ValueError("Official package hash differs from sealed local manifest")
                if not official["url"].startswith("https://files.pythonhosted.org/"):
                    raise ValueError("Unexpected package download origin")
                with (
                    urllib.request.urlopen(
                        official["url"], timeout=60, context=context
                    ) as response,
                    partial.open("xb") as out,
                ):
                    while block := response.read(1024 * 1024):
                        out.write(block)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
                raise ValueError("Downloaded/uploaded package SHA-256 mismatch")
            partial.replace(target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("Package cache SHA-256 mismatch")
        result.append(target)
    return result


def main():
    root = Path(sys.argv[1]).resolve()
    base = Path(sys.argv[2]).resolve()
    cache = Path(sys.argv[3])
    if (
        cache.name != "mecar-runtime-cache"
        or cache.parent.name != "opt"
        or cache.is_symlink()
        or cache.resolve() != cache
    ):
        raise ValueError("Expected canonical opt/mecar-runtime-cache")
    if cache.stat().st_uid != os.getuid():
        raise ValueError("Runtime cache must be owned by the current user")
    wheel_paths = acquire_wheels(root, cache)
    wheels = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheel_paths}
    base_manifest = json.loads((base / "runtime.json").read_text())
    specification = {
        "base_environment": base.name,
        "base_versions": base_manifest["versions"],
        "python": sys.version,
        "wheels": wheels,
        "schema": "flybrain-overlay-v2",
    }
    identity = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    target = cache / "flybrain-environments" / identity
    target.parent.mkdir(exist_ok=True)
    import fcntl

    with (cache / ".flybrain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (target / "runtime.json").exists():
            if target.exists():
                raise ValueError("Incomplete overlay exists; inspect before retrying")
            stage = target.parent / (".building-" + uuid.uuid4().hex)
            venv.EnvBuilder(with_pip=True, symlinks=True).create(stage)
            version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            site = stage / "lib" / version / "site-packages"
            (site / "flybrain-existing-runtime.pth").write_text(
                str(base / "lib" / version / "site-packages")
                + "\n"
                + "/mnt/data1/dsl01/site-packages\n"
            )
            # Install into the new overlay only; no network nor shared environment mutation.
            command = [
                str(stage / "bin/python"),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                *map(str, wheel_paths),
            ]
            subprocess.run(command, check=True)
            (stage / "runtime.json").write_text(
                json.dumps({"identity": identity, "specification": specification})
            )
            stage.rename(target)
    output = root / "output"
    output.mkdir(exist_ok=True)
    (output / "runtime.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "base_environment": base.name,
                "runtime_path": str(target),
                "specification": specification,
            },
            indent=2,
        )
    )
    print(str(target / "bin/python"))


if __name__ == "__main__":
    main()
