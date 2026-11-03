"""Validate collected GPU preflight evidence before authorizing main submission."""

import hashlib
import json
import tarfile
from pathlib import Path

from flybrain_drive.graph import sha256
from scripts.remote_gpu import remote_path


def verify_preflight(output, root):
    try:
        return _verify_preflight(output, root)
    except (FileNotFoundError, PermissionError):
        # SMB can temporarily deny reads while the remote collector updates a file.
        # Pending evidence never authorizes submission; the caller enforces its deadline.
        return None


def _verify_preflight(output, root):
    output, root = Path(output), Path(root)
    lifecycle = output / "collection/lifecycle-status.json"
    if not lifecycle.exists():
        return None
    try:
        state = json.loads(lifecycle.read_text(encoding="utf-8"))["state"]
    except json.JSONDecodeError:
        # Only the live lifecycle file may be observed during an incomplete write.
        return None
    if state == "failed":
        raise RuntimeError("Remote preflight collection failed")
    if state != "complete":
        return None
    submission = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    if submission["mode"] != "probe" or len(submission["jobs"]) != 1:
        raise ValueError("Expected a single owned GPU probe submission")
    remote = remote_path(submission["remote_run"])
    verified = json.loads(
        (output / "collection/batch-00-verified.json").read_text(encoding="utf-8")
    )
    archive_path = output / "collection/batch-00-results.tar.gz"
    if verified["paths"] != [remote] or sha256(archive_path) != verified["sha256"]:
        raise ValueError("Remote preflight archive receipt mismatch")
    with tarfile.open(archive_path) as archive:
        prefix = "files" + remote + "/"

        def read(name):
            members = [m for m in archive.getmembers() if m.name == prefix + name]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("Missing or ambiguous preflight evidence: " + name)
            return archive.extractfile(members[0]).read()

        manifest = json.loads(read("stage-manifest.json"))
        if hashlib.sha256(read("campaign.zip")).hexdigest() != submission["archive_sha256"]:
            raise ValueError("Sealed preflight source archive mismatch")
        for name, digest in manifest["files_sha256"].items():
            if hashlib.sha256(read(name)).hexdigest() != digest:
                raise ValueError("Preflight staged file hash mismatch: " + name)
            if name.startswith("src/") or name in (
                "scripts/run_gpu.sh", "scripts/select_remote_egl.py", "scripts/gpu_probe.py",
            ):
                if sha256(root / name) != digest:
                    raise ValueError("Runtime source changed after GPU preflight: " + name)
        exit_record = json.loads(read("output/job-exit.json"))
        if exit_record != {"exit_code": 0, "slurm_job": submission["jobs"][0]}:
            raise RuntimeError("GPU preflight job did not exit successfully")
        probe = json.loads(read("output/gpu-probe.json"))
        egl = json.loads(read("output/egl-device.json"))
    if (probe.get("status") != "passed" or probe.get("model_device") != "cuda:0"
            or "nvidia" not in probe.get("gl_vendor", "").lower()
            or probe.get("camera_std", 0) <= 1 or probe.get("oracle_valid_sum") != 0):
        raise RuntimeError("GPU camera / learning / sensor-only preflight failed")
    matches = [d["egl_index"] for d in egl.get("devices", [])
               if d.get("cuda_uuid") == egl.get("cuda_uuid") and d.get("cuda_uuid")]
    if (egl.get("status") != "passed" or matches != [egl.get("egl_index")]
            or probe.get("egl_device") != str(egl.get("egl_index"))):
        raise RuntimeError("CUDA and EGL allocation identity mismatch")
    return {"status": "passed", "job": submission["jobs"][0],
            "archive_sha256": verified["sha256"], "source_revision": manifest["source_revision"],
            "cuda_uuid": egl["cuda_uuid"], "probe": probe}
