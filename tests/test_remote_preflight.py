import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

pytest.importorskip("paramiko")

from scripts.verify_remote_preflight import verify_preflight


def evidence(tmp_path, change=None):
    output, root = tmp_path / "probe", tmp_path / "root"
    collection = output / "collection"
    collection.mkdir(parents=True)
    files = {name: b"verified source" for name in (
        "src/actor.py", "scripts/select_remote_egl.py", "scripts/run_gpu.sh",
        "scripts/gpu_probe.py",
    )}
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {"source_revision": "test-revision", "files_sha256": {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }}
    probe = {"status": "passed", "model_device": "cuda:0", "gl_vendor": "NVIDIA",
             "camera_std": 24, "oracle_valid_sum": 0, "egl_device": "2"}
    egl = {"status": "passed", "cuda_uuid": "allocated", "egl_index": 2,
           "devices": [{"egl_index": 2, "cuda_uuid": "allocated"}]}
    exit_record = {"exit_code": 0, "slurm_job": "123"}
    if change:
        change(probe, egl, exit_record)
    files.update({"stage-manifest.json": json.dumps(manifest).encode(),
                  "campaign.zip": b"sealed-source-archive-fixture",
                  "output/gpu-probe.json": json.dumps(probe).encode(),
                  "output/egl-device.json": json.dumps(egl).encode(),
                  "output/job-exit.json": json.dumps(exit_record).encode()})
    remote = "/data1/dsl01/campaigns/.mecar-flybrain-test"
    archive_path = collection / "batch-00-results.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo("files" + remote + "/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    (collection / "batch-00-verified.json").write_text(json.dumps({
        "paths": [remote], "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }))
    (collection / "lifecycle-status.json").write_text('{"state":"complete"}')
    (output / "submission.json").write_text(json.dumps({
        "mode": "probe", "jobs": ["123"], "remote_run": remote,
        "archive_sha256": hashlib.sha256(files["campaign.zip"]).hexdigest(),
    }))
    return output, root


def test_pending_probe_cannot_authorize_main_submission(tmp_path):
    assert verify_preflight(tmp_path / "probe", tmp_path / "root") is None


def test_collected_source_gpu_rendering_and_sensor_contract_must_all_pass(tmp_path):
    output, root = evidence(tmp_path)
    proof = verify_preflight(output, root)
    assert proof["status"] == "passed" and proof["job"] == "123"
    (root / "src/actor.py").write_text("changed source")
    with pytest.raises(ValueError, match="source changed"):
        verify_preflight(output, root)


@pytest.mark.parametrize("change", [
    lambda p, e, x: p.update(gl_vendor="Mesa"),
    lambda p, e, x: p.update(oracle_valid_sum=1),
    lambda p, e, x: p.update(model_device="cpu"),
    lambda p, e, x: e.update(cuda_uuid="unallocated"),
    lambda p, e, x: x.update(exit_code=1),
])
def test_failed_probe_cannot_pass_gate(tmp_path, change):
    output, root = evidence(tmp_path, change)
    with pytest.raises(RuntimeError):
        verify_preflight(output, root)


def test_modified_collection_archive_cannot_authorize_submission(tmp_path):
    output, root = evidence(tmp_path)
    with (output / "collection/batch-00-results.tar.gz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="receipt mismatch"):
        verify_preflight(output, root)


def test_transient_smb_read_denial_waits_then_rechecks_complete_gpu_proof(tmp_path, monkeypatch):
    output, root = evidence(tmp_path)
    original = Path.read_text
    denied = [True]

    def read_text(path, *args, **kwargs):
        if path.name == "lifecycle-status.json" and denied[0]:
            denied[0] = False
            raise PermissionError("SMB read sharing violation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert verify_preflight(output, root) is None
    assert verify_preflight(output, root)["status"] == "passed"


def test_partial_live_lifecycle_write_does_not_authorize_submission(tmp_path):
    output, root = evidence(tmp_path)
    (output / "collection/lifecycle-status.json").write_text('{"state":')
    assert verify_preflight(output, root) is None
