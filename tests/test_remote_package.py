import hashlib
import json
import zipfile

import pytest

pytest.importorskip("paramiko")

from scripts.remote_gpu import remote_path, seal


def test_sealed_files_have_matching_hashes_and_no_runs_or_credentials(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    with zipfile.ZipFile(wheels / "synthetic-test-wheel.whl", "w") as wheel:
        wheel.writestr(
            "synthetic_test-1.0.dist-info/METADATA", "Name: synthetic-test\nVersion: 1.0\n"
        )
    manifest = seal(tmp_path / "stage", {"device": "cuda:0"}, wheels)
    with zipfile.ZipFile(tmp_path / "stage/campaign.zip") as archive:
        embedded = json.loads(archive.read("stage-manifest.json"))
        assert embedded["files_sha256"] == manifest["files_sha256"]
        for name, digest in embedded["files_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert not name.startswith(("runs/", ".git/", "assets/external/"))
        assert "MECAR_SSH_PASSWORD" not in embedded["config"]
    with pytest.raises(FileExistsError):
        seal(tmp_path / "stage", {}, wheels)


def test_remote_cleanup_path_is_exact_owned_tree():
    assert remote_path("/data1/dsl01/campaigns/.mecar-flybrain-test")
    for path in (
        "/data1/dsl01",
        "/tmp/.mecar-flybrain-test",
        "../.mecar-flybrain-test",
        "/data1/dsl01/campaigns/.mecar-flybrain-test/../another",
    ):
        with pytest.raises(ValueError):
            remote_path(path)
