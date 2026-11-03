"""One source/schema compatibility check; no real capture/model/device operations."""

import ast
import json
from pathlib import Path

import pytest


def test_lifecycle_b2_is_prospective_exact_parent_and_full_history_compatible(tmp_path):
    from flybrain_drive.camera_status_policy_v1 import from_record
    from flybrain_drive.pedal_load_b2 import NATIVE_SOURCE_FIELDS, load_candidate
    from flybrain_drive.pedal_train_b2 import (
        LIFECYCLE_INPUTS,
        PROTECTED,
        SOURCE_FILES,
        sha_bytes,
        validate_declaration,
    )
    from scripts.build_pedal_b2 import BASE, TARGET, build, edits, inverse

    assert len(edits()) == 6 and build(BASE.read_bytes()) == TARGET.read_bytes()
    assert inverse(TARGET.read_bytes()) == BASE.read_bytes()
    recorder = ast.parse(
        Path("scripts/record_ac_engine_ai_probe_lifecycle_status_v1.py").read_bytes()
    )
    actual = next(
        ast.literal_eval(n.value)
        for n in recorder.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SAME_COPY_SOURCE_FILES" for t in n.targets)
    )
    assert len(actual) == 13 and NATIVE_SOURCE_FIELDS == frozenset(actual)
    assert NATIVE_SOURCE_FIELDS <= set(SOURCE_FILES)
    closure = json.loads(
        Path(
            "C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1895-mmf-camera-lifecycle-status-preparation/closure.json"
        ).read_bytes()
    )
    assert len(closure["source_sha256"]) == 91 and set(closure["source_sha256"]) <= set(
        SOURCE_FILES
    )
    assert set(LIFECYCLE_INPUTS) <= set(SOURCE_FILES)
    draft = json.loads(Path("configs/pedal-b2-20261030.json").read_bytes())
    validate_declaration(draft, sealed=False)
    from_record(json.loads(Path(draft["camera_status_profile"]).read_bytes()))
    assert (
        draft["data_or_fit_permission"] is False
        and draft["actual_pipeline_qualification_completed"] is False
    )
    assert draft["nonce"] is None and all(v is None for v in draft["bindings"].values())
    assert draft["source_sha256"] == {} and draft["protected_regions"] == [
        list(r) for r in PROTECTED
    ]
    with pytest.raises(ValueError):
        validate_declaration(draft)
    # Only a manifest exists: old roles/schema must fail before payload access.
    old = json.loads(Path("configs/pedal-b1-20261030.json").read_bytes())
    for i, config in enumerate(
        (
            dict(role="diagnostic", original_train_declaration=draft),
            dict(role="train", original_train_declaration=old),
        )
    ):
        root = tmp_path / str(i)
        root.mkdir()
        raw = json.dumps(dict(config=config)).encode()
        (root / "manifest.json").write_bytes(raw)
        with pytest.raises(ValueError):
            load_candidate(root, expected_capture_manifest_sha256=sha_bytes(raw))
    # Body/clock/held-input/loss and geographic observer functions keep the B1 implementation.
    for stem, names in (
        ("pedal_load", ("derive_history", "api_windows", "temporal_reasons", "camera_intervals")),
        ("pedal_train", ("route_admission", "pedal_gate")),
    ):
        old_text = Path(f"src/flybrain_drive/{stem}_b1.py").read_text()
        new_text = Path(f"src/flybrain_drive/{stem}_b2.py").read_text()
        for name in names:
            before = next(n for n in ast.parse(old_text).body if getattr(n, "name", None) == name)
            after = next(n for n in ast.parse(new_text).body if getattr(n, "name", None) == name)
            assert ast.dump(before) == ast.dump(after), name
