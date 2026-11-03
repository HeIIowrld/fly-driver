"""No actual mapping/process creation; private fixture files only."""

import json

import pytest

from scripts import qualify_session_heartbeat_ipc as qualifier


def plan():
    return dict(
        schema=qualifier.SCHEMA,
        scope="host_QPC_producer_fixture_only",
        fit_allowed=False,
        snapshots=qualifier.SNAPSHOTS,
        game_or_CSP_calls_allowed=False,
        actual_Lua_qualified=False,
        legacy_quantization_padding=True,
        input_sha256={p: qualifier.sha(p) for p in qualifier.SOURCE_FILES},
    )


def test_fresh_fixture_namespace_cannot_address_game_maps():
    assert (
        qualifier.fixture_name("a" * 32, 0)
        == "Local\\FlyBrainHeartbeatIPCQualification_" + "a" * 32 + "_0"
    )
    assert "acpmf" not in qualifier.fixture_name("b" * 32, 1)


@pytest.mark.parametrize(
    "owner,index", [("acpmf_physics", 0), ("a" * 32, 2), ("A" * 32, 0), ("a" * 32, False)]
)
def test_malformed_or_extra_fixture_slot_is_rejected(owner, index):
    with pytest.raises(ValueError):
        qualifier.fixture_name(owner, index)


@pytest.mark.parametrize(
    "mutation",
    [
        {"scope": "CSP_game"},
        {"game_or_CSP_calls_allowed": True},
        {"actual_Lua_qualified": True},
        {"fit_allowed": True},
        {"snapshots": 100},
        {"legacy_quantization_padding": False},
        {"input_sha256": {}},
    ],
)
def test_plan_refuses_expanded_scope_before_any_os_resource(mutation):
    p = plan()
    p.update(mutation)
    with pytest.raises(ValueError):
        qualifier.verify_plan(p)


def test_self_calibration_explicitly_has_no_CSP_clock_exchange():
    record, bound = qualifier.self_qpc_calibration("fixture-generation")
    assert len(record["exchanges"]) == 4
    assert "actual CSP clock unqualified" in record["fixture_scope"]
    assert bound.generation == "fixture-generation" and bound.valid_for_ns == 10_000_000_000


def test_real_private_file_timestamp_measurement_records_requested_and_observed(tmp_path):
    target = tmp_path / "synthetic.bin"
    result = qualifier.measured_mtime_quantum(target, 1_700_000_000_000_000_000)
    assert len(result["measurements"]) == 11
    assert result["observed_increment_gcd_ns"] is None or result["observed_increment_gcd_ns"] > 0
    assert all(row["observed_ns"] <= row["requested_ns"] for row in result["measurements"])


def test_review_mode_does_not_start_ipc_or_process(tmp_path, monkeypatch):
    registration = tmp_path / "plan.json"
    registration.write_text(json.dumps(plan()), encoding="utf-8")
    monkeypatch.setattr(qualifier.sys, "argv", ["qualifier", "--plan", str(registration)])
    monkeypatch.setattr(
        qualifier, "kernel_api", lambda: pytest.fail("No Windows IPC in review mode")
    )
    monkeypatch.setattr(
        qualifier.subprocess, "Popen", lambda *a, **k: pytest.fail("No process in review mode")
    )
    assert qualifier.main() == 0


def test_game_mapping_name_is_rejected_in_internal_producer_control(tmp_path):
    source = tmp_path / "producer-control.json"
    c = dict(
        schema=qualifier.SCHEMA,
        fixture_only=True,
        owner_id="a" * 32,
        generation="host-qpc-fixture-" + "a" * 32,
        snapshots=qualifier.SNAPSHOTS,
        names=["Local\\acpmf_physics", "Local\\acpmf_graphics"],
        output=str(tmp_path),
        program_sha256=qualifier.sha(qualifier.__file__),
        input_sha256=plan()["input_sha256"],
    )
    with pytest.raises(ValueError, match="owned non-game fixture"):
        qualifier.validate_control(c, source)
