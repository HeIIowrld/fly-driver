"""Independent decoder/clock/timing adversarial tests; no producer or OS imports."""

import json
import struct
import sys
import zlib

import pytest

from scripts import audit_session_heartbeat_evidence as audit

OWNER, GEN, LEASE = "a" * 32, "synthetic-private", "b" * 64


def encoded(value, *, sequence=1, control=False):
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    capacity, magic = (2048, 0x46424843) if control else (8192, 0x46424832)
    return (
        struct.pack("<5I", magic, 1, sequence * 2, len(payload), zlib.crc32(payload))
        + payload
        + bytes(capacity - len(payload))
    )


def packet(*, clock=31.1237799, bodyclock=None, owner=OWNER):
    body = dict(
        schema="flybrain.csp.session.status.v1",
        generation=GEN,
        script_clock=clock if bodyclock is None else bodyclock,
    )
    return dict(
        schema="flybrain.session_heartbeat.packet.v1",
        owner_id=owner,
        generation=GEN,
        sequence=1,
        source_clock=clock,
        snapshot_json=json.dumps(body, indent=3),
    )


def decode(raw, second=None):
    return audit.decode_pair(
        raw, raw if second is None else second, kind="data", owner=OWNER, generation=GEN
    )


def test_exact_original_nested_bytes_and_actual_nonzero_wire_clock():
    value = packet()
    raw = encoded(value)
    state, row = decode(raw)
    assert state == "stable" and row["body"] == value["snapshot_json"].encode()
    assert row["source_clock"] == float("31.123779899999999")
    assert row["source_clock"] - float("31.123779900000002") != 0


@pytest.mark.parametrize("disposition", ["torn", "odd", "zero"])
def test_unavailable_original_pairs_are_classified_without_repair(disposition):
    raw = encoded(packet())
    unavailable = raw[:8] + struct.pack("<I", 1 if disposition == "odd" else 0) + raw[12:]
    first, second = (raw, unavailable) if disposition == "torn" else (unavailable, unavailable)
    assert decode(first, second) == (disposition, None)


def test_stable_corrupt_crc_is_rejected_even_when_blob_hash_would_be_updated():
    raw = bytearray(encoded(packet()))
    raw[24] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        decode(bytes(raw))


@pytest.mark.parametrize("change", ["foreign", "nestedclock", "nonfinite", "header"])
def test_foreign_or_inconsistent_original_data_never_becomes_valid(change):
    value = (
        packet(owner="c" * 32)
        if change == "foreign"
        else packet(bodyclock=31.1237798)
        if change == "nestedclock"
        else packet()
    )
    raw = encoded(value)
    if change == "nonfinite":
        payload = json.dumps(value).replace("31.1237799", "NaN").encode()
        raw = (
            struct.pack("<5I", 0x46424832, 1, 2, len(payload), zlib.crc32(payload))
            + payload
            + bytes(8192 - len(payload))
        )
    elif change == "header":
        raw = struct.pack("<I", 0) + raw[4:]
    with pytest.raises(ValueError):
        decode(raw)


def test_control_buffer_requires_exact_raw_lease_and_wire_clock_preserved():
    value = dict(
        schema="flybrain.session_heartbeat.control.v1",
        owner_id=OWNER,
        generation=GEN,
        lease_sha256=LEASE,
        command="run",
        committed_sequence=1,
        committed_source_clock=31.1237799,
    )
    raw = encoded(value, control=True)
    state, row = audit.decode_pair(
        raw, raw, kind="control", owner=OWNER, generation=GEN, lease_sha=LEASE
    )
    assert state == "stable" and row["value"] == value
    with pytest.raises(ValueError, match="lease"):
        audit.decode_pair(raw, raw, kind="control", owner=OWNER, generation=GEN, lease_sha="d" * 64)


def test_calibration_recomputes_original_intersection_not_a_new_probe():
    record = dict(
        exchanges=[
            dict(sent=100.0, received=100.00001, producer_clock=10.0),
            dict(sent=100.01, received=100.01001, producer_clock=10.01),
        ],
        offset_lower_seconds=90.0,
        offset_upper_seconds=90.00001,
    )
    low, high, quantum, at = audit.calibration_values(record)
    assert low == 90.0 and abs(high - 90.00001) < 1e-10 and quantum == 0 and at == 100010010000
    record["offset_upper_seconds"] = 90.01
    with pytest.raises(ValueError, match="intersection"):
        audit.calibration_values(record)


def ledger(rows):
    return dict(
        schema="flybrain.csp.source_timing.v1",
        clock_domain="csp_os_preciseClock_seconds",
        record_fields=[
            "stage",
            "started_clock",
            "finished_clock",
            "duration_seconds",
            "succeeded",
            "update_index",
            "capture_index",
            "detail",
        ],
        overflow_records=0,
        record_count=len(rows),
        records=rows,
        actor_input=False,
        timing_observation_only=True,
        per_frame_timing_disk_writes=0,
        records_frozen=True,
        flushed_after_capture=True,
        active_capture_closed=True,
        capture_closed_clock=2.0,
        frozen_clock=2.1,
        finished_clock=2.2,
    )


def test_new_mmf_heartbeat_can_have_zero_nested_io_save_without_old_assertion():
    rows = [
        ["session_heartbeat", 1.0, 1.001, 0.001, True, 1, 0, None],
        ["write.io.save", 1.2, 1.203, 0.003, True, 2, 1, "other/status.json"],
    ]
    result = audit.source_timing_summary(ledger(rows))
    assert result["nested_io_save_count_distribution"] == {0: 1}
    assert result["actual_nested_io_saves"] == 0 and result["stages"]["write.io.save"]["count"] == 1
    assert result["one_save_per_heartbeat_assumed"] is False


def test_actual_legacy_nested_io_save_is_reported_if_observed():
    rows = [
        ["session_heartbeat", 1.0, 1.1, 0.1, True, 1, 0, None],
        ["write.io.save", 1.01, 1.08, 0.07, False, 1, 0, "session/status.json"],
    ]
    result = audit.source_timing_summary(ledger(rows))
    assert result["nested_io_save_count_distribution"] == {1: 1}
    assert result["actual_nested_io_saves"] == 1 and result["nested_io_save_maximum_ms"] == 70
    assert result["unequal_run_causal_comparison_claimed"] is False


def test_map_probe_requires_explicit_parent_terminal_flag_before_any_file_or_OS_read(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["audit", "--run", "absent", "--output", "absent-out", "--check-retired-maps"]
    )
    monkeypatch.setattr(audit, "retired_map_absence", lambda *_: pytest.fail("No map call"))
    with pytest.raises(ValueError, match="terminal notice"):
        audit.main()
