"""Independent offline heartbeat evidence replay; map absence is explicit opt-in.

No producer/runtime imports, clock pings, actor, API, mapping creation or model use.
Raw bytes are authoritative; producer-derived indexes are crosschecks only.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import datetime
import hashlib
import json
import math
import re
import struct
import zlib
from pathlib import Path

HEADER = struct.Struct("<5I")
LIMITS = dict(
    reads_per_channel=16384,
    raw_bytes=24 * 1024**2,
    raw_variants=dict(data=1024, lua_control=2048, host_control=512, setup=8, snapshot=256),
    stable_data_sequences=256,
    stable_control_sequences=512,
)
DOMAIN = "host_perf_counter_QueryPerformanceCounter_ns"
SCHEMA = "flybrain.session_heartbeat.raw_evidence.v1"


def sha(content):
    return hashlib.sha256(content).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def object_json(raw):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    value = json.loads(raw, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def decode_pair(first, second, *, kind, owner, generation, lease_sha=None):
    """Independent struct/CRC/identity replay, including original unavailable pairs."""
    capacity = 8192 if kind == "data" else 2048
    magic = 0x46424832 if kind == "data" else 0x46424843
    maximum = 256 if kind == "data" else 512
    if len(first) != HEADER.size + capacity or len(second) != HEADER.size + capacity:
        raise ValueError("Exact original mapping buffer size required")
    if first != second:
        return "torn", None
    got_magic, version, seq2, length, crc = HEADER.unpack_from(first)
    if seq2 == 0:
        return "zero", None
    if seq2 % 2:
        return "odd", None
    if (
        got_magic != magic
        or version != 1
        or not 1 <= seq2 // 2 <= maximum
        or not 0 < length <= capacity
    ):
        raise ValueError("Stable buffer header differs")
    payload = first[HEADER.size : HEADER.size + length]
    if zlib.crc32(payload) != crc:
        raise ValueError("Stable original CRC differs")
    value = object_json(payload)
    if value.get("owner_id") != owner or value.get("generation") != generation:
        raise ValueError("Foreign original owner/generation")
    sequence = seq2 // 2
    result = dict(sequence=sequence, value=value, payload=payload)
    if kind == "data":
        if (
            value.get("schema") != "flybrain.session_heartbeat.packet.v1"
            or value.get("sequence") != sequence
            or not isinstance(value.get("snapshot_json"), str)
            or not finite(value.get("source_clock"))
        ):
            raise ValueError("Original data envelope differs")
        body = value["snapshot_json"].encode("utf8")
        status = object_json(body)
        if (
            not 0 < len(body) <= 6144
            or status.get("schema") != "flybrain.csp.session.status.v1"
            or status.get("generation") != generation
            or status.get("script_clock") != value["source_clock"]
        ):
            raise ValueError("Exact nested wire clock/status differs")
        result.update(body=body, status=status, source_clock=value["source_clock"])
    elif (
        value.get("schema") != "flybrain.session_heartbeat.control.v1"
        or value.get("lease_sha256") != lease_sha
    ):
        raise ValueError("Original control lease/schema differs")
    return "stable", result


def calibration_values(record):
    rows = record.get("exchanges", [])
    quantum = record.get("host_clock_quantization_seconds", 0.0)
    if len(rows) < 2 or not finite(quantum) or not 0 <= quantum <= 0.001:
        raise ValueError("Original bounded calibration exchanges required")
    lows, highs, received = [], [], []
    for row in rows:
        sent, end, source = (row.get(k) for k in ("sent", "received", "producer_clock"))
        if not all(finite(x) for x in (sent, end, source)) or not 0 <= sent <= end or source < 0:
            raise ValueError("Original causal calibration exchange differs")
        lows.append(sent - source - quantum)
        highs.append(end - source + quantum)
        received.append(end)
    low, high = max(lows), min(highs)
    if (
        not low <= high
        or high - low > 0.01
        or abs(low - record["offset_lower_seconds"]) > 1e-10
        or abs(high - record["offset_upper_seconds"]) > 1e-10
    ):
        raise ValueError("Independent original calibration intersection differs")
    return low, high, quantum, math.floor(max(received) * 1e9)


def retired_map_absence(owner):
    """Only after a human/parent terminal notice; exact retired owner names only."""
    if not re.fullmatch("[0-9a-f]{32}", owner):
        raise ValueError("Exact reviewed retired owner required")
    from ctypes import wintypes

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    api.OpenFileMappingW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    names = [f"Local\\FlyBrainSessionHeartbeat_{owner}_{i}" for i in (0, 1)]
    names += [f"Local\\FlyBrainSessionHeartbeatControl_{owner}_{d}" for d in ("host", "lua")]
    observations = []
    for name in names:
        ctypes.set_last_error(0)
        handle = api.OpenFileMappingW(4, False, name)
        error = ctypes.get_last_error()
        if handle:
            api.CloseHandle(handle)
        observations.append(
            dict(
                name=name,
                access="FILE_MAP_READ",
                opened=bool(handle),
                winerror=error,
                view_created=False,
            )
        )
    return observations


def audit_raw_evidence(path, parent):
    """Return independent checks/summary; never write to source evidence."""
    path = Path(path).resolve()
    index = object_json((path / "index.json").read_bytes())
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise ValueError(name)

    check(
        "complete bounded original raw evidence",
        index["schema"] == SCHEMA
        and index["status"] == "complete"
        and index["flags"] == {}
        and index["limits"] == LIMITS,
    )
    owner, generation = index["owner_id"], index["generation"]
    check(
        "exact nonempty original identity",
        re.fullmatch("[0-9a-f]{32}", owner) and isinstance(generation, str) and bool(generation),
    )
    blobs, groups = {}, collections.Counter()
    for digest, item in index["blobs"].items():
        check(
            "hash reference syntax " + digest,
            re.fullmatch("[0-9a-f]{64}", digest) and item["file"] == "blobs/" + digest + ".bin",
        )
        target = (path / item["file"]).resolve()
        check("blob path containment " + digest, target.parent == path / "blobs")
        raw = target.read_bytes()
        check("original blob SHA/size " + digest, sha(raw) == digest and len(raw) == item["size"])
        check("known bounded blob group " + digest, item["group"] in LIMITS["raw_variants"])
        blobs[digest] = raw
        groups[item["group"]] += 1
    check("total original raw byte cap", sum(map(len, blobs.values())) <= LIMITS["raw_bytes"])
    check(
        "all original variant caps", all(groups[k] <= v for k, v in LIMITS["raw_variants"].items())
    )
    setup = index["setup_sha256"]
    check(
        "all original static bytes retained",
        set(setup) == {"calibration", "lease", "arm", "sync_status"}
        and all(v in blobs and len(blobs[v]) <= 65536 for v in setup.values()),
    )
    lease, arm, cal = (object_json(blobs[setup[k]]) for k in ("lease", "arm", "calibration"))
    lease_sha = setup["lease"]
    check(
        "original lease identity/domain/fingerprint",
        lease["schema"] == "flybrain.session_heartbeat.file_lease.v1"
        and lease["owner_id"] == owner
        and lease["generation"] == generation
        and lease["clock_domain"] == DOMAIN
        and lease["calibration_sha256"] == setup["calibration"],
    )
    registered, deadline = lease["registered_qpc_ns"], lease["deadline_qpc_ns"]
    check(
        "original finite unchanged lease lifetime",
        type(registered) is int
        and type(deadline) is int
        and 0 <= registered < deadline <= registered + 60_000_000_000,
    )
    low, high, quantum, calibrated = calibration_values(cal)
    check(
        "original calibration constructor fingerprint exact",
        sha(json.dumps(cal, sort_keys=True, allow_nan=False).encode()) == setup["calibration"],
    )
    check(
        "calibration QPC metadata exact and same generation",
        cal["generation"] == generation
        and cal["host_clock_domain"] == DOMAIN
        and cal["host_clock_callable"] == "time.perf_counter"
        and cal["host_clock_metadata"] == lease["clock_metadata"],
    )
    check(
        "calibration remains valid for original lease",
        calibrated <= registered < deadline <= calibrated + 60_000_000_000,
    )
    margin = arm["clock_rounding_margin_ns"]
    check(
        "existing clock rounding margin retained", type(margin) is int and 1 <= margin <= 1_000_000
    )
    check(
        "exact original arm lease CRC/SHA/identity",
        arm["schema"] == "flybrain.session_heartbeat.arm.v1"
        and arm["owner_id"] == owner
        and arm["generation"] == generation
        and arm["lease_sha256"] == lease_sha
        and arm["lease_crc32"] == zlib.crc32(blobs[lease_sha])
        and arm["registered_qpc_ns"] == registered
        and arm["deadline_qpc_ns"] == deadline
        and arm["clock_domain"] == DOMAIN
        and arm["host_ready"] is True
        and arm["host_created_slots"] is True
        and arm["max_snapshots"] == 256,
    )
    check(
        "original conservative source deadline unchanged",
        arm["deadline_source_clock"] == deadline / 1e9 - high,
    )

    def lower(source):
        return math.floor((source + low - quantum) * 1e9) - margin

    check(
        "original read column identities",
        index["read_columns"] == ["observed_after_copy_qpc_ns", "first_sha256", "second_sha256"],
    )
    check("exact original read channels", set(index["reads"]) == {"data0", "data1", "lua_control"})
    stable_data, stable_control, bodies = {}, {}, {}
    dispositions = {}
    first_data_read = {}
    for channel, records in index["reads"].items():
        check("original read cap " + channel, len(records) <= LIMITS["reads_per_channel"])
        counter, previous = collections.Counter(), registered
        for n, row in enumerate(records):
            check(
                f"original read reference {channel}/{n}",
                len(row) == 3
                and type(row[0]) is int
                and previous <= row[0] <= deadline
                and row[1] in blobs
                and row[2] in blobs,
            )
            observed, one, two = row
            previous = observed
            kind = "data" if channel.startswith("data") else "control"
            disposition, decoded = decode_pair(
                blobs[one],
                blobs[two],
                kind=kind,
                owner=owner,
                generation=generation,
                lease_sha=lease_sha,
            )
            counter[disposition] += 1
            if decoded is None:
                continue
            seq = str(decoded["sequence"])
            if kind == "data":
                check(
                    f"original data slot parity {channel}/{n}",
                    decoded["sequence"] % 2 == int(channel[-1]),
                )
                check(
                    f"original data epoch/causal clock {channel}/{n}",
                    decoded["status"]["epoch"] == arm["epoch"]
                    and lower(decoded["source_clock"]) <= observed,
                )
                body_sha = sha(decoded["body"])
                check(
                    f"exact nested snapshot bytes retained {channel}/{n}",
                    body_sha in blobs and blobs[body_sha] == decoded["body"],
                )
                value = dict(
                    channel=channel,
                    raw_sha256=one,
                    snapshot_json_sha256=body_sha,
                    source_clock=decoded["source_clock"],
                )
                check(
                    f"original data sequence immutable {channel}/{n}",
                    seq not in stable_data or stable_data[seq] == value,
                )
                stable_data[seq], bodies[seq] = value, decoded["status"]
                first_data_read.setdefault(seq, observed)
            else:
                value = dict(raw_sha256=one, value=decoded["value"])
                check(
                    f"original Lua sequence immutable {channel}/{n}",
                    seq not in stable_control or stable_control[seq] == value,
                )
                stable_control[seq] = value
        dispositions[channel] = dict(counter)
    check("independent stable data index exact", stable_data == index["stable_data"])
    check(
        "independent stable Lua control index exact", stable_control == index["stable_lua_control"]
    )
    check(
        "independent exact snapshot index",
        {k: v["snapshot_json_sha256"] for k, v in stable_data.items()}
        == index["snapshot_json_sha256"],
    )
    check("no original observed sequence gaps", index["observed_data_sequence_gaps"] == [])
    seqs = sorted(map(int, stable_data))
    check("all emitted original source sequences retained", seqs == list(range(1, max(seqs) + 1)))
    sources = [stable_data[str(k)]["source_clock"] for k in seqs]
    check(
        "original wire clocks strictly increase",
        all(a < b for a, b in zip(sources, sources[1:], strict=False)),
    )
    final = max(seqs)
    commands, committed, previous_qpc = [], 0, registered
    sends = index["original_host_sends"]
    check("original host send cap", 1 <= len(sends) <= 512)
    for ordinal, item in enumerate(sends, 1):
        check(
            "original host send completed sequence " + str(ordinal),
            item["sequence"] == ordinal
            and item["completed"] is True
            and item["error"] is None
            and item["sha256"] in blobs
            and previous_qpc <= item["observed_qpc_ns"] <= deadline,
        )
        previous_qpc = item["observed_qpc_ns"]
        _, decoded = decode_pair(
            blobs[item["sha256"]],
            blobs[item["sha256"]],
            kind="control",
            owner=owner,
            generation=generation,
            lease_sha=lease_sha,
        )
        check("original host header sequence " + str(ordinal), decoded["sequence"] == ordinal)
        value = decoded["value"]
        count = value["committed_sequence"]
        check(
            "original host committed source cannot regress " + str(ordinal),
            type(count) is int and committed <= count <= final,
        )
        if count:
            check(
                "strict original wire source echo " + str(ordinal),
                value["committed_source_clock"] == stable_data[str(count)]["source_clock"],
            )
        else:
            check(
                "zero commit never carries synthetic source clock",
                "committed_source_clock" not in value,
            )
        committed = count
        commands.append(value["command"])
    ranks = {"run": 0, "retire": 1, "writer_fenced": 2, "release_confirmed": 3}
    check(
        "original host lifecycle order",
        commands[0] == "run"
        and commands[-1] == "release_confirmed"
        and commands.count("writer_fenced") == commands.count("release_confirmed") == 1
        and "retire" in commands
        and all(c in ranks for c in commands)
        and [ranks[c] for c in commands] == sorted(ranks[c] for c in commands),
    )
    witness, handoff = index["sync_witness"], index["actual_handoff"]
    check(
        "original SYNC exact retained bytes",
        witness["status_sha256"] == setup["sync_status"]
        and handoff["status_sha256"] == setup["sync_status"],
    )
    one, two = witness["ack_first_sha256"], witness["ack_second_sha256"]
    disposition, ack = decode_pair(
        blobs[one],
        blobs[two],
        kind="control",
        owner=owner,
        generation=generation,
        lease_sha=lease_sha,
    )
    check(
        "original raw stable SYNCACK exact decoded witness",
        disposition == "stable"
        and ack["sequence"] == witness["ack_sequence"]
        and ack["value"] == witness["ack_value"],
    )
    sync_bytes = blobs[setup["sync_status"]]
    sync = object_json(sync_bytes)
    ack = ack["value"]
    check(
        "independent original SYNC CRC/length/source/hash",
        ack["phase"] == "sync_return"
        and ack["sync_crc32"] == zlib.crc32(sync_bytes)
        and ack["sync_length"] == len(sync_bytes)
        and ack["sync_source_clock"] == sync["script_clock"] == handoff["original_source_clock"]
        and ack["final_sequence"] == final
        and ack["committed_sequence"] == final
        and ack["committed_source_clock"] == sources[-1],
    )
    check(
        "original final SYNC state quiescent",
        sync["schema"] == "flybrain.csp.session.status.v1"
        and sync["generation"] == generation
        and sync["epoch"] == arm["epoch"]
        and sync["paused"] is True
        and sync["ai_controlled"] is False
        and all(finite(sync[k]) for k in ("speed_kmh", "gas", "brake"))
        and abs(sync["speed_kmh"]) < 0.1
        and sync["gas"] < 0.02
        and sync["brake"] > 0.98
        and not any(
            sync.get(k)
            for k in (
                "camera_active",
                "expert_probe_active",
                "brake_settle_active",
                "pending_receipt_id",
                "error",
            )
        ),
    )
    release = handoff["owned_release"]
    check(
        "actual same writer fenced before original owned release",
        handoff["same_writer_joined"] is True
        and handoff["lease_removed"] is True
        and release["owner_id"] == owner
        and release["generation"] == generation
        and release["expected_sha256"] == lease_sha
        and release["descriptor_closed"] is True
        and release["unlinked"] is True,
    )
    check(
        "actual owned release original deadline/freshSYNC",
        registered
        <= release["started_qpc_ns"]
        <= release["finished_qpc_ns"]
        <= release["deadline_qpc_ns"]
        <= deadline
        and release["deadline_qpc_ns"] - release["started_qpc_ns"] == 10_000_000
        and sources[-1] < sync["script_clock"]
        and lower(sync["script_clock"]) <= release["started_qpc_ns"]
        and release["finished_qpc_ns"] - lower(sync["script_clock"]) <= 2_000_000_000,
    )
    check(
        "original same-fd SYNC mtime retained",
        type(witness["original_mtime_ns"]) is int and witness["original_mtime_ns"] > 0,
    )
    check(
        "parent actual handoff equals captured witness",
        parent["handoff"] == handoff
        and parent["transport_state"] == "closed"
        and parent["transport_error"] is None
        and parent["owned_threads_still_live"] == []
        and parent["lease_retained"] is False
        and parent["terminal_control_ack"]["sequence"] == witness["ack_sequence"]
        and parent["terminal_control_ack"]["value"] == witness["ack_value"],
    )
    publication_ages = []
    proof_expected = []
    for item in parent["worker_records"]:
        seq = str(item["sequence"])
        bracket = item["wall_qpc_bracket"]
        check(
            "actual worker exact original bytes " + seq,
            item["status"] == "published"
            and item["owner_id"] == owner
            and item["generation"] == generation
            and item["payload_sha256"] == stable_data[seq]["snapshot_json_sha256"]
            and item["source_clock"] == stable_data[seq]["source_clock"],
        )
        check(
            "independent original source lower " + seq,
            item["source_host_lower_ns"] == lower(item["source_clock"]),
        )
        check(
            "independent Unix conservative mtime " + seq,
            item["source_unix_lower_ns"]
            == item["source_host_lower_ns"]
            + bracket["unix_ns"]
            - bracket["qpc_after_ns"]
            - margin
            - 1_000_000
            and item["actual_mtime_ns"] <= item["source_unix_lower_ns"],
        )
        age = item["published_qpc_ns"] - item["source_host_lower_ns"]
        check(
            "actual publication original source age " + seq,
            0 <= age <= 2_000_000_000 and registered <= item["published_qpc_ns"] <= deadline,
        )
        publication_ages.append(age)
        proof_expected.append(
            dict(
                sequence=item["sequence"],
                snapshot_sha256=item["payload_sha256"],
                original_source_clock=item["source_clock"],
                source_host_lower_qpc_ns=item["source_host_lower_ns"],
            )
        )
    raw_delta = float(ack.get("last_wire_delta_text", "0"))
    check(
        "actual raw-wire clock difference within existing margin",
        finite(raw_delta)
        and abs(raw_delta) * 1e9 <= margin
        and float(ack["last_wire_source_clock_text"]) == sources[-1]
        and float(ack["last_wire_source_clock_text"]) - float(ack["last_raw_source_clock_text"])
        == raw_delta,
    )
    return dict(
        schema="flybrain.session_heartbeat.independent_raw_evidence_audit.v1",
        status="passed_original_raw_evidence",
        checks=len(checks),
        check_details=checks,
        owner_id=owner,
        generation=generation,
        index_sha256=sha((path / "index.json").read_bytes()),
        raw_blob_count=len(blobs),
        raw_bytes=sum(map(len, blobs.values())),
        raw_group_counts=dict(groups),
        original_read_counts={k: len(v) for k, v in index["reads"].items()},
        read_dispositions=dispositions,
        source_sequences=seqs,
        actual_publications=len(parent["worker_records"]),
        first_data_read_qpc_ns=first_data_read,
        original_host_command_counts=dict(collections.Counter(commands)),
        original_calibration_intersection=[low, high],
        original_host_quantization_seconds=quantum,
        lease_sha256=lease_sha,
        calibration_sha256=setup["calibration"],
        arm_sha256=setup["arm"],
        original_status_target=lease["target"],
        original_lease_registered_qpc_ns=registered,
        original_lease_deadline_qpc_ns=deadline,
        original_sync_sha256=setup["sync_status"],
        original_sync_crc32=zlib.crc32(sync_bytes),
        original_sync_length=len(sync_bytes),
        original_sync_same_fd_mtime_ns=witness["original_mtime_ns"],
        actual_raw_wire_delta_seconds=raw_delta,
        maximum_publication_source_age_upper_ns=max(publication_ages),
        proof_expected=proof_expected,
        original_snapshots_and_clocks_changed=False,
        original_pristine_zero_checks_captured=False,
        core_pristine_checks_source_backed=True,
        exact_physical_sample_clock_claimed=False,
        fit_allowed=False,
        student_qualified=False,
        model_updates=0,
        controls_sent=0,
        live_game_mapping_accesses=0,
    )


def source_timing_summary(ledger):
    """Summarize actual stages; MMF heartbeat intentionally need not nest io.save."""
    if (
        ledger["schema"] != "flybrain.csp.source_timing.v1"
        or ledger["clock_domain"] != "csp_os_preciseClock_seconds"
        or ledger["record_fields"]
        != [
            "stage",
            "started_clock",
            "finished_clock",
            "duration_seconds",
            "succeeded",
            "update_index",
            "capture_index",
            "detail",
        ]
        or ledger["overflow_records"] != 0
        or len(ledger["records"]) != ledger["record_count"]
    ):
        raise ValueError("Original bounded source timing schema/counts differ")
    if (
        ledger.get("actor_input") is not False
        or ledger.get("timing_observation_only") is not True
        or ledger.get("per_frame_timing_disk_writes") != 0
        or not ledger.get("records_frozen")
        or not ledger.get("flushed_after_capture")
        or not ledger.get("active_capture_closed")
        or not ledger["capture_closed_clock"] <= ledger["frozen_clock"] <= ledger["finished_clock"]
    ):
        raise ValueError("Original timing ledger closure/observation scope differs")
    by = collections.defaultdict(list)
    for row in ledger["records"]:
        if (
            len(row) != 8
            or not isinstance(row[0], str)
            or not all(finite(x) for x in row[1:4])
            or row[2] < row[1]
            or abs(row[3] - (row[2] - row[1])) > 1e-9
            or type(row[4]) is not bool
        ):
            raise ValueError("Original source timing row differs")
        by[row[0]].append(row)
    heartbeats = by.get("session_heartbeat", [])
    nested_save_counts = []
    nested_saves = []
    for hb in heartbeats:
        nested = [
            r
            for r in by.get("write.io.save", [])
            if r[5] == hb[5] and hb[1] - 1e-9 <= r[1] and r[2] <= hb[2] + 1e-9
        ]
        nested_save_counts.append(len(nested))
        nested_saves.extend(nested)
    return dict(
        original_clock_domain=ledger["clock_domain"],
        records=ledger["record_count"],
        overflow=0,
        stages={
            k: dict(
                count=len(v),
                maximum_ms=1000 * max(r[3] for r in v),
                false_returns=sum(r[4] is False for r in v),
            )
            for k, v in by.items()
        },
        session_heartbeat_count=len(heartbeats),
        nested_io_save_count_distribution=dict(collections.Counter(nested_save_counts)),
        actual_nested_io_saves=len(nested_saves),
        nested_io_save_maximum_ms=1000 * max((r[3] for r in nested_saves), default=0),
        source_clock_retimed=False,
        one_save_per_heartbeat_assumed=False,
        unequal_run_causal_comparison_claimed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registration-plan")
    parser.add_argument("--terminal-confirmed", action="store_true")
    parser.add_argument("--check-retired-maps", action="store_true")
    args = parser.parse_args()
    if args.check_retired_maps and not args.terminal_confirmed:
        raise ValueError("Parent/human terminal notice required before exact retired map check")
    run, out = Path(args.run), Path(args.output)
    if out.exists():
        raise FileExistsError("Never overwrite an independent audit")
    parent = object_json((run / "result.json").read_bytes())
    if not parent["status"].startswith("completed"):
        raise ValueError(
            "Completed original parent result required; preserve failed runs separately"
        )
    ref = parent["heartbeat_evidence"]
    index_path = Path(ref["index"])
    if sha(index_path.read_bytes()) != ref["index_sha256"]:
        raise ValueError("Original evidence index digest differs")
    result = audit_raw_evidence(index_path.parent, parent)
    result["source_run_result_sha256"] = sha((run / "result.json").read_bytes())
    result["source_program_sha256"] = sha(Path(__file__).read_bytes())
    result["original_proof_bindings"] = []
    for expected in result["proof_expected"]:
        proof_path = (
            Path(result["original_status_target"]).parent
            / ".flybrain-heartbeat-provenance"
            / result["owner_id"]
            / (expected["snapshot_sha256"] + ".json")
        )
        raw_proof = proof_path.read_bytes()
        proof = object_json(raw_proof)
        if (
            any(proof.get(k) != v for k, v in expected.items())
            or proof.get("schema") != "flybrain.session_heartbeat.source_provenance.v1"
            or proof.get("owner_id") != result["owner_id"]
            or proof.get("generation") != result["generation"]
            or proof.get("lease_sha256") != result["lease_sha256"]
            or proof.get("calibration_sha256") != result["calibration_sha256"]
            or proof.get("expires_qpc_ns") != result["original_lease_deadline_qpc_ns"]
            or proof.get("clock_domain") != DOMAIN
        ):
            raise ValueError("Original immutable source proof differs")
        result["original_proof_bindings"].append(dict(path=str(proof_path), sha256=sha(raw_proof)))
    recording = object_json((run / "recording/result.json").read_bytes())
    budget = recording["heartbeat_activation_budget"]
    checked = budget["checked_qpc_ns"]
    remaining = (result["original_lease_deadline_qpc_ns"] - checked) / 1e9
    if (
        recording["status"] != "completed"
        or budget["owner_id"] != result["owner_id"]
        or budget["generation"] != result["generation"]
        or budget["deadline_qpc_ns"] != result["original_lease_deadline_qpc_ns"]
        or not result["original_lease_registered_qpc_ns"]
        <= checked
        <= result["original_lease_deadline_qpc_ns"]
        or budget["collection_seconds"] not in (3, 30)
        or budget["cleanup_seconds"] != 13
        or budget["handoff_seconds"] != 2
        or budget["remaining_seconds"] != remaining
        or remaining < budget["collection_seconds"] + 15
    ):
        raise ValueError("Original actual pre-Popen activation budget differs")
    result["original_pre_Popen_activation_budget"] = budget
    result["recording_result_sha256"] = sha((run / "recording/result.json").read_bytes())
    timing_ref = recording["vision"]["source_timing"]
    timing_raw = Path(timing_ref["producer_file"]).read_bytes()
    if sha(timing_raw) != timing_ref["sha256"]:
        raise ValueError("Original source timing ledger SHA differs")
    result["source_timing"] = source_timing_summary(object_json(timing_raw))
    result["source_timing"].update(ledger=str(timing_ref["producer_file"]), sha256=sha(timing_raw))
    result["registration_bindings"] = []
    if args.registration_plan:
        plan = object_json(Path(args.registration_plan).read_bytes())
        for name, digest in plan["input_sha256"].items():
            matches = sha(Path(name).read_bytes()) == digest
            result["registration_bindings"].append(dict(path=name, sha256=digest, matches=matches))
            if not matches:
                raise ValueError("Registered current/snapshot source differs: " + name)
        result["registration_plan_sha256"] = sha(Path(args.registration_plan).read_bytes())
    result["retired_map_observations"] = []
    if args.check_retired_maps:
        observations = retired_map_absence(result["owner_id"])
        result["retired_map_observations"] = observations
        if any(r["opened"] or r["winerror"] != 2 for r in observations):
            raise ValueError("Exact retired owned map still exists or absence unproven")
    result["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out.mkdir(parents=True)
    raw = (json.dumps(result, indent=2) + "\n").encode()
    (out / "result.json").write_bytes(raw)
    receipt = dict(
        result_sha256=sha(raw),
        executed_source=str(Path(__file__).resolve()),
        executed_source_sha256=sha(Path(__file__).read_bytes()),
        source_run=str(run),
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print(
        json.dumps(
            dict(
                status=result["status"],
                checks=result["checks"],
                result_sha256=receipt["result_sha256"],
            )
        )
    )


if __name__ == "__main__":
    main()
