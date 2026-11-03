"""Isolated Windows host-QPC IPC fixture; never qualifies actual CSP/Lua or game use."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from unittest.mock import patch

from flybrain_drive import session_heartbeat_candidate as candidate
from flybrain_drive import session_heartbeat_provenance as provenance
from flybrain_drive.evidence import write_json

SCHEMA = "flybrain.session_heartbeat.host_ipc_qualification.v1"
SOURCE_FILES = (
    "scripts/qualify_session_heartbeat_ipc.py",
    "src/flybrain_drive/session_heartbeat_candidate.py",
    "src/flybrain_drive/session_heartbeat_provenance.py",
    "src/flybrain_drive/evidence.py",
)
SNAPSHOTS = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_name(owner, index):
    if (
        not isinstance(owner, str)
        or re.fullmatch("[0-9a-f]{32}", owner) is None
        or type(index) is not int
        or index not in (0, 1)
    ):
        raise ValueError("Fresh fixture owner and exactly two fixture slots required")
    return f"Local\\FlyBrainHeartbeatIPCQualification_{owner}_{index}"


def verify_plan(plan):
    if (
        plan.get("schema") != SCHEMA
        or plan.get("scope") != "host_QPC_producer_fixture_only"
        or plan.get("fit_allowed") is not False
        or plan.get("snapshots") != SNAPSHOTS
        or plan.get("game_or_CSP_calls_allowed") is not False
        or plan.get("actual_Lua_qualified") is not False
        or plan.get("legacy_quantization_padding") is not True
    ):
        raise ValueError("Explicit host-only fixture registration required")
    if set(plan.get("input_sha256", {})) != set(SOURCE_FILES):
        raise ValueError("Exact fixture source set required")
    for path, digest in plan["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Fixture source changed: " + path)


def kernel_api():
    from ctypes import wintypes

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    api.OpenFileMappingW.restype = wintypes.HANDLE
    api.MapViewOfFile.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_size_t,
    ]
    api.MapViewOfFile.restype = ctypes.c_void_p
    api.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    api.UnmapViewOfFile.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    return api


def validate_control(control, path):
    owner = control["owner_id"]
    if (
        control.get("schema") != SCHEMA
        or control.get("fixture_only") is not True
        or control.get("names") != [fixture_name(owner, i) for i in (0, 1)]
        or control.get("snapshots") != SNAPSHOTS
        or control.get("generation") != "host-qpc-fixture-" + owner
        or Path(control["output"]).resolve() != Path(path).resolve().parent
        or control.get("program_sha256") != sha(__file__)
        or set(control.get("input_sha256", {})) != set(SOURCE_FILES)
    ):
        raise ValueError("Only this owned non-game fixture control is allowed")
    lease_file = Path(control["lease_file"])
    if (
        lease_file.resolve()
        != Path(control["output"]).resolve() / "fixture-status.json.flybrain-heartbeat-owner"
        or sha(lease_file) != control["lease_sha256"]
    ):
        raise ValueError("Finite producer requires this exact owned fixture lease")
    owner_record = json.loads(lease_file.read_text(encoding="utf-8"))
    if (
        owner_record["owner_id"] != owner
        or owner_record["generation"] != control["generation"]
        or owner_record["host_pid"] != control["parent_pid"]
        or not owner_record["registered_qpc_ns"]
        <= time.perf_counter_ns()
        <= owner_record["deadline_qpc_ns"]
    ):
        raise ValueError("Fixture lease owner/generation/lifetime differs")
    for source, digest in control["input_sha256"].items():
        if sha(source) != digest:
            raise ValueError("Producer fixture source changed")


def producer_fixture(control_file):
    control = json.loads(Path(control_file).read_text(encoding="utf-8"))
    validate_control(control, control_file)
    out = Path(control["output"])
    api = kernel_api()
    handles, views, rows = [], [], []
    result = dict(
        status="failed",
        scope="host_QPC_producer_fixture_only",
        actual_CSP_or_Lua_qualified=False,
        game_calls=0,
        gpu_calls=0,
    )
    try:
        for name in control["names"]:
            handle = api.OpenFileMappingW(2, False, name)  # FILE_MAP_WRITE, own fixture only.
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
            view = api.MapViewOfFile(handle, 2, 0, 0, candidate.SLOT_SIZE)
            if not view:
                raise ctypes.WinError(ctypes.get_last_error())
            views.append(view)
        for sequence in range(1, SNAPSHOTS + 1):
            source = time.perf_counter()
            content = json.dumps(
                dict(
                    schema=candidate.STATUS_SCHEMA,
                    generation=control["generation"],
                    script_clock=source,
                    paused=True,
                    ai_controlled=False,
                    synthetic_host_QPC_fixture=True,
                    fixture_sequence=sequence,
                ),
                separators=(",", ":"),
            ).encode()
            snapshot = candidate.HeartbeatSnapshot(
                control["owner_id"], control["generation"], sequence, source, content
            )
            data = candidate.encode_slot(snapshot)
            view = views[sequence % 2]
            ctypes.c_uint32.from_address(view + 8).value = sequence * 2 - 1
            time.sleep(0.005)  # Deliberate odd-slot visibility, not a source retiming.
            ctypes.memmove(view, data[:8], 8)
            ctypes.memmove(view + 12, data[12:], candidate.SLOT_SIZE - 12)
            ctypes.c_uint32.from_address(view + 8).value = sequence * 2
            rows.append(
                dict(
                    sequence=sequence,
                    source_clock=source,
                    snapshot_json=content.decode(),
                    snapshot_sha256=hashlib.sha256(content).hexdigest(),
                )
            )
            time.sleep(0.03)
        result.update(status="completed_owned_host_QPC_producer_fixture", snapshots=len(rows))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        close_ok = True
        for view in views:
            close_ok = bool(api.UnmapViewOfFile(view)) and close_ok
        for handle in handles:
            close_ok = bool(api.CloseHandle(handle)) and close_ok
        result["all_owned_handles_and_views_released"] = close_ok
        write_json(out / "producer-trace.json", rows)
        write_json(out / "producer-result.json", result)
    return 0 if result["status"].startswith("completed") and close_ok else 1


def self_qpc_calibration(generation):
    exchanges = []
    for _ in range(4):
        sent = time.perf_counter()
        producer = time.perf_counter()  # Explicit same-QPC fixture, no CSP clock ping.
        received = time.perf_counter()
        exchanges.append(dict(sent=sent, received=received, producer_clock=producer))
    low = max(r["sent"] - r["producer_clock"] for r in exchanges)
    high = min(r["received"] - r["producer_clock"] for r in exchanges)
    record = dict(
        generation=generation,
        exchanges=exchanges,
        offset_lower_seconds=low,
        offset_upper_seconds=high,
        fixture_scope="same Windows QPC only; actual CSP clock unqualified",
    )
    return record, candidate.CachedClockLowerBound.from_record(
        record, generation=generation, valid_for_seconds=10.0
    )


def measured_mtime_quantum(path, base_ns):
    """Own fixture file only; preserve requested/observed values without rounding claims."""
    Path(path).write_bytes(b"synthetic timestamp fixture")
    measurements = []
    for offset in (0, 1, 2, 50, 99, 100, 101, 199, 200, 333, 1000):
        requested = base_ns + offset
        os.utime(path, ns=(requested, requested))
        measurements.append(dict(requested_ns=requested, observed_ns=Path(path).stat().st_mtime_ns))
    observed = sorted({row["observed_ns"] for row in measurements})
    differences = [b - a for a, b in zip(observed, observed[1:], strict=False)]
    return dict(
        measurements=measurements,
        observed_increment_gcd_ns=math.gcd(*differences) if differences else None,
    )


def run_fixture(plan, output):
    verify_plan(plan)
    runtime = (Path(os.environ["LOCALAPPDATA"]) / "flybrain" / "runtime-runs").resolve()
    output = Path(output).resolve()
    if output.parent != runtime or not re.fullmatch(r"20261030-t\d+-[A-Za-z0-9-]+", output.name):
        raise ValueError("A new immediate owned runtime-runs fixture directory is required")
    output.mkdir(exist_ok=False)
    (output / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "registration.json", plan)
    owner = uuid.uuid4().hex
    generation = "host-qpc-fixture-" + owner
    names = [fixture_name(owner, i) for i in (0, 1)]
    result = dict(
        schema=SCHEMA,
        status="failed",
        scope="host_QPC_producer_fixture_only",
        actual_CSP_or_Lua_qualified=False,
        production_transport_admitted=False,
        fit_allowed=False,
        optimizer_updates=0,
        game_api_calls=0,
        live_game_mmf_opens=0,
        gpu_calls=0,
        owner_id=owner,
        mapping_names=names,
    )
    child = log = lease = maps = writer = None
    api = None
    seen, readbacks = [], []
    started = time.perf_counter()
    try:
        metadata = provenance.qpc_clock_metadata()
        wall = provenance.PreciseWindowsWallClock()
        brackets = [vars(wall.bracket()) for _ in range(6)]
        legacy_padding_ns = math.ceil(time.get_clock_info("time").resolution * 1e9)
        if not 0 < legacy_padding_ns <= 100_000_000:
            raise ValueError("Explicit bounded legacy reader clock quantum required")
        quantum = measured_mtime_quantum(
            output / "mtime-quantum-fixture.bin", wall.unix_ns() - 100_000_000
        )
        write_json(output / "mtime-quantum.json", quantum)
        record, calibration = self_qpc_calibration(generation)
        write_json(output / "self-qpc-calibration.json", record)
        write_json(
            output / "clock-metadata.json",
            dict(
                qpc=metadata,
                wall_api="GetSystemTimePreciseAsFileTime",
                wall_filetime_tick_ns=100,
                legacy_wall_clock=vars(time.get_clock_info("time")),
                brackets=brackets,
            ),
        )
        now = time.perf_counter_ns()
        target = output / "fixture-status.json"
        lease = provenance.ProvenanceWriterLease(
            target,
            owner_id=owner,
            generation=generation,
            calibration_sha256=calibration.calibration_sha256,
            registered_qpc_ns=now,
            deadline_qpc_ns=now + 10_000_000_000,
            clock_metadata=metadata,
        ).acquire()
        competitor = provenance.ProvenanceWriterLease(
            target,
            owner_id=uuid.uuid4().hex,
            generation=generation,
            calibration_sha256=calibration.calibration_sha256,
            registered_qpc_ns=now,
            deadline_qpc_ns=now + 10_000_000_000,
            clock_metadata=metadata,
        )
        try:
            competitor.acquire()
        except FileExistsError:
            result["second_file_lease_refused"] = True
        else:
            raise AssertionError("Competing lease unexpectedly acquired")
        with patch.object(candidate, "channel_name", side_effect=lambda o, i: fixture_name(o, i)):
            maps = candidate.ExclusiveHeartbeatMaps(lease)
            map_receipt = maps.create()
            duplicate = candidate.ExclusiveHeartbeatMaps(lease)
            try:
                duplicate.create()
            except FileExistsError:
                result["preexisting_fixture_map_refused"] = True
            else:
                duplicate.close()
                raise AssertionError("Existing fixture map was silently opened")
        assert maps.repeated_copies() == [(bytes(candidate.SLOT_SIZE),) * 2] * 2
        write_json(output / "mapping-creation.json", map_receipt)
        control = dict(
            schema=SCHEMA,
            fixture_only=True,
            owner_id=owner,
            generation=generation,
            snapshots=SNAPSHOTS,
            names=names,
            output=str(output),
            program_sha256=sha(__file__),
            input_sha256=plan["input_sha256"],
            lease_file=str(lease.path),
            lease_sha256=lease.payload_sha256,
            parent_pid=os.getpid(),
        )
        write_json(output / "producer-control.json", control)

        def padded_stamp(path, original_lower_ns):
            stamped = original_lower_ns - legacy_padding_ns
            os.utime(path, ns=(stamped, stamped))

        writer = candidate.OwnedLatestHeartbeatWriter(
            lease,
            calibration,
            bracket_clock=wall.bracket,
            maximum_attempts=SNAPSHOTS,
            stamp_private=padded_stamp,
        )
        log = (output / "producer.log").open("xb")
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.qualify_session_heartbeat_ipc",
                "--fixture-producer",
                str(output / "producer-control.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        write_json(
            output / "owned-child.json",
            dict(
                pid=child.pid,
                role="finite host-QPC fixture producer only",
                restart_allowed=False,
                kill_used=False,
            ),
        )
        deadline = time.perf_counter() + 5.0
        last_sequence, last_clock = 0, None
        odd_samples = 0
        while time.perf_counter() < deadline:
            copies = maps.repeated_copies()
            odd_samples += any(candidate.HEADER.unpack_from(pair[0])[2] % 2 for pair in copies)
            snapshot = candidate.latest_snapshot(
                copies,
                owner_id=owner,
                generation=generation,
                last_sequence=last_sequence,
                last_source_clock=last_clock,
            )
            if snapshot is not None:
                lower = calibration.source_host_lower_ns(
                    snapshot.source_clock, generation, time.perf_counter_ns()
                )
                provenance.write_immutable_proof(lease, snapshot, lower)
                writer.submit(snapshot)
                publication = writer.pump_one()
                content = target.read_bytes()
                assert content == snapshot.snapshot_json
                value, modified, proof = provenance.read_proven_status(target)
                assert value == json.loads(snapshot.snapshot_json)
                assert proof["snapshot_sha256"] == hashlib.sha256(content).hexdigest()
                assert target.stat().st_mtime_ns <= publication["source_unix_lower_ns"]
                seen.append(
                    dict(
                        sequence=snapshot.sequence,
                        source_clock=snapshot.source_clock,
                        snapshot_sha256=hashlib.sha256(content).hexdigest(),
                        publication=publication,
                    )
                )
                unpadded_probe = output / "unpadded-wall-age-fixture.bin"
                unpadded_probe.write_bytes(b"synthetic unpadded timestamp comparison")
                unpadded = publication["source_unix_lower_ns"]
                os.utime(unpadded_probe, ns=(unpadded, unpadded))
                unpadded_age = time.time() - unpadded_probe.stat().st_mtime
                legacy_age = time.time() - modified
                if not 0 <= legacy_age <= 2:
                    raise ValueError(
                        "Committed fixture violates original coarse-wall legacy age gate"
                    )
                readbacks.append(
                    dict(
                        sequence=snapshot.sequence,
                        proof=proof,
                        actual_mtime_ns=target.stat().st_mtime_ns,
                        precise_wall_age_seconds=(wall.unix_ns() - target.stat().st_mtime_ns) / 1e9,
                        legacy_time_time_age_seconds=legacy_age,
                        unpadded_legacy_time_time_age_seconds=unpadded_age,
                        additional_pastward_legacy_quantization_padding_ns=legacy_padding_ns,
                    )
                )
                last_sequence, last_clock = snapshot.sequence, snapshot.source_clock
                write_json(output / "host-readbacks.json", dict(seen=seen, readbacks=readbacks))
                if last_sequence == SNAPSHOTS:
                    break
            time.sleep(0.001)
        if not seen or last_sequence != SNAPSHOTS:
            raise TimeoutError(
                "Original finite fixture producer did not deliver its final snapshot"
            )
        child.wait(timeout=5)
        assert child.returncode == 0
        produced = json.loads((output / "producer-trace.json").read_text(encoding="utf-8"))
        assert len(produced) == SNAPSHOTS
        for row in seen:
            original = produced[row["sequence"] - 1]
            assert row["snapshot_sha256"] == original["snapshot_sha256"]
            assert row["source_clock"] == original["source_clock"]
        captured = provenance.capture_lease(target)
        last = candidate.HeartbeatSnapshot(
            owner, generation, last_sequence, last_clock, target.read_bytes()
        )
        writer.retire()
        try:
            writer.submit(last)
        except RuntimeError:
            result["retired_writer_refuses_publication"] = True
        else:
            raise AssertionError("Retired writer accepted work")
        writer.close()
        lease = None
        # An old opened read retains its QPC guard across lease retirement.
        try:
            provenance.validate_opened_status(
                target,
                last.snapshot_json,
                json.loads(last.snapshot_json),
                captured_lease=captured,
                now_qpc_ns=10_000_000_000 + captured[1]["deadline_qpc_ns"],
                clock_metadata=metadata,
            )
        except provenance.HeartbeatProvenanceError:
            result["old_descriptor_after_retire_cannot_revert_to_wall_only"] = True
        else:
            raise AssertionError("Retired opened descriptor lost its QPC guard")
        maps.close()
        maps = None
        api = kernel_api()
        for name in names:
            handle = api.OpenFileMappingW(4, False, name)
            if handle:
                api.CloseHandle(handle)
                raise AssertionError("Owned mapping still exists after all producer/host closes")
            assert ctypes.get_last_error() == 2
        result.update(
            status="completed_isolated_windows_host_QPC_IPC_qualification",
            fixture_snapshots_produced=SNAPSHOTS,
            snapshots_observed=len(seen),
            observed_sequences=[r["sequence"] for r in seen],
            odd_slot_samples_rejected=odd_samples,
            exact_original_payload_hashes_and_source_clocks=True,
            immutable_proof_before_status=True,
            all_observed_source_QPC_ages_within_two_seconds=True,
            all_mtimes_not_later_than_original_conservative_bound=True,
            all_owned_child_views_handles_terminal_and_released=True,
            measured_os_utime_stat_increment_gcd_ns=quantum["observed_increment_gcd_ns"],
            additional_pastward_legacy_quantization_padding_ns=legacy_padding_ns,
            unpadded_coarse_legacy_negative_age_count=sum(
                r["unpadded_legacy_time_time_age_seconds"] < 0 for r in readbacks
            ),
            coarse_legacy_age_negative_count=sum(
                r["legacy_time_time_age_seconds"] < 0 for r in readbacks
            ),
            coarse_legacy_all_ages_within_original_guard=all(
                0 <= r["legacy_time_time_age_seconds"] <= 2 for r in readbacks
            ),
            actual_CSP_clock_mapping_tested=False,
            actual_Lua_layout_visibility_tested=False,
            production_retire_handoff_tested=False,
            source_wall_clock_steps_actually_injected=False,
        )
    except Exception as error:
        result.update(
            error=type(error).__name__, reason=str(error), traceback=traceback.format_exc()
        )
    finally:
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                result["same_owned_child_still_live_pid"] = child.pid
        cleanup_errors = []
        cleanup = []
        if maps is not None:
            cleanup.append(("owned_maps", maps.close))
        if lease is not None and (child is None or child.poll() is not None):
            cleanup.append(("owned_lease", lease.release))
        if log is not None:
            cleanup.append(("owned_log", log.close))
        for name, action in cleanup:
            try:
                action()
            except Exception as error:
                cleanup_errors.append(
                    dict(resource=name, error=type(error).__name__, reason=str(error))
                )
        if cleanup_errors:
            result.update(status="failed", cleanup_errors=cleanup_errors)
        try:
            verify_plan(plan)
            result["registered_sources_reverified_at_exit"] = True
        except Exception as error:
            result.update(status="failed", source_reverification_error=str(error))
        result["wall_seconds"] = time.perf_counter() - started
        result["program_sha256"] = sha(__file__)
        write_json(output / "result.json", result)
        write_json(
            output / "receipt.json",
            dict(
                result_sha256=sha(output / "result.json"),
                program_sha256=sha(__file__),
                original_source_payloads_copied=0,
                actual_CSP_or_Lua_qualified=False,
                production_transport_admitted=False,
            ),
        )
    print(json.dumps(result), flush=True)
    return 0 if result["status"].startswith("completed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--output")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture-producer", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fixture_producer:
        if args.execute or args.plan or args.output:
            raise ValueError("Internal finite producer mode cannot mix with other modes")
        return producer_fixture(args.fixture_producer)
    if not args.plan:
        parser.error("--plan is required")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    verify_plan(plan)
    if not args.execute:
        print(json.dumps(dict(status="review_only_no_IPC_or_worker_started", scope=__doc__)))
        return 0
    if not args.output:
        parser.error("--output required with --execute")
    return run_fixture(plan, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
