"""Explicit bounded32 persistence; original eight-task writer remains unchanged.

The cap covers queued/in-flight immutable files AND stream metadata. The same
single producer checks worst-case room before copy/ACK. No I/O starts on import.
"""

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .immutable_frame_writer import ImmutableFrameWriter

PROFILE_SCHEMA = "flybrain.frame_persistence.bounded_profile.v1"
PROFILE_NAME = "buffered32_32mib_v1"
MIB = 1024**2


@dataclass(frozen=True)
class BufferedFrameProfile:
    maximum_pending: int = 32
    maximum_retained_payload_bytes: int = 32 * MIB
    maximum_group_payload_bytes: int = MIB
    maximum_groups: int = 1200

    def __post_init__(self):
        expected = (32, 32 * MIB, MIB, 1200)
        values = (
            self.maximum_pending,
            self.maximum_retained_payload_bytes,
            self.maximum_group_payload_bytes,
            self.maximum_groups,
        )
        if any(type(v) is not int for v in values) or values != expected:
            raise ValueError("Only explicit fixed32/32MiB/1MiB/1200 profile is reviewed")

    def record(self):
        return dict(
            schema=PROFILE_SCHEMA,
            name=PROFILE_NAME,
            maximum_pending=self.maximum_pending,
            maximum_retained_payload_bytes=self.maximum_retained_payload_bytes,
            maximum_group_payload_bytes=self.maximum_group_payload_bytes,
            maximum_groups=self.maximum_groups,
        )

    @classmethod
    def from_record(cls, record):
        value = cls()
        if record != value.record():
            raise ValueError("Exact registered bounded32 profile required")
        return value

    @property
    def sha256(self):
        raw = json.dumps(self.record(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def load_buffered_frame_profile(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Bounded explicit persistence profile JSON required")
    return BufferedFrameProfile.from_record(json.loads(raw))


class BufferedImmutableFrameWriter(ImmutableFrameWriter):
    """New opt-in version; inherited persistence I/O implementation is unchanged."""

    def __init__(
        self, directory, stream_path, *, profile, clock=time.perf_counter, mmf_metadata=True
    ):
        if not isinstance(profile, BufferedFrameProfile) or mmf_metadata is not True:
            raise ValueError("Explicit registered profile and MMF metadata required")
        super().__init__(directory, stream_path, clock=clock, mmf_metadata=True)
        self.profile = profile
        # Only this NEW class changes admission; original writer still rejects>8.
        self.maximum_pending = profile.maximum_pending
        self.producer_thread = None
        self.payload_by_future = {}
        self.retained_payload_bytes = 0
        self.maximum_retained_payload_bytes_observed = 0
        self.payload_admissions = []

    def check_capacity(self):
        producer = threading.get_ident()
        if self.producer_thread is None:
            self.producer_thread = producer
        elif producer != self.producer_thread:
            raise RuntimeError("One original camera producer owns capacity/copy/ACK/enqueue")
        if self.closed:
            raise ValueError("Frame writer is closed")
        while self.pending and self.pending[0].done():
            future = self.pending.popleft()
            self.retained_payload_bytes -= self.payload_by_future.pop(future)
            future.result()
        if self.error:
            raise RuntimeError("Immutable frame persistence failed") from self.error
        if self.submitted >= self.profile.maximum_groups:
            raise BufferError("Finite registered frame-group budget exhausted")
        if len(self.pending) >= self.maximum_pending:
            raise BufferError("Immutable frame queue full; no frame may be silently dropped")
        # One producer cannot increase retained bytes between pre-copy and submit
        # checks. Reserve worst-case room for its current construction group too.
        if self.retained_payload_bytes + self.profile.maximum_group_payload_bytes > (
            self.profile.maximum_retained_payload_bytes
        ):
            raise BufferError("Immutable payload byte cap; no frame may be silently dropped")

    def submit(self, group_id, files, stream_line, *, metadata_files=()):
        if not isinstance(stream_line, bytes):
            raise ValueError("Original immutable stream bytes required")
        entries = (*files, *metadata_files)
        if any(not isinstance(raw, bytes) for _, raw in entries):
            raise ValueError("Original immutable payload bytes required")
        size = sum(len(raw) for _, raw in entries) + len(stream_line)
        if size > self.profile.maximum_group_payload_bytes:
            raise ValueError("Files plus original stream line exceed1MiB group cap")
        submitted = super().submit(group_id, files, stream_line, metadata_files=metadata_files)
        future = self.pending[-1]
        self.payload_by_future[future] = size
        self.retained_payload_bytes += size
        self.maximum_retained_payload_bytes_observed = max(
            self.maximum_retained_payload_bytes_observed, self.retained_payload_bytes
        )
        self.payload_admissions.append(
            dict(
                group_id=group_id,
                payload_bytes=size,
                pending_count_after_submit=len(self.pending),
                retained_payload_bytes_after_submit=self.retained_payload_bytes,
            )
        )
        return submitted

    def close(self):
        try:
            super().close()
        finally:
            # Frozen close sets closed only after same-worker shutdown returns.
            if self.closed:
                self.payload_by_future.clear()
                self.retained_payload_bytes = 0

    def statistics(self):
        return dict(
            super().statistics(),
            bounded_profile=self.profile.record(),
            bounded_profile_sha256=self.profile.sha256,
            retained_payload_bytes=self.retained_payload_bytes,
            maximum_retained_payload_bytes_observed=self.maximum_retained_payload_bytes_observed,
            payload_admissions=[dict(r) for r in self.payload_admissions],
        )


@dataclass(frozen=True)
class BufferedFrameWriterFactory:
    profile: BufferedFrameProfile

    def __post_init__(self):
        if not isinstance(self.profile, BufferedFrameProfile):
            raise ValueError("Explicit registered profile required")

    def __call__(self, directory, stream_path, *, clock=time.perf_counter, mmf_metadata=False):
        return BufferedImmutableFrameWriter(
            directory, stream_path, profile=self.profile, clock=clock, mmf_metadata=mmf_metadata
        )


def verify_payload_accounting(journal, lines):
    profile = BufferedFrameProfile.from_record(journal.get("bounded_profile"))
    if (
        journal.get("bounded_profile_sha256") != profile.sha256
        or journal.get("retained_payload_bytes") != 0
        or journal.get("maximum_pending") != profile.maximum_pending
    ):
        raise ValueError("Actual bounded profile/terminal byte accounting differs")
    admissions, records = journal.get("payload_admissions"), journal["records"]
    if (
        not isinstance(admissions, list)
        or len(admissions) != len(records)
        or len(records) > profile.maximum_groups
    ):
        raise ValueError("Finite original per-group admission accounting required")
    sizes = [
        sum(f["bytes"] for f in row["files"]) + len(line)
        for row, line in zip(records, lines, strict=True)
    ]
    for index, (admission, row, size) in enumerate(zip(admissions, records, sizes, strict=True)):
        count = admission.get("pending_count_after_submit")
        retained = admission.get("retained_payload_bytes_after_submit")
        if type(count) is not int or not 1 <= count <= min(index + 1, profile.maximum_pending):
            raise ValueError("Original payload/task cap accounting differs")
        # Only a completed FIFO prefix can be drained. Retained work is an exact
        # contiguous suffix, even when completed tasks stay counted until check.
        suffix = range(index + 1 - count, index + 1)
        unfinished = [
            i
            for i, r in enumerate(records[: index + 1])
            if i == index or r["finished_at"] > row["submitted_at"]
        ]
        if (
            admission.get("group_id") != index
            or admission.get("payload_bytes") != size
            or not 0 < size <= profile.maximum_group_payload_bytes
            or not set(unfinished).issubset(suffix)
            or type(retained) is not int
            or retained != sum(sizes[i] for i in suffix)
            or not 0 < retained <= profile.maximum_retained_payload_bytes
        ):
            raise ValueError("Original payload/task cap accounting differs")
    if journal.get("maximum_retained_payload_bytes_observed") != max(
        a["retained_payload_bytes_after_submit"] for a in admissions
    ) or journal["maximum_queued"] != max(a["pending_count_after_submit"] for a in admissions):
        raise ValueError("Original peak task/byte evidence differs")
    return profile


def verify_buffered_persistence_archive(camera, *, expected_mmf=False):
    """Independently verify every deferred group, including frames the actor never selected."""
    journal = camera.json("frame-persistence-journal.json")
    lines = camera.read("stream.jsonl").splitlines(keepends=True)
    records = journal["records"]
    profile = BufferedFrameProfile.from_record(journal.get("bounded_profile"))
    count = journal["submitted"]
    schema = (
        "flybrain.immutable_frame_persistence.v2"
        if expected_mmf
        else ("flybrain.immutable_frame_persistence.v1")
    )
    if (
        journal["schema"] != schema
        or journal["closed"] is not True
        or journal["error"] is not None
        or journal["failed"] != 0
        or journal["completed"] != count
        or len(records) != count
        or len(lines) != count
        or not count
        or not 1
        <= journal["maximum_queued"]
        <= journal["maximum_pending"]
        == profile.maximum_pending
    ):
        raise ValueError("Deferred persistence journal incomplete or failed")
    rows = {}
    for index, (record, line) in enumerate(zip(records, lines, strict=True)):
        row = json.loads(line)
        pipeline = row["shared_pipeline"]
        clocks = [
            pipeline["acknowledged_copy_clock"],
            pipeline["persistence_started_clock"],
            row["received"],
            record["submitted_at"],
            record["started_at"],
            record["finished_at"],
        ]
        if (
            record["group_id"] != index
            or row["capture_id"] != index
            or record["status"] != "persisted"
            or not all(math.isfinite(c) for c in clocks)
            or clocks != sorted(clocks)
            or pipeline["revision"]
            != (
                "copy_all_then_mmf_ack_then_async_persist_v3"
                if expected_mmf
                else "copy_all_then_ack_then_async_persist_v2"
            )
            or pipeline["stream_schema"]
            != (
                "flybrain.csp.immutable_rgb_stream.v3"
                if expected_mmf
                else "flybrain.csp.immutable_rgb_stream.v2"
            )
            or pipeline["immutable_memory_ready_clock"] != row["received"]
            or pipeline["actor_published_only_after_persistence"] is not False
            or pipeline["persistence_required_before_run_admission"] is not True
            or "persistence_finished_clock" in pipeline
            or hashlib.sha256(line).hexdigest() != record["stream_line_sha256"]
        ):
            raise ValueError("Deferred receipt identity, clock or protocol differs")
        files = {f["file"]: f for f in record["files"]}
        view_names = {Path(v["file"]).name for v in row["views"]}
        metadata_names = (
            {f"frame-{index:04d}.json", f"ack-{index:04d}.json", f"mmf-{index:04d}.mmf"}
            if expected_mmf
            else set()
        )
        if (
            len(files) != 4 + len(metadata_names)
            or len(record["files"]) != len(files)
            or len(row["views"]) != 4
            or view_names | metadata_names != set(files)
        ):
            raise ValueError("Four distinct persisted views required")
        for view in row["views"]:
            name = Path(view["file"]).name
            raw = camera.read(name)
            if (
                name not in files
                or len(raw) != files[name]["bytes"]
                or hashlib.sha256(raw).hexdigest() != files[name]["sha256"]
                or files[name]["sha256"] != view["file_sha256"]
            ):
                raise ValueError("Deferred frame bytes differ from immutable receipt")
        if expected_mmf:
            from .ac_metadata_mmf import verify_mmf_archive_row

            verify_mmf_archive_row(camera, row)
            for name in metadata_names:
                raw = camera.read(name)
                if (
                    len(raw) != files[name]["bytes"]
                    or hashlib.sha256(raw).hexdigest() != files[name]["sha256"]
                ):
                    raise ValueError("Deferred original MMF metadata differs from persistence")
        rows[index] = row
    verify_payload_accounting(journal, lines)
    return rows
