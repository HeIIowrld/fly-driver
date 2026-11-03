"""Private files/synthetic futures only; no game, GPU, mappings or control."""

import hashlib
import json
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest

from flybrain_drive import immutable_frame_writer as legacy
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_metadata_mmf import encode_packet
from flybrain_drive.frame_persistence_profile_v3 import (
    BufferedFrameProfile,
    BufferedFrameWriterFactory,
    BufferedImmutableFrameWriter,
    load_buffered_frame_profile,
    verify_buffered_persistence_archive,
    verify_payload_accounting,
)


class ManualExecutor:
    def __init__(self, **kwargs):
        assert kwargs["max_workers"] == 1
        self.tasks = []

    def submit(self, fn, *args):
        future = Future()
        self.tasks.append((future, fn, args))
        return future

    def shutdown(self, *, wait, cancel_futures):
        assert wait is True and cancel_futures is False
        for future, fn, args in self.tasks:
            try:
                future.set_result(fn(*args))
            except BaseException as error:
                future.set_exception(error)


def group(index):
    files = [
        (f"{index:04d}-{name}.bmp", name.encode()) for name in ("front", "left", "right", "rear")
    ]
    metadata = [
        (f"frame-{index:04d}.json", b"{}"),
        (f"ack-{index:04d}.json", b"{}"),
        (f"mmf-{index:04d}.mmf", b"raw"),
    ]
    return files, metadata


def writer(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, "ThreadPoolExecutor", ManualExecutor)
    return BufferedImmutableFrameWriter(
        tmp_path, tmp_path / "stream.jsonl", profile=BufferedFrameProfile(), clock=lambda: 4.0
    )


def test_default_eight_is_unchanged_and_new_profile_holds32_then_fails_before33(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(legacy, "ThreadPoolExecutor", ManualExecutor)
    old = legacy.ImmutableFrameWriter(tmp_path, tmp_path / "old.jsonl")
    assert old.maximum_pending == 8
    old.close()
    with pytest.raises(ValueError, match="1..8"):
        legacy.ImmutableFrameWriter(tmp_path, tmp_path / "bad.jsonl", maximum_pending=32)
    current = writer(tmp_path, monkeypatch)
    try:
        expected = 0
        for i in range(32):
            files, metadata = group(i)
            line = (str(i) + "\n").encode()
            current.check_capacity()  # exact pre-copy/pre-ACK surface
            current.submit(i, files, line, metadata_files=metadata)
            expected += sum(len(b) for _, b in (*files, *metadata)) + len(line)
        assert current.retained_payload_bytes == expected <= 32 * 1024**2
        assert current.maximum_queued == len(current.pool.tasks) == 32
        with pytest.raises(BufferError, match="queue full"):
            current.check_capacity()
        assert current.submitted == 32
    finally:
        current.close()
    assert current.statistics()["completed"] == 32
    assert current.statistics()["retained_payload_bytes"] == 0
    assert (tmp_path / "stream.jsonl").read_bytes() == b"".join(
        (str(i) + "\n").encode() for i in range(32)
    )
    verify_payload_accounting(
        current.statistics(), (tmp_path / "stream.jsonl").read_bytes().splitlines(keepends=True)
    )


def test_stream_metadata_counts_toward_group_byte_cap_before_enqueue(tmp_path, monkeypatch):
    current = writer(tmp_path, monkeypatch)
    files, metadata = group(0)
    file_bytes = sum(len(b) for _, b in (*files, *metadata))
    try:
        with pytest.raises(ValueError, match="stream line"):
            current.submit(0, files, b"x" * (1024**2 - file_bytes + 1), metadata_files=metadata)
        assert current.submitted == 0 and not current.pool.tasks
    finally:
        current.close()


def test_second_producer_cannot_invalidate_pre_copy_byte_reservation(tmp_path, monkeypatch):
    current = writer(tmp_path, monkeypatch)
    current.check_capacity()
    current.producer_thread += 1
    try:
        with pytest.raises(RuntimeError, match="One original camera producer"):
            current.check_capacity()
    finally:
        current.close()


def test_profile_is_fixed_and_config_does_not_silently_expand_any_budget(tmp_path):
    profile = load_buffered_frame_profile("configs/ac-frame-persistence-buffered32-20261030.json")
    assert profile == BufferedFrameProfile()
    record = profile.record()
    record["maximum_retained_payload_bytes"] += 1
    target = tmp_path / "invalid.json"
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Exact registered"):
        load_buffered_frame_profile(target)
    with pytest.raises(ValueError, match="fixed32"):
        BufferedFrameProfile(maximum_pending=33)


def test_factory_requires_explicit_mmf_profile_before_any_worker(tmp_path):
    with pytest.raises(ValueError, match="MMF metadata"):
        BufferedFrameWriterFactory(BufferedFrameProfile())(
            tmp_path, tmp_path / "stream.jsonl", mmf_metadata=False
        )


def test_none_factory_uses_original_constructor_and_optin_is_per_instance(tmp_path, monkeypatch):
    from flybrain_drive import ac_csp_vision as vision

    calls = []
    marker = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return marker

    monkeypatch.setattr(vision, "ImmutableFrameWriter", original)
    recorder = object.__new__(CSPVisionRecorder)
    recorder.directory, recorder.local = tmp_path, tmp_path / "local"
    recorder.host_clock, recorder.mmf_metadata = lambda: 1.0, True
    recorder.frame_writer_factory = None
    assert recorder._make_frame_writer() is marker
    assert calls == [
        (
            (tmp_path, tmp_path / "local/stream.jsonl"),
            dict(clock=recorder.host_clock, mmf_metadata=True),
        )
    ]
    alternate = []
    recorder.frame_writer_factory = lambda *a, **k: alternate.append((a, k))
    recorder._make_frame_writer()
    assert len(calls) == len(alternate) == 1


def test_bad_factory_rejected_before_accessing_game():
    with pytest.raises(ValueError, match="Explicit callable"):
        CSPVisionRecorder(0, {}, frame_writer_factory=object())


class Archive:
    def __init__(self, root):
        self.root = root

    def read(self, name):
        return (self.root / name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))


def valid_archive(tmp_path, monkeypatch):
    current = writer(tmp_path, monkeypatch)
    files, _ = group(0)
    frame = dict(
        index=0,
        request_id="synthetic",
        generation="g",
        transport="shared_immutable_v1",
        metadata_transport="mmf_slot_sha256_v1",
        views=[
            dict(view_id=name, image_writer="host_lossless_bmp", shared_texture_handle=str(i + 10))
            for i, name in enumerate(("front", "left", "right", "rear"))
        ],
    )
    ack = dict(request_id="synthetic", generation="g", capture_id=0, copied_before_ack=1.4)
    raw_frame, raw_ack = json.dumps(frame).encode(), json.dumps(ack).encode()
    slots = encode_packet(0, raw_frame) + encode_packet(0, raw_ack)
    row = dict(
        capture_id=0,
        generation="g",
        received=3.0,
        file="vision/frame-0000.json",
        file_sha256=hashlib.sha256(raw_frame).hexdigest(),
        views=[
            dict(
                view_id=name,
                file="vision/" + file,
                file_sha256=hashlib.sha256(raw).hexdigest(),
                render_started_lower=0.5,
            )
            for name, (file, raw) in zip(("front", "left", "right", "rear"), files, strict=True)
        ],
        shared_copy_receipts=[
            dict(
                view_id=name,
                original_handle=i + 10,
                source_texture_modified=False,
                started_at=1.2,
                finished_at=1.3,
                description=dict(
                    Width=320, Height=180, MipLevels=1, ArraySize=1, SampleCount=1, Format=28
                ),
            )
            for i, name in enumerate(("front", "left", "right", "rear"))
        ],
        metadata_transport="mmf_slot_sha256_v1",
        metadata_mmf_receipt=dict(
            sequence=2,
            started_at=0.9,
            finished_at=1.0,
            raw_sha256=hashlib.sha256(raw_frame).hexdigest(),
            slots_sha256=hashlib.sha256(slots).hexdigest(),
            producer_slot_modified=False,
            repeated_slot_copies_equal=True,
        ),
        shared_pipeline=dict(
            revision="copy_all_then_mmf_ack_then_async_persist_v3",
            stream_schema="flybrain.csp.immutable_rgb_stream.v3",
            acknowledged_copy_clock=1.4,
            persistence_started_clock=2.0,
            immutable_memory_ready_clock=3.0,
            actor_published_only_after_persistence=False,
            persistence_required_before_run_admission=True,
        ),
    )
    metadata = [("frame-0000.json", raw_frame), ("ack-0000.json", raw_ack), ("mmf-0000.mmf", slots)]
    current.submit(0, files, (json.dumps(row) + "\n").encode(), metadata_files=metadata)
    current.close()
    journal = current.statistics()
    (tmp_path / "frame-persistence-journal.json").write_text(json.dumps(journal))
    return journal, row


def test_separate_verifier_checks_real32_profile_and_old_verifier_still_rejects(
    tmp_path, monkeypatch
):
    journal, row = valid_archive(tmp_path, monkeypatch)
    assert verify_buffered_persistence_archive(Archive(tmp_path), expected_mmf=True)[0] == row
    assert journal["maximum_pending"] == 32
    with pytest.raises(ValueError, match="incomplete"):
        legacy.verify_persistence_archive(Archive(tmp_path), expected_mmf=True)
    journal["maximum_pending"] = 8  # No fake old-journal projection allowed.
    (tmp_path / "frame-persistence-journal.json").write_text(json.dumps(journal))
    with pytest.raises(ValueError):
        verify_buffered_persistence_archive(Archive(tmp_path), expected_mmf=True)


def test_tampered_byte_accounting_is_rejected(tmp_path, monkeypatch):
    journal, _ = valid_archive(tmp_path, monkeypatch)
    journal["payload_admissions"][0]["payload_bytes"] -= 1
    (tmp_path / "frame-persistence-journal.json").write_text(json.dumps(journal))
    with pytest.raises(ValueError, match="cap accounting"):
        verify_buffered_persistence_archive(Archive(tmp_path), expected_mmf=True)


@pytest.mark.parametrize("count,byte_factor", [(32, None), (1, None), (2, 2)])
def test_impossible_first_fifo_suffix_cannot_be_forged_under_the_caps(
    tmp_path, monkeypatch, count, byte_factor
):
    journal, _ = valid_archive(tmp_path, monkeypatch)
    admission = journal["payload_admissions"][0]
    retained = 32 * 1024**2 if byte_factor is None else admission["payload_bytes"] * byte_factor
    admission["pending_count_after_submit"] = count
    admission["retained_payload_bytes_after_submit"] = retained
    journal["maximum_queued"] = count
    journal["maximum_retained_payload_bytes_observed"] = retained
    (tmp_path / "frame-persistence-journal.json").write_text(json.dumps(journal))
    with pytest.raises(ValueError, match="cap accounting"):
        verify_buffered_persistence_archive(Archive(tmp_path), expected_mmf=True)


def test_new_derivatives_leave_original_actor_replay_loop_and_old_files_unchanged():
    from scripts.prepare_buffered_frame_profile import BASES, build

    outputs = build()
    assert all(p.read_bytes() == raw for p, raw in outputs.items())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in BASES.items())
    original = Path("scripts/audit_ac_engine_ai_probe.py").read_text(encoding="utf8")
    current = outputs[Path("scripts/audit_ac_engine_ai_probe_buffered.py")].decode()
    start, end = "        for index, event in enumerate(events):", "    intervals = np.diff("
    assert (
        original[original.index(start) : original.index(end)]
        == (current[current.index(start) : current.index(end)])
    )


def test_new_runner_defaults_to_inventory_and_passes_explicit_profile(monkeypatch, capsys):
    from scripts import qualify_session_heartbeat_evidence_buffered as runner

    profile = "configs/ac-frame-persistence-buffered32-20261030.json"
    arguments = runner.collection_arguments("pace30", "output", profile)
    assert arguments[arguments.index("--frame-persistence-profile") + 1] == profile
    assert "--pace-task" in arguments
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--mode",
            "camera3",
            "--expected-engine-pid",
            "1",
            "--expected-generation",
            "synthetic",
            "--expected-epoch",
            "0",
            "--expected-installed-sha256",
            "0" * 64,
            "--frame-persistence-profile",
            profile,
            "--output",
            "not-created",
        ],
    )
    monkeypatch.setattr(runner, "run", lambda args: pytest.fail("No default live work"))
    runner.main()
    assert json.loads(capsys.readouterr().out)["live_actions"] == 0
