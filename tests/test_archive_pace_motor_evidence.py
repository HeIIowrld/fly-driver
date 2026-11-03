from pathlib import Path

import pytest

from scripts import archive_pace_motor_evidence as a


def put(path, content="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def fixture_tree(tmp_path):
    repo, runtime = tmp_path / "repo", tmp_path / "runtime"
    repo.mkdir()
    runtime.mkdir()
    put(repo / "source.py", "answer = 42\n")
    put(runtime / "20261030-t1704-a/result.json", '{"status":"completed", "fit_allowed":false}')
    return repo, runtime


def test_exact_date_immediate_child_and_stage_file_policy(tmp_path):
    repo, runtime = fixture_tree(tmp_path)
    run = runtime / "20261030-t1704-a"
    put(runtime / "20261029-t1705-other/result.json")
    put(runtime / "20261030-t1732-other/result.json")
    put(runtime / "unrelated/20261030-t1706-nested/result.json")
    for name in ["events.jsonl", "native.jsonl", "observations.npz", "weights.pt", "checks.json"]:
        put(run / name, "payload must not be selected")
    put(run / "camera/manifest.json", "camera must not be selected")
    put(run / "recording/events.jsonl", "must not be selected")
    put(run / "recording/control/api-receipt.json")
    put(run / "attempt-00/result.json")
    put(run / "attempt-00/not-approved.json")
    put(run / "attempt-00/executed-program.py")
    put(run / "attempt-00/executed-inputs/src/model.py", "old = 1")
    put(run / "attempt-00/executed-inputs/payload.npz", "never")
    put(run / "executed-program.py", "old = 2")
    put(run / "audit-v2.py", "historical_second_auditor = 1")
    put(run / "combine.py", "historical_combiner = 1")
    put(run / "unapproved-auditor.py", "must not be selected")
    entries, names = a.collect_entries(repo, runtime, ("source.py",))
    selected = {
        e.source.relative_to(runtime).as_posix()
        for e in entries
        if e.kind != "current_source_at_archive_time"
    }
    assert names == ["20261030-t1704-a"]
    assert selected == {
        "20261030-t1704-a/result.json",
        "20261030-t1704-a/executed-program.py",
        "20261030-t1704-a/audit-v2.py",
        "20261030-t1704-a/combine.py",
        "20261030-t1704-a/recording/control/api-receipt.json",
        "20261030-t1704-a/attempt-00/result.json",
        "20261030-t1704-a/attempt-00/executed-program.py",
        "20261030-t1704-a/attempt-00/executed-inputs/src/model.py",
    }


def test_payload_declarations_are_preserved_without_payload_open(tmp_path, monkeypatch):
    repo, runtime = fixture_tree(tmp_path)
    put(
        runtime / "20261030-t1704-a/manifest.json",
        '{"events_sha256":"'
        + "a" * 64
        + '","input_sha256":{"missing.npz":"'
        + "b" * 64
        + '"},"recording":"C:/runtime-runs/run/recording"}',
    )
    entries, _ = a.collect_entries(repo, runtime, ("source.py",))
    original = Path.read_bytes

    def guard(path):
        assert path.suffix not in a.PAYLOAD_SUFFIXES
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guard)
    refs, facts = a.metadata_evidence(entries)
    assert any(r.get("declared_path") == "missing.npz" for r in refs)
    assert all(r["payload_opened"] is False for r in refs)
    assert facts[0]["field"] == "fit_allowed" and facts[0]["value"] is False


def test_archive_separates_current_and_executed_and_validates_every_copy(tmp_path):
    repo, runtime = fixture_tree(tmp_path)
    put(runtime / "20261030-t1704-a/executed-program.py", "historical = 1")
    entries, names = a.collect_entries(repo, runtime, ("source.py",))
    output = tmp_path / "archive"
    result = a.archive(
        entries, names, output, all_owned_jobs_terminal=True, final_docs_reviewed=True
    )
    assert result["copied_files"] == len(entries)
    for entry in entries:
        assert a.digest((output / entry.destination).read_bytes()) == entry.sha256
    assert (output / "current-sources/source.py").exists()
    assert (output / "executed-snapshots/20261030-t1704-a/executed-program.py").exists()
    assert not result["fit_permission_granted"]
    assert not (output / "archive-failure.json").exists()
    with pytest.raises(FileExistsError):
        a.archive(entries, names, output, all_owned_jobs_terminal=True, final_docs_reviewed=True)


def test_both_owner_confirmations_are_required_before_output_creation(tmp_path):
    repo, runtime = fixture_tree(tmp_path)
    entries, names = a.collect_entries(repo, runtime, ("source.py",))
    output = tmp_path / "archive"
    for jobs, docs in [(False, True), (True, False), (False, False)]:
        with pytest.raises(ValueError, match="Owner must confirm"):
            a.archive(
                entries, names, output, all_owned_jobs_terminal=jobs, final_docs_reviewed=docs
            )
        assert not output.exists()


def test_size_limits_reject_without_creating_archive(tmp_path, monkeypatch):
    repo, runtime = fixture_tree(tmp_path)
    monkeypatch.setattr(a, "MAX_FILE_BYTES", 3)
    with pytest.raises(ValueError, match="small regular file"):
        a.collect_entries(repo, runtime, ("source.py",))
    monkeypatch.setattr(a, "MAX_FILE_BYTES", 2_000_000)
    monkeypatch.setattr(a, "MAX_TOTAL_BYTES", 3)
    with pytest.raises(ValueError, match="exceeds"):
        a.collect_entries(repo, runtime, ("source.py",))


def test_source_change_preserves_partial_archive_failure(tmp_path):
    repo, runtime = fixture_tree(tmp_path)
    entries, names = a.collect_entries(repo, runtime, ("source.py",))
    put(repo / "source.py", "changed = 2")
    output = tmp_path / "archive"
    with pytest.raises(ValueError, match="changed before copy"):
        a.archive(entries, names, output, all_owned_jobs_terminal=True, final_docs_reviewed=True)
    assert (output / "archive-failure.json").exists()
    assert (output / "runtime-metadata/20261030-t1704-a/result.json").exists()
    assert not (output / "result.json").exists()


def test_unapproved_current_payload_and_missing_current_source_fail(tmp_path):
    repo, runtime = fixture_tree(tmp_path)
    put(repo / "payload.npz", "no")
    with pytest.raises(ValueError, match="source type"):
        a.collect_entries(repo, runtime, ("payload.npz",))
    with pytest.raises(FileNotFoundError):
        a.collect_entries(repo, runtime, ("missing.py",))


def test_facts_are_never_fabricated_from_unknown_result_status(tmp_path):
    repo, runtime = fixture_tree(tmp_path)
    put(runtime / "20261030-t1704-a/result.json", '{"status":"running","valid_laps":0}')
    entries, _ = a.collect_entries(repo, runtime, ("source.py",))
    _, facts = a.metadata_evidence(entries)
    assert facts == []


def test_only_the_named_derived_figure_is_allowed_as_current_png(tmp_path):
    repo, runtime = fixture_tree(tmp_path)
    put(repo / a.DERIVED_FIGURE, "derived plot, not camera")
    entries, _ = a.collect_entries(repo, runtime, (a.DERIVED_FIGURE,))
    assert entries[-1].kind == "derived_figure_not_camera"
    put(repo / "docs/figures/other.png", "not declared")
    with pytest.raises(ValueError, match="source type"):
        a.collect_entries(repo, runtime, ("docs/figures/other.png",))


def test_generated_metadata_counts_toward_total_limit(tmp_path, monkeypatch):
    repo, runtime = fixture_tree(tmp_path)
    entries, names = a.collect_entries(repo, runtime, ("source.py",))
    monkeypatch.setattr(a, "MAX_TOTAL_BYTES", 1000)
    output = tmp_path / "archive"
    with pytest.raises(ValueError, match="plus generated"):
        a.archive(entries, names, output, all_owned_jobs_terminal=True, final_docs_reviewed=True)
    assert not output.exists()


def test_stage_metadata_and_exact_preactivation_names_preserved(tmp_path):
    repo, runtime = fixture_tree(tmp_path)
    run = runtime / "20261030-t1704-a"
    allowed = [
        "attempt-00-before.json",
        "transit-00-assignment.json",
        "attempt-00/executed-audit.py",
        "attempt-00/REPORT.md",
        "attempt-00/receipt.json",
        "attempt-01/executed-audit.py",
        "attempt-01/REPORT.md",
        "attempt-01/receipt.json",
        "transit-00/transit-01-assignment.json",
        "candidate-00/receipt.json",
    ]
    denied = [
        "attempt-00-after.json",
        "attempt-0-before.json",
        "transit-00-assignment-extra.json",
        "recording/REPORT.md",
        "recording/receipt.json",
        "recording/executed-audit.py",
        "control/REPORT.md",
        "control/receipt.json",
        "control/executed-audit.py",
        "control/attempt-00-before.json",
        "random/receipt.json",
    ]
    for relative in allowed + denied:
        put(run / relative)
    entries, _ = a.collect_entries(repo, runtime, ("source.py",))
    selected = {
        e.source.relative_to(run).as_posix() for e in entries if e.source.is_relative_to(run)
    }
    assert set(allowed) <= selected
    assert not set(denied) & selected


def test_current_whitelist_contains_machine_mechanism_plan():
    assert "configs/fly-flight-mechanism-plan-20261029.json" in a.CURRENT_SOURCE_WHITELIST
