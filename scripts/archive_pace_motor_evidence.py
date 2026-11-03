"""Create a small, source-only archive after the owner confirms terminal jobs.

Default: inventory only. No sensor API, process inspection, payload reads or deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 30_000_000
OUTPUT_NAME = "flight-pace-and-motor-capacity-20261030-t1732"
RUN_NAME = re.compile(r"^20261030-t(\d+)(?:-|$)")
STAGE_NAME = re.compile(r"^(?:attempt|candidate|transit)-\d{2}$")
PREACTIVATION_METADATA = re.compile(r"^(?:attempt-\d{2}-before|transit-\d{2}-assignment)\.json$")
SOURCE_SUFFIXES = {".py", ".json", ".lua"}
NESTED_METADATA = {
    "manifest.json",
    "result.json",
    "request.json",
    "api-receipt.json",
    "config.json",
    "configuration.json",
}
TOP_METADATA = NESTED_METADATA | {
    "receipt.json",
    "report.json",
    "validation.json",
    "final-validation.json",
    "partial-result.json",
    "references.json",
    "inputrefs.json",
    "input-references.json",
    "identity.json",
    "collection-contract.json",
    "final-collection-contract.json",
    "pace30-role-contract.json",
    "motor-loader-shape-1715.json",
    "transit-launcher-proposal-1718.json",
    "audit-development-notes.json",
    "review-development-notes.json",
    "capability-supplement.json",
    "prefix-length-erratum.json",
    "prefix-length-erratum-receipt.json",
    "supplement-receipt.json",
    "mock-task-status.json",
    "mock-status.json",
    "assignment-before-activation.json",
    "live-child.json",
    "batch-registration.json",
    "registered-plan.json",
    "transit-plan.json",
}
RUNTIME_SOURCES = {
    "audit.py", "audit-v2.py", "combine.py", "review.py", "supplement.py", "installed-before.lua"
}
PAYLOAD_SUFFIXES = {
    ".jsonl",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".zip",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".png",
    ".bin",
    ".raw",
    ".dds",
}
FORBIDDEN_DIRECTORIES = {"camera", "vision", "rawslots", "raw-slots", "raw_slots", "frames"}
DERIVED_FIGURE = "docs/figures/flight-pace-diagnostic-20261030.png"
CURRENT_SOURCE_WHITELIST = (
    "AGENTS.md",
    "README.md",
    "docs/README.md",
    "docs/PROJECT_CHARTER.md",
    "docs/CONTEXT.md",
    "docs/research-plan.md",
    "docs/implementation-status.md",
    "docs/flight-pace-data-and-motor-learning-20261030.md",
    "docs/flight-motor-boundary-learning-20261030.md",
    DERIVED_FIGURE,
    "configs/flight-new-diverse-corpus-20261030.json",
    "configs/fly-flight-mechanism-plan-20261029.json",
    "configs/flight-motor-boundary-training-contract-20261030.json",
    "configs/flight-motor-boundary-cpu-qualification-20261030.json",
    "configs/flight-motor-boundary-cpu-qualification-v2-20261030.json",
    "configs/flight-motor-boundary-cpu-qualification-v3-20261030.json",
    "configs/flight-pace30-role-contract-20261030.json",
    "configs/flight-pace30-diagnostic-pilot-20261030.json",
    "configs/flight-pace30-diagnostic-pilot-v2-20261030.json",
    "configs/flight-pace30-train-batch-20261030.json",
    "configs/flight-pace-region-transit-20261030.json",
    "scripts/archive_pace_motor_evidence.py",
    "scripts/audit_ac_motor_regimes.py",
    "scripts/qualify_trainable_maneuver_boundary.py",
    "scripts/record_pace_task.py",
    "scripts/record_ac_engine_ai_probe.py",
    "scripts/control_ac_session.py",
    "scripts/transit_to_pace_region.py",
    "src/flybrain_drive/flight_maneuver_learning.py",
    "src/flybrain_drive/prospective_corpus.py",
    "src/flybrain_drive/teacher_pace.py",
    "tests/test_archive_pace_motor_evidence.py",
    "tests/test_flight_maneuver_learning.py",
    "tests/test_prospective_corpus.py",
    "tests/test_teacher_pace.py",
    "tests/test_transit_to_pace_region.py",
    "integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua",
)
FACT_FIELDS = {
    "fit_allowed",
    "goal_achieved",
    "student_deployed",
    "training_updates",
    "driving_fit_updates",
    "driving_training_updates",
    "student_valid_laps",
    "visual_student_valid_laps",
    "valid_laps",
    "optimizer_updates",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        raise ValueError(f"Links/junctions are not archive sources: {path}")
    resolved.relative_to(root.resolve(strict=True))
    return resolved


@dataclass(frozen=True)
class Entry:
    source: Path
    destination: str
    kind: str
    byte_count: int
    sha256: str

    def reference(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "destination": self.destination,
            "kind": self.kind,
            "bytes": self.byte_count,
            "sha256": self.sha256,
        }


def make_entry(source: Path, destination: str, kind: str, root: Path) -> Entry:
    source = inside(source, root)
    if not source.is_file() or source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Not a small regular file: {source}")
    data = source.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"File grew above the per-file limit: {source}")
    return Entry(source, destination, kind, len(data), digest(data))


def selected_runs(runtime_root: Path) -> list[Path]:
    result = []
    for path in sorted(runtime_root.iterdir()):
        match = RUN_NAME.match(path.name)
        if match and 1704 <= int(match[1]) <= 1731 and path.is_dir():
            inside(path, runtime_root)
            result.append(path)
    return result


def runtime_files(run: Path):
    """Visit only metadata containers; camera/payload directories are never entered."""

    def visit(folder: Path, top: bool = False, snapshots: bool = False):
        inside(folder, run)
        for path in sorted(folder.iterdir()):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError(f"Links/junctions are not archive sources: {path}")
            name = path.name.lower()
            if path.is_dir():
                if name in FORBIDDEN_DIRECTORIES:
                    continue
                if snapshots:
                    yield from visit(path, snapshots=True)
                elif name == "executed-inputs":
                    yield from visit(path, snapshots=True)
                elif STAGE_NAME.fullmatch(name):
                    yield from visit(path, top=True)
                elif name in {"recording", "control"}:
                    yield from visit(path)
            elif path.is_file():
                if snapshots:
                    if path.suffix.lower() in SOURCE_SUFFIXES:
                        yield path, "executed_source_snapshot"
                elif top and (
                    (name.startswith("executed") and path.suffix == ".py")
                    or name in RUNTIME_SOURCES
                ):
                    yield path, "executed_source_snapshot"
                elif name in (TOP_METADATA if top else NESTED_METADATA) or (
                    top and (path.suffix.lower() == ".md" or PREACTIVATION_METADATA.fullmatch(name))
                ):
                    yield path, "runtime_metadata"

    yield from visit(run, top=True)


def collect_entries(
    repo: Path, runtime_root: Path, whitelist: tuple[str, ...] = CURRENT_SOURCE_WHITELIST
) -> tuple[list[Entry], list[str]]:
    entries = []
    runs = selected_runs(runtime_root)
    if not runs:
        raise ValueError("No exact t1704..t1731 runtime directories found")
    for run in runs:
        for source, kind in runtime_files(run):
            relative = source.relative_to(run).as_posix()
            if kind == "executed_source_snapshot":
                destination = f"executed-snapshots/{run.name}/{relative}"
            else:
                destination = f"runtime-metadata/{run.name}/{relative}"
            entries.append(make_entry(source, destination, kind, run))
    for relative in whitelist:
        if relative != DERIVED_FIGURE and Path(relative).suffix.lower() not in SOURCE_SUFFIXES | {
            ".md"
        }:
            raise ValueError(f"Not an approved current source type: {relative}")
        kind = (
            "derived_figure_not_camera"
            if relative == DERIVED_FIGURE
            else "current_source_at_archive_time"
        )
        entries.append(make_entry(repo / relative, f"current-sources/{relative}", kind, repo))
    destinations = [entry.destination.casefold() for entry in entries]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate archive destinations")
    if sum(entry.byte_count for entry in entries) > MAX_TOTAL_BYTES:
        raise ValueError("Copied metadata/source exceeds 30 MB")
    return entries, [run.name for run in runs]


def metadata_evidence(entries: list[Entry]) -> tuple[list[dict], list[dict]]:
    """Record declarations verbatim; never resolve, stat, read, hash or copy payloads."""
    references, facts = [], []

    def walk(value: Any, pointer: str, entry: Entry):
        if isinstance(value, dict):
            for key, child in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                child_pointer = f"{pointer}/{escaped}"
                path_key = Path(str(key)).suffix.lower() in PAYLOAD_SUFFIXES
                if path_key:
                    references.append(
                        {
                            "metadata_source": str(entry.source),
                            "metadata_sha256": entry.sha256,
                            "json_pointer": child_pointer,
                            "declared_path": str(key),
                            "declared_value": child,
                            "payload_opened": False,
                        }
                    )
                walk(child, child_pointer, entry)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{pointer}/{index}", entry)
        elif isinstance(value, str):
            leaf = pointer.rsplit("/", 1)[-1].lower()
            normalized = value.replace("\\", "/").lower()
            is_path = Path(normalized).suffix in PAYLOAD_SUFFIXES or any(
                f"/{part}/" in normalized for part in FORBIDDEN_DIRECTORIES
            )
            is_hash = bool(re.fullmatch(r"[a-fA-F0-9]{64}", value)) and any(
                term in leaf
                for term in (
                    "event",
                    "observ",
                    "native",
                    "camera",
                    "slot",
                    "frame",
                    "checkpoint",
                    "episode",
                    "payload",
                )
            )
            is_recording_reference = leaf in {"recording", "recording_run", "source_run"} and (
                "runtime-runs" in normalized or "/recording" in normalized
            )
            if is_path or is_hash or is_recording_reference:
                references.append(
                    {
                        "metadata_source": str(entry.source),
                        "metadata_sha256": entry.sha256,
                        "json_pointer": pointer,
                        "declared_value": value,
                        "payload_opened": False,
                        "payload_sha256_independently_recomputed": False,
                    }
                )

    for entry in entries:
        if entry.kind != "runtime_metadata" or entry.source.suffix.lower() != ".json":
            continue
        data = entry.source.read_bytes()
        if digest(data) != entry.sha256:
            raise ValueError(f"Source changed during inventory: {entry.source}")
        value = json.loads(data.decode("utf-8-sig"))
        walk(value, "", entry)
        if entry.source.name == "result.json" and isinstance(value, dict):
            status = value.get("status")
            terminal = isinstance(status, str) and status.lower().startswith(
                ("completed", "passed", "failed", "rejected", "stopped", "aborted")
            )
            if terminal:
                for key in sorted(FACT_FIELDS & value.keys()):
                    facts.append(
                        {
                            "source": str(entry.source),
                            "source_sha256": entry.sha256,
                            "status": status,
                            "field": key,
                            "value": value[key],
                        }
                    )
    return references, facts


def archive(
    entries: list[Entry],
    run_names: list[str],
    output: Path,
    *,
    all_owned_jobs_terminal: bool,
    final_docs_reviewed: bool,
) -> dict:
    if not all_owned_jobs_terminal or not final_docs_reviewed:
        raise ValueError("Owner must confirm all owned jobs terminal and final documents reviewed")
    if output.exists():
        raise FileExistsError(f"Archive output already exists: {output}")
    references, facts = metadata_evidence(entries)
    manifest = {
        "schema": "flybrain.pace_motor_small_evidence_archive.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_directories": run_names,
        "owner_attestations": {"all_owned_jobs_terminal": True, "final_docs_reviewed": True},
        "per_file_limit_bytes": MAX_FILE_BYTES,
        "total_limit_bytes": MAX_TOTAL_BYTES,
        "copies": [entry.reference() for entry in entries],
        "research_facts_from_terminal_results_only": facts,
        "missing_research_fields": "Unknown; never filled with inferred zeros or success claims.",
        "fit_permission_granted_by_archiving": False,
        "current_sources_are_not_historical_executed_sources": True,
    }
    generated = {
        "archive-manifest.json": encoded(manifest),
        "payload-references.json": encoded(
            {"declarations_only": True, "payload_files_opened": 0, "references": references}
        ),
        "README.md": (
            "# Pace and motor evidence archive\n\n"
            "This archive preserves small metadata and source files. "
            "It is not a dataset copy or a driving qualification. "
            "Current sources and historical executed snapshots are stored separately. "
            "Payload paths and hashes are original declarations only; "
            "no payload was opened or copied. "
            "The named PNG is a derived diagnostic figure, not a camera frame.\n\n"
            "The full goal connects fly compound-eye/motion vision and flight-derived "
            "balance/motor circuits to a car as a virtual body. Body tilt, rotation, "
            "load and tyre state feed back into steering, acceleration and braking, "
            "with valid autonomous laps and faster racing lines/laptimes still required. "
            "Archiving adds no fit permission, learned-driving claim, visual valid lap, "
            "or whole-goal achievement. Factual values are quoted only from terminal "
            "result metadata with source hashes; absent values remain unknown.\n"
        ).encode("utf-8"),
    }
    if any(len(data) > MAX_FILE_BYTES for data in generated.values()):
        raise ValueError("Generated metadata exceeds the per-file limit")
    result = {
        "status": "completed_verified_small_evidence_archive",
        "copied_files": len(entries),
        "all_copied_hashes_verified": True,
        "copied_bytes": sum(entry.byte_count for entry in entries),
        "payload_files_copied": 0,
        "payload_files_opened": 0,
        "controls_sent": 0,
        "fit_permission_granted": False,
        "generated_metadata_sha256": {name: digest(data) for name, data in generated.items()},
    }
    result_data = encoded(result)
    total = (
        sum(entry.byte_count for entry in entries)
        + sum(map(len, generated.values()))
        + len(result_data)
    )
    if total > MAX_TOTAL_BYTES:
        raise ValueError("Total copied plus generated metadata exceeds 30 MB")
    output.mkdir(parents=False, exist_ok=False)
    verified = 0
    try:
        for entry in entries:
            data = entry.source.read_bytes()
            if len(data) != entry.byte_count or digest(data) != entry.sha256:
                raise ValueError(f"Source changed before copy: {entry.source}")
            relative = Path(entry.destination)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe destination: {entry.destination}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(data)
            if digest(destination.read_bytes()) != entry.sha256:
                raise ValueError(f"Copy hash mismatch: {destination}")
            verified += 1
        for name, data in generated.items():
            with (output / name).open("xb") as handle:
                handle.write(data)
            if digest((output / name).read_bytes()) != digest(data):
                raise ValueError(f"Generated metadata hash mismatch: {name}")
        with (output / "result.json").open("xb") as handle:
            handle.write(result_data)
        assert json.loads((output / "result.json").read_text(encoding="utf-8")) == result
    except Exception as exc:
        failure = output / "archive-failure.json"
        with failure.open("xb") as handle:
            handle.write(
                encoded(
                    {
                        "status": "failed_partial_archive_preserved",
                        "error": str(exc),
                        "verified_copies_before_failure": verified,
                        "deletion_performed": False,
                    }
                )
            )
        raise
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Create the archive; default only inventories"
    )
    parser.add_argument("--all-owned-jobs-terminal", action="store_true")
    parser.add_argument("--final-docs-reviewed", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    runtime_root = Path(os.environ["LOCALAPPDATA"]) / "flybrain" / "runtime-runs"
    output = repo / "runs" / OUTPUT_NAME
    entries, names = collect_entries(repo, runtime_root)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "inventory_only_no_archive_created",
                    "runtime_directories": names,
                    "selected_files": len(entries),
                    "selected_bytes": sum(e.byte_count for e in entries),
                    "output": str(output),
                    "payloads_opened": 0,
                },
                indent=2,
            )
        )
        return 0
    result = archive(
        entries,
        names,
        output,
        all_owned_jobs_terminal=args.all_owned_jobs_terminal,
        final_docs_reviewed=args.final_docs_reviewed,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "result_sha256": digest((output / "result.json").read_bytes()),
                **result,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
