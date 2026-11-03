"""Explicit small evidence selection; default prints every entry, never scans payload trees.

No archive is created without --execute and both root-owner attestations. The
existing archive format/copy verification is reused without changing the old tool.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from scripts.archive_pace_motor_evidence import (
    FORBIDDEN_DIRECTORIES,
    MAX_FILE_BYTES,
    MAX_TOTAL_BYTES,
    PAYLOAD_SUFFIXES,
    Entry,
    archive,
    digest,
    inside,
    make_entry,
    metadata_evidence,
)

SCHEMA = 'flybrain.embodiment_timing.explicit_evidence_archive.v1'
DERIVED_FIGURE = 'docs/figures/flight-embodiment-v2-actual-20261030.png'
RUN = re.compile(r'^\d{8}-t\d+(?:-|$)')
SOURCE_SUFFIXES = {'.py', '.lua', '.json', '.md', '.ps1'}
METADATA_SUFFIXES = {'.json', '.md', '.ini', '.log', '.txt'}
TRACE_PREFIXES = (
    'calibration-decoded', 'calibration-trajector', 'quiet-trajector', 'validation-trajector',
    'cpu-recomputed', 'cpu-state-replay', 'gpu-state-replay', 'state-replay',
    'gpu-repeat', 'raw-baseline', 'raw-sensor', 'sensor-snapshot',
    'native-before', 'native-after', 'native-motion', 'events', 'samples', 'trajectory',
)
JSON_SAMPLE_TOKEN = re.compile(r'(?:^|[-.])samples?\d*(?:[-.]|$)')
JSON_STATE_REPLAY = re.compile(r'(?:^|-)(?:cpu-|gpu-)?state-replay(?:-|$)')
JSON_NATIVE_SNAPSHOT = re.compile(
    r'^native-(?:(?:physics|body|graphics|ai)-)?(?:before|after|snapshot)(?:-|$)'
)
REPO_ROOTS = {'docs', 'configs', 'scripts', 'src', 'integrations', 'tests'}
DESTINATION_PREFIX = {
    'runtime_metadata': 'runtime-metadata',
    'executed_source_snapshot': 'executed-snapshots',
    'current_source_at_archive_time': 'current-sources',
    'derived_figure_not_camera': 'current-sources',
}


def relative_name(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or any(c in value for c in '*?[]{}') or PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).drive
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('Exact canonical relative file path required; no glob or escape')
    return PurePosixPath(value)


def source_file(root: Path, relative: PurePosixPath) -> Path:
    current = root.resolve(strict=True)
    for part in relative.parts:
        current = current/part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Symlink/junction archive sources are forbidden')
    return inside(current, root)


def reject_payload(relative: PurePosixPath, *, derived: bool = False) -> None:
    if any(part.casefold() in FORBIDDEN_DIRECTORIES for part in relative.parts):
        raise ValueError('Camera/payload directories are never archive sources')
    if relative.suffix.lower() in PAYLOAD_SUFFIXES and not derived:
        raise ValueError('Payload suffix is forbidden, including explicit selections')
    if relative.suffix.lower() == '.json':
        normalized = relative.stem.casefold().replace('_', '-')
        # This must run before source_file/make_entry: explicit listing cannot
        # authorize original sensor traces. Root still supplies a closed,
        # content-reviewed metadata/source list; this is no broad file discovery.
        if (normalized == 'native-motion-samples'
                or JSON_SAMPLE_TOKEN.search(normalized)
                or JSON_STATE_REPLAY.search(normalized)
                or JSON_NATIVE_SNAPSHOT.match(normalized)
                or any(part.casefold().replace('_', '-').startswith(TRACE_PREFIXES)
                       for part in relative.parts)):
            raise ValueError('Original JSON observation/trajectory/snapshot traces are payloads')


def collect_explicit_entries(
    repo: Path, runtime_root: Path, plan: dict
) -> tuple[list[Entry], list[str]]:
    if (not isinstance(plan, dict) or plan.get('schema') != SCHEMA
            or set(plan) != {'schema', 'runtime_entries', 'current_source_whitelist'}
            or not isinstance(plan['runtime_entries'], list)
            or not isinstance(plan['current_source_whitelist'], list)):
        raise ValueError('Exact explicit evidence manifest schema required')
    entries, run_names = [], set()
    for item in plan['runtime_entries']:
        if not isinstance(item, dict) or set(item) - {'source', 'destination', 'kind', 'sha256'}:
            raise ValueError('Runtime entries require source/destination/kind')
        source = relative_name(item.get('source'))
        destination = relative_name(item.get('destination'))
        kind = item.get('kind')
        if (len(source.parts) < 2 or not RUN.match(source.parts[0])
                or kind not in ('runtime_metadata', 'executed_source_snapshot')
                or destination.parts[0] != DESTINATION_PREFIX[kind]):
            raise ValueError('Explicit runtime run/kind/destination partition differs')
        reject_payload(source)
        allowed = METADATA_SUFFIXES if kind == 'runtime_metadata' else SOURCE_SUFFIXES
        if source.suffix.lower() not in allowed:
            raise ValueError('Runtime entry is not small metadata or executed source')
        entry = make_entry(source_file(runtime_root, source), destination.as_posix(), kind,
                           runtime_root)
        if 'sha256' in item and entry.sha256 != item['sha256']:
            raise ValueError('Explicit expected runtime source SHA differs')
        entries.append(entry)
        run_names.add(source.parts[0])
    for item in plan['current_source_whitelist']:
        if isinstance(item, str):
            item = dict(source=item, destination='current-sources/'+item,
                        kind='current_source_at_archive_time')
        if not isinstance(item, dict) or set(item) - {'source', 'destination', 'kind', 'sha256'}:
            raise ValueError('Current whitelist entries require explicit source identity')
        source = relative_name(item.get('source'))
        destination = relative_name(item.get('destination'))
        kind = item.get('kind')
        derived = source.as_posix() == DERIVED_FIGURE and kind == 'derived_figure_not_camera'
        if (kind not in ('current_source_at_archive_time', 'derived_figure_not_camera')
                or destination.parts[0] != 'current-sources'
                or kind == 'derived_figure_not_camera' and not derived
                or source.parts[0] not in REPO_ROOTS | {'AGENTS.md', 'README.md'}):
            raise ValueError('Current source whitelist/kind/destination differs')
        reject_payload(source, derived=derived)
        if not derived and source.suffix.lower() not in SOURCE_SUFFIXES:
            raise ValueError('Current source type is not approved')
        entry = make_entry(source_file(repo, source), destination.as_posix(), kind, repo)
        if 'sha256' in item and entry.sha256 != item['sha256']:
            raise ValueError('Explicit expected current source SHA differs')
        entries.append(entry)
    destinations = [entry.destination.casefold() for entry in entries]
    if not entries or len(destinations) != len(set(destinations)):
        raise ValueError('Nonempty selection with distinct case-insensitive destinations required')
    if sum(entry.byte_count for entry in entries) > MAX_TOTAL_BYTES:
        raise ValueError('Explicit metadata/source selection exceeds 30 MB')
    return entries, sorted(run_names)


def checked_output(repo: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo/path
    runs = (repo/'runs').resolve(strict=True)
    if path.parent.resolve(strict=True) != runs or path.name in ('', '.', '..'):
        raise ValueError('Explicit unique archive output must be an immediate child of repo/runs')
    if path.exists() or path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        raise FileExistsError('Archive output already exists or is a link')
    return path


def main(argv=None, *, repo=None, runtime_root=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--all-owned-jobs-terminal', action='store_true')
    parser.add_argument('--final-docs-reviewed', action='store_true')
    args = parser.parse_args(argv)
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    runtime_root = (Path(runtime_root) if runtime_root is not None else
                    Path(os.environ['LOCALAPPDATA'])/'flybrain/runtime-runs')
    if args.execute and not (args.all_owned_jobs_terminal and args.final_docs_reviewed):
        raise ValueError('Root owner must attest terminal jobs and final documents before execute')
    manifest = Path(args.manifest)
    if (manifest.suffix.lower() != '.json' or manifest.is_symlink()
            or not manifest.is_file() or manifest.stat().st_size > MAX_FILE_BYTES):
        raise ValueError('Explicit small regular JSON selection manifest required')
    original = manifest.read_bytes()
    if len(original) > MAX_FILE_BYTES:
        raise ValueError('Selection manifest grew above small metadata limit')
    entries, names = collect_explicit_entries(repo, runtime_root, json.loads(original))
    output = checked_output(repo, args.output)
    references, facts = metadata_evidence(entries)
    inventory = dict(status='inventory_only_no_archive_created',
        selection_manifest=str(manifest.resolve()), selection_sha256=digest(original),
        runtime_directories=names, selected_files=len(entries),
        selected_bytes=sum(entry.byte_count for entry in entries), output=str(output),
        entries=[entry.reference() for entry in entries],
        payload_declarations=references, terminal_result_facts=facts,
        payloads_opened=0, controls_sent=0, fit_permission_granted=False)
    if not args.execute:
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return 0
    result = archive(entries, names, output,
                     all_owned_jobs_terminal=args.all_owned_jobs_terminal,
                     final_docs_reviewed=args.final_docs_reviewed)
    print(json.dumps(dict(output=str(output), selection_sha256=digest(original), **result),
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
