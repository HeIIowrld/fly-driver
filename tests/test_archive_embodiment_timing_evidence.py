"""Explicit selection and rejection only; never execute an evidence archive."""

import json
from pathlib import Path

import pytest

from scripts import archive_embodiment_timing_evidence as tool


def put(path, value='{}'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding='utf8')
    return path


def fixture(tmp_path):
    repo, runtime = tmp_path/'repo', tmp_path/'runtime'
    (repo/'runs').mkdir(parents=True)
    runtime.mkdir()
    put(repo/'README.md', 'current documentation')
    put(runtime/'20261030-t1755-test/result.json',
        '{"status":"completed","fit_allowed":false,"original":"missing.pt"}')
    return repo, runtime, dict(schema=tool.SCHEMA, runtime_entries=[dict(
        source='20261030-t1755-test/result.json',
        destination='runtime-metadata/20261030-t1755-test/result.json', kind='runtime_metadata')],
        current_source_whitelist=['README.md'])


def test_default_full_inventory_never_walks_payloads_or_creates_archive(
        tmp_path, monkeypatch, capsys):
    repo, runtime, plan = fixture(tmp_path)
    manifest = put(tmp_path/'selection.json', json.dumps(plan))
    original = Path.read_bytes

    def read(path):
        assert path.suffix not in {'.pt', '.npz', '.jsonl', '.png'}
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    monkeypatch.setattr(Path, 'iterdir', lambda self: pytest.fail('implicit traversal'))
    monkeypatch.setattr(tool, 'archive', lambda *a, **kw: pytest.fail('archive not authorized'))
    assert tool.main(['--manifest', str(manifest), '--output', 'runs/explicit-test'],
                     repo=repo, runtime_root=runtime) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory['status'] == 'inventory_only_no_archive_created'
    assert len(inventory['entries']) == 2
    assert inventory['payload_declarations'][0]['declared_value'] == 'missing.pt'
    assert inventory['terminal_result_facts'][0]['value'] is False
    assert not (repo/'runs/explicit-test').exists()


@pytest.mark.parametrize('source', [
    '20261030-t1755-test/events.jsonl', '20261030-t1755-test/model.pt',
    '20261030-t1755-test/observations.npz', '20261030-t1755-test/frame.png',
    '20261030-t1755-test/camera/result.json', '20261030-t1755-test/vision/manifest.json',
    '20261030-t1755-test/calibration-decoded.json',
    '20261030-t1755-test/calibration-trajectory.json',
    '20261030-t1755-test/quiet-trajectories.json',
    '20261030-t1755-test/validation-trajectories.json',
    '20261030-t1755-test/cpu-recomputed.json', '20261030-t1755-test/gpu-repeat-02.json',
    '20261030-t1755-test/raw-baselines.json', '20261030-t1755-test/samples.json',
])
def test_explicit_payload_and_original_json_traces_rejected_before_open(tmp_path, source):
    repo, runtime, plan = fixture(tmp_path)
    plan['runtime_entries'][0]['source'] = source
    with pytest.raises(ValueError, match='Payload|payload|Camera'):
        tool.collect_explicit_entries(repo, runtime, plan)


@pytest.mark.parametrize('name', [
    'native-motion-samples.json', 'native_motion_sample.JSON',
    'run-samples-v2.json', 'prefix.sample.before.json', 'archived-samples2.json',
    'native-before.json', 'native-before-activation.json', 'native-after.json',
    'native_after.json', 'native-body-snapshot.json', 'native-ai-before.json',
    'cpu-state-replay.json', 'v2-cpu-state-replay-original.json',
    'gpu_state_replay.json', 'sensor-snapshot.json',
])
def test_nested_control_json_payload_families_never_reach_source_open(
        tmp_path, monkeypatch, name):
    repo, runtime, plan = fixture(tmp_path)
    plan['runtime_entries'][0]['source'] = '20261030-t1755-test/control/'+name
    monkeypatch.setattr(tool, 'source_file', lambda *a: pytest.fail('payload source accessed'))
    monkeypatch.setattr(tool, 'make_entry', lambda *a: pytest.fail('payload source opened'))
    with pytest.raises(ValueError, match='payload'):
        tool.collect_explicit_entries(repo, runtime, plan)


def test_root_selection_must_be_closed_explicit_lists(tmp_path, monkeypatch):
    repo, runtime, plan = fixture(tmp_path)
    plan['runtime_entries'] = '**/*.json'
    monkeypatch.setattr(tool, 'source_file', lambda *a: pytest.fail('implicit source traversal'))
    with pytest.raises(ValueError, match='manifest schema'):
        tool.collect_explicit_entries(repo, runtime, plan)


def test_native_ai_fact_metadata_is_not_a_sensor_snapshot(tmp_path):
    repo, runtime, plan = fixture(tmp_path)
    name = '20261030-t1755-test/control/native_ai_result.json'
    put(runtime/name, '{"status":"completed","native_ai":false}')
    plan['runtime_entries'][0]['source'] = name
    entries, _ = tool.collect_explicit_entries(repo, runtime, plan)
    assert entries[0].source.name == 'native_ai_result.json'


@pytest.mark.parametrize('field,value', [
    ('source', '../outside.json'), ('source', 'C:/private/result.json'),
    ('source', '20261030-t1755-test/../outside.json'),
    ('source', '20261030-t1755-test/*.json'), ('destination', '../outside.json'),
    ('destination', 'C:/outside/result.json'), ('destination', 'runtime-metadata/../result.json'),
])
def test_source_and_destination_escape_or_glob_rejected(tmp_path, field, value):
    repo, runtime, plan = fixture(tmp_path)
    plan['runtime_entries'][0][field] = value
    with pytest.raises(ValueError, match='relative'):
        tool.collect_explicit_entries(repo, runtime, plan)


def test_executed_python_and_markdown_names_need_no_broad_filename_scan(tmp_path):
    repo, runtime, plan = fixture(tmp_path)
    for name in ['build-contract.py', 'review.py', 'audit.py', 'cleanup-source-note.py',
                 'executed-figure.py', 'historical-review.md', 'executed-cleanup.ps1']:
        put(runtime/'20261030-t1755-test'/name, 'explicit historical source')
        plan['runtime_entries'].append(dict(source='20261030-t1755-test/'+name,
            destination='executed-snapshots/20261030-t1755-test/'+name,
            kind='executed_source_snapshot'))
    entries, _ = tool.collect_explicit_entries(repo, runtime, plan)
    assert sum(entry.kind == 'executed_source_snapshot' for entry in entries) == 7


def test_only_explicit_current_derived_png_is_allowed(tmp_path):
    repo, runtime, plan = fixture(tmp_path)
    put(repo/tool.DERIVED_FIGURE, 'synthetic derived figure placeholder')
    plan['current_source_whitelist'].append(dict(source=tool.DERIVED_FIGURE,
        destination='current-sources/'+tool.DERIVED_FIGURE, kind='derived_figure_not_camera'))
    entries, _ = tool.collect_explicit_entries(repo, runtime, plan)
    assert entries[-1].kind == 'derived_figure_not_camera'
    plan['current_source_whitelist'][-1]['source'] = 'docs/figures/other.png'
    with pytest.raises(ValueError, match='whitelist'):
        tool.collect_explicit_entries(repo, runtime, plan)


def test_casefold_duplicate_and_foreign_output_rejected(tmp_path):
    repo, runtime, plan = fixture(tmp_path)
    plan['runtime_entries'].append(dict(plan['runtime_entries'][0]))
    with pytest.raises(ValueError, match='distinct'):
        tool.collect_explicit_entries(repo, runtime, plan)
    with pytest.raises(ValueError, match='repo/runs'):
        tool.checked_output(repo, str(tmp_path/'foreign-output'))


def test_symlink_sources_rejected_without_copy(tmp_path):
    repo, runtime, plan = fixture(tmp_path)
    source = runtime/'20261030-t1755-test/result.json'
    link = source.with_name('alias.json')
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip('Host does not grant symlink creation in synthetic fixture')
    plan['runtime_entries'][0]['source'] = '20261030-t1755-test/alias.json'
    with pytest.raises(ValueError, match='Symlink'):
        tool.collect_explicit_entries(repo, runtime, plan)


def test_execute_without_both_owner_attestations_never_calls_archiver(tmp_path, monkeypatch):
    repo, runtime, plan = fixture(tmp_path)
    manifest = put(tmp_path/'selection.json', json.dumps(plan))
    monkeypatch.setattr(tool, 'archive', lambda *a, **kw: pytest.fail('not authorized'))
    with pytest.raises(ValueError, match='Root owner'):
        tool.main(['--manifest', str(manifest), '--output', 'runs/not-created', '--execute'],
                  repo=repo, runtime_root=runtime)
    assert not (repo/'runs/not-created').exists()
