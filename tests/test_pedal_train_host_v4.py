"""NEW-only hostV4 admission tests; actual diagnostics are refusal/schema fixtures.

No role is assigned to real data. Synthetic roles exist only below pytest tmp_path.
No NPZ/pixels/model/maps/control/source-clock queries are opened or performed.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from flybrain_drive.frame_persistence_profile_v4 import HostCopyProfile
from flybrain_drive.pedal_train_candidate_host_v4 import (
    SOURCE_FILES,
    SOURCE_SEEDS,
    local_import_closure,
    sha_file,
    validate_declaration,
)
from flybrain_drive.pedal_train_loader_host_v4 import NATIVE_SOURCE_FIELDS, load_candidate
from scripts.prepare_pedal_train_collector_host_v4 import BASE, TARGET, build, inverse
from scripts.prepare_pedal_train_host_v4 import BASES
from scripts.prepare_pedal_train_host_v4 import build as build_all

_spec = importlib.util.spec_from_file_location(
    'synthetic_host_v4_fixtures', Path(__file__).with_name('test_pedal_train_host_v4_inherited.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
RUNTIME = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')


def test_all_derivatives_exact_and_historical_bases_unchanged():
    for path, expected in BASES.items():
        assert sha_file(path) == expected
    for path, raw in build_all().items():
        assert path.read_bytes() == raw
    base, parent = BASE.read_bytes(), TARGET.read_bytes()
    assert parent == build(base) and inverse(parent) == base
    assert b'len(events) != result[\'observations\']' in Path(
        'src/flybrain_drive/pedal_train_loader.py').read_bytes()


def test_effective_closure_and_exact_ten_child_source_fields():
    source = Path('scripts/record_ac_engine_ai_probe_host_v4.py').read_text(encoding='utf8')
    assignment = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'SAME_COPY_SOURCE_FILES'
                              for t in node.targets))
    assert NATIVE_SOURCE_FIELDS == set(ast.literal_eval(assignment.value))
    assert len(NATIVE_SOURCE_FIELDS) == 10
    assert NATIVE_SOURCE_FIELDS <= set(SOURCE_FILES)
    assert set(local_import_closure(SOURCE_SEEDS)) <= set(SOURCE_FILES)
    assert {
        'src/flybrain_drive/ac_shared_texture_host_v4.py',
        'src/flybrain_drive/collection_cleanup_v4.py',
        'src/flybrain_drive/frame_persistence_profile_v3.py',
        'src/flybrain_drive/immutable_frame_writer.py',
        'src/flybrain_drive/ac_csp_clock.py',
        'scripts/prepare_pedal_train_host_v4.py',
        'scripts/prepare_frame_host_v4.py',
    } <= set(SOURCE_FILES)


def test_real_draft_stays_unsealed_and_host_v4_gate_cannot_be_truthy_alias():
    path = Path('configs/flight-pedal-new-train-candidate-host-v4-20261030.json')
    draft = json.loads(path.read_bytes())
    assert draft['nonce'] is None and not any(draft['bindings'].values())
    assert draft['source_sha256'] == {} and draft['fit_allowed'] is False
    assert draft['actual_pipeline_qualification_completed'] is False
    validate_declaration(draft, sealed=False)
    value = _fixture.declaration()
    for flag in (False, None, 1, 'true'):
        with pytest.raises(ValueError, match='Actual reviewed hostV4'):
            validate_declaration(dict(value, actual_pipeline_qualification_completed=flag))
    for key, wrong in [('profile', 'configs/ac-frame-persistence-buffered32-20261030.json'),
                       ('host_copy_profile_schema',
                        'flybrain.frame_persistence.bounded_profile.v1')]:
        with pytest.raises(ValueError, match='hostV4 profile'):
            validate_declaration(dict(value, **{key: wrong}))


@pytest.mark.parametrize('kind', ['missing', 'bool', 'observations_alias', 'wrong_count'])
def test_only_actual_decisions_count_schema_is_accepted(tmp_path, monkeypatch, kind):
    _fixture.test_complete_synthetic_file_loader_uses_metadata_only(tmp_path, monkeypatch)
    root = tmp_path/'future-train-candidate'
    path = root/'recording/result.json'
    value = json.loads(path.read_bytes())
    if kind == 'missing':
        del value['decisions']
    elif kind == 'bool':
        value['decisions'] = True
    elif kind == 'observations_alias':
        value['observations'] = value.pop('decisions')
    else:
        value['decisions'] += 1
    path.write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError, match='decisions count|original observation'):
        load_candidate(root, expected_capture_manifest_sha256=sha_file(root/'manifest.json'))


@pytest.mark.parametrize('where', ['manifest', 'result', 'argument'])
def test_exact_host_v4_profile_required_in_both_original_child_records(
        tmp_path, monkeypatch, where):
    _fixture.test_complete_synthetic_file_loader_uses_metadata_only(tmp_path, monkeypatch)
    root = tmp_path/'future-train-candidate'
    path = root/'recording'/('result.json' if where == 'result' else 'manifest.json')
    value = json.loads(path.read_bytes())
    if where == 'result':
        value['frame_persistence_profile']['original_actor_interval_seconds'] = .081
    elif where == 'manifest':
        value['config']['frame_persistence_profile']['original_camera_freshness_seconds'] = .151
    else:
        value['config']['arguments']['frame_persistence_profile'] = (
            'configs/ac-frame-persistence-buffered32-20261030.json')
    path.write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError, match='hostV4|hostV432'):
        load_candidate(root, expected_capture_manifest_sha256=sha_file(root/'manifest.json'))


@pytest.mark.parametrize('name,count', [
    ('20261030-t1845-host-v4-heartbeat-evidence-camera3', 72),
    ('20261030-t1849-host-v4-heartbeat-evidence-pace30', 703),
])
def test_saved_actual_diagnostic_schema_without_role_promotion(name, count, monkeypatch):
    root = RUNTIME/name
    manifest = root/'manifest.json'
    actual = json.loads((root/'recording/result.json').read_bytes())
    assert actual['decisions'] == count and 'observations' not in actual
    HostCopyProfile.from_record(actual['frame_persistence_profile'])
    original_open = Path.open

    def no_payload(self, *args, **kwargs):
        if self.name == 'events.jsonl' or self.suffix.lower() in ('.npz', '.bmp', '.zip'):
            raise AssertionError('Diagnostic role must be rejected before payload open')
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', no_payload)
    with pytest.raises(ValueError, match='no role promotion'):
        load_candidate(root, expected_capture_manifest_sha256=sha_file(manifest))
