import json
from pathlib import Path

import pytest

from scripts import record_ac_engine_ai_probe as recorder


def proof(tmp_path, monkeypatch):
    config = tmp_path/'stream.json'
    config.write_text('{"task_speed_kmh":30}', encoding='utf8')
    root = tmp_path/'verified'
    root.mkdir()
    (root/'result.json').write_text(json.dumps(dict(status='completed_coupled_stream_replay',
        author_and_flight_parameters_unchanged=True)), encoding='utf8')
    sources = ['flight_stream', 'flyvis_camera', 'flyvis_compat', 'virtual_body',
               'flight_circuit', 'flight_commands', 'flight_navigation', 'vehicle_circuit',
               'flight_learning', 'ac_csp_vision', 'ac_csp_geometry', 'ac_native_physics',
               'sensor_contract']
    manifest = dict(config=dict(config_sha256='same', seed=863009),
                    source_sha256={'src/flybrain_drive/'+n+'.py':'same' for n in sources})
    (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    monkeypatch.setattr(recorder, 'sha256', lambda path: 'same')
    return config, root


def test_shadow_requires_completed_unchanged_replay_and_matching_code(tmp_path, monkeypatch):
    config, root = proof(tmp_path, monkeypatch)
    accepted, metadata = recorder.verified_shadow(config, root)
    assert accepted['task_speed_kmh'] == 30 and metadata['initialization_seed'] == 863009
    monkeypatch.setattr(recorder, 'sha256', lambda p:
                        'changed' if str(p).endswith('flight_stream.py') else 'same')
    with pytest.raises(ValueError, match='source changed'):
        recorder.verified_shadow(config, root)


@pytest.mark.parametrize('status,unchanged', [('failed', True),
                                            ('completed_coupled_stream_replay', False)])
def test_failed_or_mutating_replay_cannot_admit_shadow(tmp_path, monkeypatch, status, unchanged):
    config, root = proof(tmp_path, monkeypatch)
    (root/'result.json').write_text(json.dumps(dict(status=status,
        author_and_flight_parameters_unchanged=unchanged)), encoding='utf8')
    with pytest.raises(ValueError, match='Successful exact coupled replay'):
        recorder.verified_shadow(config, root)


def test_changed_runtime_configuration_cannot_use_an_old_replay(tmp_path, monkeypatch):
    config, root = proof(tmp_path, monkeypatch)
    monkeypatch.setattr(recorder, 'sha256', lambda p: 'changed' if Path(p) == config else 'same')
    with pytest.raises(ValueError, match='Successful exact coupled replay'):
        recorder.verified_shadow(config, root)
