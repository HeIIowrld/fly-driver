import json

import pytest

from flybrain_drive import flight_fast_stream as module
from flybrain_drive.graph import sha256


def test_parent_only_certificate_never_admits_new_dynamics(tmp_path, monkeypatch):
    config_file = tmp_path/'config.json'
    config_file.write_text('{}', encoding='utf8')
    verification = tmp_path/'verification'
    verification.mkdir()
    (verification/'result.json').write_text(json.dumps(dict(
        status='completed_coupled_stream_replay', author_and_flight_parameters_unchanged=True)),
        encoding='utf8')
    (verification/'manifest.json').write_text(json.dumps(dict(config=dict(
        fast_config_sha256=sha256(config_file)))), encoding='utf8')
    monkeypatch.setattr(module, 'fast_definition', lambda path:
        (dict(fine_neuron_tolerance=.001), {}, {}, None, None, None))
    with pytest.raises(ValueError, match='fast-stream certificate'):
        module.verified_fast_stream(config_file, verification)


def test_fast_model_cannot_be_silently_requested_by_certificate_only():
    assert module.verified_fast_plan({}) is None
    with pytest.raises(ValueError, match='without an explicit'):
        module.verified_fast_plan(dict(fast_stream_verification='unreviewed'))
