import json
import os
import time
from pathlib import Path

import pytest

from scripts.record_ac_engine_ai_probe import read_ai_phase


def test_lateral_recovery_has_its_own_fixed_profile():
    from scripts.record_ac_engine_ai_probe import expert_command

    assert expert_command(30, 15, lateral_recovery=True) == 'expert-lateral-recovery15'
    assert expert_command(30, 15) == 'expert-collect30'
    for duration, speed, overspeed in [(3, 15, False), (30, 30, False), (30, 15, True)]:
        with pytest.raises(ValueError):
            expert_command(duration, speed, overspeed, lateral_recovery=True)


def test_read_ai_phase_retries_replaced_metadata_without_mutating(tmp_path, monkeypatch):
    path = tmp_path/'status.json'
    value = dict(generation='one', paused=False, ai_controlled=True)
    path.write_text(json.dumps(value), encoding='utf8')
    stamp = time.time()-.1
    os.utime(path, (stamp, stamp))
    original = Path.open
    attempts = []

    def replaced(self, *args, **kwargs):
        attempts.append(self)
        if len(attempts) == 1:
            raise FileNotFoundError('atomic replacement gap')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', replaced)
    assert read_ai_phase(path, 'one') is True
    assert len(attempts) == 2
    with original(path, encoding='utf8') as stream:
        assert json.load(stream) == value


@pytest.mark.parametrize('value', [dict(generation='other'), dict(generation='one', error='bad')])
def test_phase_rejects_changed_generation_and_error(tmp_path, value):
    path = tmp_path/'status.json'
    path.write_text(json.dumps(value), encoding='utf8')
    with pytest.raises(ValueError):
        read_ai_phase(path, 'one')


def test_phase_does_not_treat_stale_or_missing_sensor_as_running(tmp_path):
    path = tmp_path/'status.json'
    with pytest.raises(FileNotFoundError):
        read_ai_phase(path, 'one', timeout=.01)
    path.write_text(json.dumps(dict(generation='one', paused=True, ai_controlled=False)))
    stamp = time.time()-.1
    os.utime(path, (stamp, stamp))
    assert read_ai_phase(path, 'one') is False
    os.utime(path, (1, 1))
    with pytest.raises(ValueError):
        read_ai_phase(path, 'one')


def test_default_expert_recording_uses_reviewed_student_sensor_registration():
    from flybrain_drive.ac_view_contract import validate_declared_live_view
    from flybrain_drive.graph import sha256
    from scripts.record_ac_engine_ai_probe import DEFAULT_RETINA_CONFIG

    live = json.loads(Path('configs/ac-live-view-contract.json').read_text(encoding='utf8'))
    assert DEFAULT_RETINA_CONFIG == live['retina_configuration']
    configuration = json.loads(Path(DEFAULT_RETINA_CONFIG).read_text(encoding='utf8'))
    assert validate_declared_live_view(configuration) == live
    assert sha256(DEFAULT_RETINA_CONFIG) == live['retina_configuration_file_sha256']


def test_explicit_speed_profiles_preserve_low_speed_requests():
    from scripts.record_ac_engine_ai_probe import expert_command

    assert expert_command(3, 15) == 'expert-motion-probe'
    assert expert_command(30, 15) == 'expert-collect30'
    assert expert_command(3, 30) == 'expert-motion-speed30'
    assert expert_command(30, 30) == 'expert-collect30-speed30'
    assert expert_command(30, 30, True) == 'expert-speed-recovery15'
    assert expert_command(30, 20, True) == 'expert-speed20-recovery15'
    with pytest.raises(ValueError):
        expert_command(30, 20)
    with pytest.raises(ValueError):
        expert_command(3, 30, True)
    with pytest.raises(ValueError):
        expert_command(120, 60)
def test_brake_intervention_has_a_separate_command_and_cannot_be_natural_ai_labels():
    from flybrain_drive.ac_engine_ai_labels import common_expert_settings
    from scripts.record_ac_engine_ai_probe import expert_command

    assert expert_command(30, 30, brake_pulse=True) == 'expert-brake-pulse30'
    for seconds, speed in [(3, 30), (30, 15), (30, 20)]:
        with pytest.raises(ValueError, match='Brake pulse'):
            expert_command(seconds, speed, brake_pulse=True)
    with pytest.raises(ValueError, match='separate labels'):
        common_expert_settings([dict(command='expert-brake-pulse30', speed_limit_kmh=30)])


def test_speed60_requires_new_command_and_cannot_enter_legacy_teacher_labels():
    from flybrain_drive.ac_engine_ai_labels import common_expert_settings
    from scripts.record_ac_engine_ai_probe import expert_command

    assert expert_command(3, 60) == 'expert-motion-speed60'
    assert expert_command(30, 60) == 'expert-collect30-speed60'
    with pytest.raises(ValueError):
        expert_command(30, 60, recover_to15=True)
    with pytest.raises(ValueError):
        common_expert_settings([dict(command='expert-collect30-speed60',
                                    speed_limit_kmh=60, ai_steer_multiplier=.9)])

