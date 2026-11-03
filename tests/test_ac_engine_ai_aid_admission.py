import sys

import pytest

from scripts import seal_ac_engine_ai_receipt_dataset as sealing


@pytest.mark.parametrize('report', [
    {}, {'camera_aid_state_verified': False, 'explicit_aid_off_frames_verified': 10},
    {'camera_aid_state_verified': True, 'explicit_aid_off_frames_verified': 0},
    {'camera_aid_state_verified': True, 'explicit_aid_off_frames_verified': True},
    {'camera_aid_state_verified': True, 'explicit_aid_off_frames_verified': 10.},
    {'camera_aid_state_verified': 1, 'explicit_aid_off_frames_verified': 10},
])
def test_strict_teacher_sealing_rejects_missing_or_ambiguous_frame_aid_proof(monkeypatch, report):
    config = dict(require_aid_off=True, sources=[dict(
        recording='recording', targets='targets', raw_audit='raw-audit')])
    contents = {'config.json': config, 'targets/result.json': {},
                'targets/manifest.json': {'config': {}}, 'raw-audit/result.json': report}
    monkeypatch.setattr(sealing, 'read', lambda path: contents[str(path).replace('\\', '/')])
    monkeypatch.setattr(sys, 'argv', ['seal', '--config', 'config.json', '--output', 'unused'])
    with pytest.raises(ValueError, match='all-frame aid-OFF audit'):
        sealing.main()


def test_teacher_aid_policy_must_be_an_explicit_boolean(monkeypatch):
    monkeypatch.setattr(sealing, 'read', lambda path: dict(require_aid_off='false'))
    monkeypatch.setattr(sys, 'argv', ['seal', '--config', 'config.json', '--output', 'unused'])
    with pytest.raises(ValueError, match='boolean aid admission'):
        sealing.main()
