"""Synthetic lifecycle sequencing only; no simulator, desktop or driver access."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prepare_ac_episode as prepare


@pytest.mark.parametrize('preserve', [False, True])
def test_continuation_skips_restart_but_keeps_stationary_calibration_and_first_gear(
        tmp_path, monkeypatch, preserve):
    calls = []
    values = list(prepare.PHYSICS_PREFIX.unpack(bytes(prepare.PHYSICS_PREFIX.size)))
    values[4] = 2
    sample = SimpleNamespace(prefix_bytes=prepare.PHYSICS_PREFIX.pack(*values))
    monkeypatch.setattr(prepare, 'stationary_sample', lambda: sample)
    monkeypatch.setattr(prepare, 'physics_record', lambda _: {'synthetic_stationary': True})

    def begin(path, config):
        root = Path(path)
        root.mkdir()
        return root

    def child(command, check):
        assert check
        calls.append(command)
        root = Path(command[-1])
        root.mkdir()
        result = dict(status='completed')
        if command[2] == 'scripts.control_ac_session' and command[4] == 'status':
            result.update(status='read_only', session={
                'paused': root.name == 'status', 'live': root.name != 'status', 'in_menu': False})
        if command[2] == 'scripts.calibrate_ac_sdk_steering':
            result['producer_session_id'] = 'synthetic-test'
        prepare.write_json(root/'result.json', result)

    monkeypatch.setattr(prepare, 'begin_run', begin)
    monkeypatch.setattr(prepare.subprocess, 'run', child)
    root = tmp_path/'run'
    argv = ['prepare', '--execute', '--output', str(root)]
    if preserve:
        argv.append('--preserve-position')
    monkeypatch.setattr(sys, 'argv', argv)
    prepare.main()
    stages = [Path(call[-1]).name for call in calls]
    assert stages == (['status', 'resume']
                      + (['continuation-status'] if preserve else ['restart', 'start'])
                      + ['calibration'])
    result = json.loads((root/'result.json').read_text())
    assert result['ready'] and result['position_preserved'] == preserve
    assert result['ui_actions_sent'] == 0
    assert (root/'ready-native.json').exists()
