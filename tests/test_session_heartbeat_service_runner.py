"""The live qualifier defaults to inventory; no game or worker is started here."""

import json
import sys

import pytest

from scripts import qualify_session_heartbeat_service as runner


def test_default_inventory_cannot_read_game_or_start_the_live_qualifier(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['qualifier', '--mode', 'stationary-smoke',
        '--expected-engine-pid', '24200', '--expected-generation', 'synthetic',
        '--expected-epoch', '0', '--expected-installed-sha256', 'a'*64,
        '--output', 'not-created'])
    monkeypatch.setattr(runner, 'run', lambda args: pytest.fail('Default may not run live'))
    runner.main()
    value = json.loads(capsys.readouterr().out)
    assert value['live_actions'] == 0 and value['mode'] == 'inventory_only'


def test_only_existing_bounded_diagnostic_profiles_are_constructed(tmp_path):
    three = runner.collection_arguments('camera3', tmp_path/'three')
    pace = runner.collection_arguments('pace30', tmp_path/'pace')
    assert three[three.index('--duration-seconds')+1] == '3'
    assert three[three.index('--speed-hint-kmh')+1] == '60'
    assert '--pace-task' not in three
    assert pace[pace.index('--duration-seconds')+1] == '30' and '--pace-task' in pace
    for args in (three, pace):
        assert '--same-copy-native-authority' in args and '--source-timing' in args
    with pytest.raises(ValueError):
        runner.collection_arguments('unbounded', tmp_path)


@pytest.mark.parametrize('flag', [True, False])
def test_paused_frozen_native_flag_is_preserved_without_fresh_ownership_admission(flag):
    result = runner.frozen_native_baseline(dict(packet_id=410, builtin_ai_controlled=flag,
        prefix_sha256='a'*64, received_monotonic_seconds=123.))
    assert result['builtin_ai_controlled'] is flag and result['packet_id'] == 410
    assert result['prefix_sha256'] == 'a'*64 and result['frozen_paused_packet']
    assert result['fresh_AI_ownership_claimed'] is False and result['actor_input'] is False
