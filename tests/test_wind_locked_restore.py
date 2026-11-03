import json
import os
import subprocess
import sys

import pytest

from flybrain_drive.wind_locked_restore import (
    exclusive_existing_file,
    finalize_wind_transaction,
    inspect_interrupted_restore,
    inspect_wind_transaction,
    locked_restore,
    prepare_wind_transaction,
)
from flybrain_drive.wind_stimulus_config import isolated_wind_copy

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows sharing semantics required')

ORIGINAL = b'[WIND]\nDIRECTION_DEG=-1\nSPEED_KMH_MIN=0\nSPEED_KMH_MAX=0\n[RACE]\nCARS=1\n'


def test_transaction_snapshot_and_read_only_classification(tmp_path):
    import hashlib

    path, journal = tmp_path/'race.ini', tmp_path/'prepare'
    path.write_bytes(ORIGINAL)
    prepare_wind_transaction(path, expected_sha256=hashlib.sha256(ORIGINAL).hexdigest(),
                             direction_deg=90, speed_kmh=36, journal_dir=journal)
    assert path.read_bytes() == ORIGINAL
    assert inspect_wind_transaction(path, journal_dir=journal)['wind_state'] == 'original_wind'
    stimulus = (journal/'stimulus.bin').read_bytes().replace(b'CARS=1', b'CARS=2')
    path.write_bytes(stimulus)
    observed = inspect_wind_transaction(path, journal_dir=journal)
    assert observed['wind_state'] == 'stimulus_wind'
    assert observed['entire_file_matches_original'] is False
    assert observed['automatic_recovery_allowed'] is False
    conflict = stimulus.replace(b'DIRECTION_DEG=90', b'DIRECTION_DEG=180')
    path.write_bytes(conflict)
    observed = inspect_wind_transaction(path, journal_dir=journal)
    assert observed['wind_state'] == 'conflict_or_partial'
    assert path.read_bytes() == conflict
    (journal/'original.bin').write_bytes(b'altered')
    with pytest.raises(ValueError, match='identity'):
        inspect_wind_transaction(path, journal_dir=journal)


def test_transaction_refuses_wrong_hash_or_unsupported_stimulus(tmp_path):
    path = tmp_path/'race.ini'
    path.write_bytes(ORIGINAL)
    with pytest.raises(ValueError, match='exact original'):
        prepare_wind_transaction(path, expected_sha256='0'*64, direction_deg=90, speed_kmh=36,
                                 journal_dir=tmp_path/'bad-hash')
    with pytest.raises(ValueError, match='contract'):
        prepare_wind_transaction(path, expected_sha256='0'*64, direction_deg=90, speed_kmh=0,
                                 journal_dir=tmp_path/'bad-stimulus')
    assert path.read_bytes() == ORIGINAL


@pytest.mark.parametrize('state', ['original', 'stimulus', 'conflict'])
def test_finalization_handles_owned_wind_without_losing_other_changes(tmp_path, state):
    import hashlib

    path, journal = tmp_path/'race.ini', tmp_path/'prepare'
    path.write_bytes(ORIGINAL)
    prepare_wind_transaction(path, expected_sha256=hashlib.sha256(ORIGINAL).hexdigest(),
                             direction_deg=90, speed_kmh=36, journal_dir=journal)
    values = {'original': ORIGINAL, 'stimulus': (journal/'stimulus.bin').read_bytes(),
              'conflict': isolated_wind_copy(ORIGINAL, direction_deg=180, speed_kmh=36)}
    before = values[state].replace(b'CARS=1', b'CARS=2')
    path.write_bytes(before)
    final = tmp_path/'final'
    if state == 'conflict':
        with pytest.raises(ValueError, match='conflict'):
            finalize_wind_transaction(path, journal_dir=journal, finalization_dir=final,
                                      restart_outcome='new_engine_confirmed')
        assert path.read_bytes() == before
        return
    result = finalize_wind_transaction(path, journal_dir=journal, finalization_dir=final,
                                       restart_outcome='new_engine_confirmed')
    assert path.read_bytes() == ORIGINAL.replace(b'CARS=1', b'CARS=2')
    assert result['target_modified'] is (state == 'stimulus')
    assert (final/'before.bin').read_bytes() == before
    assert inspect_interrupted_restore(path, journal_dir=final)['current_state'] == (
        'matches_prewrite' if state == 'original' else 'matches_intended_result')


def test_finalization_refuses_unknown_outcome_and_duplicate_finalization(tmp_path):
    import hashlib

    path, journal, final = tmp_path/'race.ini', tmp_path/'prepare', tmp_path/'final'
    path.write_bytes(ORIGINAL)
    prepare_wind_transaction(path, expected_sha256=hashlib.sha256(ORIGINAL).hexdigest(),
                             direction_deg=90, speed_kmh=36, journal_dir=journal)
    path.write_bytes((journal/'stimulus.bin').read_bytes())
    with pytest.raises(ValueError, match='unknown outcomes'):
        finalize_wind_transaction(path, journal_dir=journal, finalization_dir=final,
                                  restart_outcome='unknown')
    assert path.read_bytes() == (journal/'stimulus.bin').read_bytes()
    finalize_wind_transaction(path, journal_dir=journal, finalization_dir=final,
                              restart_outcome='new_engine_confirmed')
    with pytest.raises(FileExistsError):
        finalize_wind_transaction(path, journal_dir=journal, finalization_dir=final,
                                  restart_outcome='new_engine_confirmed')


def test_finalization_partial_write_is_retained_for_inspection(tmp_path, monkeypatch):
    import hashlib

    import flybrain_drive.wind_locked_restore as module

    path, journal, final = tmp_path/'race.ini', tmp_path/'prepare', tmp_path/'final'
    path.write_bytes(ORIGINAL)
    prepare_wind_transaction(path, expected_sha256=hashlib.sha256(ORIGINAL).hexdigest(),
                             direction_deg=90, speed_kmh=36, journal_dir=journal)
    path.write_bytes((journal/'stimulus.bin').read_bytes())

    def interrupted(fd, content):
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, content[:len(content)//2])
        raise OSError('injected partial write')

    monkeypatch.setattr(module, '_write_target', interrupted)
    with pytest.raises(OSError, match='partial write'):
        finalize_wind_transaction(path, journal_dir=journal, finalization_dir=final,
                                  restart_outcome='new_engine_confirmed')
    status = inspect_interrupted_restore(path, journal_dir=final)
    assert status['current_state'] == 'unknown_or_external_change'
    assert status['automatic_recovery_allowed'] is False
    with pytest.raises(FileExistsError):
        finalize_wind_transaction(path, journal_dir=journal, finalization_dir=final,
                                  restart_outcome='new_engine_confirmed')


def test_lock_denies_other_file_opens_and_releases(tmp_path):
    path = tmp_path/'race.ini'
    path.write_bytes(ORIGINAL)
    with exclusive_existing_file(path):
        with pytest.raises(OSError):
            path.read_bytes()
        with pytest.raises(OSError):
            path.unlink()
    assert path.read_bytes() == ORIGINAL


def test_restore_journals_and_preserves_unrelated_edits(tmp_path):
    path = tmp_path/'race.ini'
    expected = isolated_wind_copy(ORIGINAL, direction_deg=90, speed_kmh=36)
    before = expected.replace(b'CARS=1', b'CARS=2')
    path.write_bytes(before)
    journal = tmp_path/'journal'
    result = locked_restore(path, expected_applied=expected, original=ORIGINAL, journal_dir=journal)
    assert path.read_bytes() == ORIGINAL.replace(b'CARS=1', b'CARS=2')
    assert (journal/'before.bin').read_bytes() == before
    assert json.loads((journal/'receipt.json').read_text()) == result
    assert json.loads((journal/'intent.json').read_text())['write_atomic'] is False


def test_conflicting_wind_refuses_any_target_change(tmp_path):
    path = tmp_path/'race.ini'
    expected = isolated_wind_copy(ORIGINAL, direction_deg=90, speed_kmh=36)
    conflict = isolated_wind_copy(ORIGINAL, direction_deg=180, speed_kmh=36)
    path.write_bytes(conflict)
    with pytest.raises(ValueError, match='differs'):
        locked_restore(path, expected_applied=expected, original=ORIGINAL,
                       journal_dir=tmp_path/'journal')
    assert path.read_bytes() == conflict


def test_journal_failure_happens_before_target_write(tmp_path, monkeypatch):
    import flybrain_drive.wind_locked_restore as module

    path = tmp_path/'race.ini'
    expected = isolated_wind_copy(ORIGINAL, direction_deg=90, speed_kmh=36)
    path.write_bytes(expected)
    actual = module._durable_new

    def fail_intent(destination, content):
        if destination.name == 'intent.json':
            raise OSError('injected durable intent failure')
        return actual(destination, content)

    monkeypatch.setattr(module, '_durable_new', fail_intent)
    with pytest.raises(OSError, match='injected'):
        locked_restore(path, expected_applied=expected, original=ORIGINAL,
                       journal_dir=tmp_path/'journal')
    assert path.read_bytes() == expected
    assert (tmp_path/'journal/before.bin').read_bytes() == expected


def test_receipt_failure_retains_original_prewrite_and_intent(tmp_path, monkeypatch):
    import flybrain_drive.wind_locked_restore as module

    path = tmp_path/'race.ini'
    expected = isolated_wind_copy(ORIGINAL, direction_deg=90, speed_kmh=36)
    path.write_bytes(expected)
    actual = module._durable_new

    def fail_receipt(destination, content):
        if destination.name == 'receipt.json':
            raise OSError('injected receipt failure')
        return actual(destination, content)

    monkeypatch.setattr(module, '_durable_new', fail_receipt)
    with pytest.raises(OSError, match='injected'):
        locked_restore(path, expected_applied=expected, original=ORIGINAL,
                       journal_dir=tmp_path/'journal')
    assert path.read_bytes() == ORIGINAL
    assert (tmp_path/'journal/before.bin').read_bytes() == expected
    assert (tmp_path/'journal/intent.json').exists()
    assert not (tmp_path/'journal/receipt.json').exists()
    status = inspect_interrupted_restore(path, journal_dir=tmp_path/'journal')
    assert status['current_state'] == 'matches_intended_result'
    assert status['automatic_recovery_allowed'] is False


def test_interrupted_write_is_detected_without_overwriting_external_bytes(tmp_path):
    path = tmp_path/'race.ini'
    expected = isolated_wind_copy(ORIGINAL, direction_deg=90, speed_kmh=36)
    path.write_bytes(expected)
    journal = tmp_path/'journal'
    child = '''
import os, sys
from pathlib import Path
import flybrain_drive.wind_locked_restore as m
from flybrain_drive.wind_stimulus_config import isolated_wind_copy
original = bytes.fromhex(sys.argv[3])
expected = isolated_wind_copy(original, direction_deg=90, speed_kmh=36)
def interrupt(fd, content):
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, content[:max(1, len(content)//3)])
    os.fsync(fd)
    os._exit(86)
m._write_target = interrupt
m.locked_restore(Path(sys.argv[1]), expected_applied=expected, original=original,
                 journal_dir=Path(sys.argv[2]))
'''
    process = subprocess.run([sys.executable, '-c', child, str(path), str(journal), ORIGINAL.hex()],
                             check=False, capture_output=True, timeout=30)
    assert process.returncode == 86, process.stderr.decode(errors='replace')
    partial = path.read_bytes()
    assert partial != expected and partial != ORIGINAL
    assert (journal/'before.bin').read_bytes() == expected
    assert (journal/'after.bin').read_bytes() == ORIGINAL
    status = inspect_interrupted_restore(path, journal_dir=journal)
    assert status['current_state'] == 'unknown_or_external_change'
    assert status['automatic_recovery_allowed'] is False
    assert path.read_bytes() == partial


def test_inspection_refuses_changed_journal_or_target_identity(tmp_path):
    path = tmp_path/'race.ini'
    expected = isolated_wind_copy(ORIGINAL, direction_deg=90, speed_kmh=36)
    path.write_bytes(expected)
    journal = tmp_path/'journal'
    locked_restore(path, expected_applied=expected, original=ORIGINAL, journal_dir=journal)
    other = tmp_path/'other.ini'
    other.write_bytes(ORIGINAL)
    with pytest.raises(ValueError, match='identity'):
        inspect_interrupted_restore(other, journal_dir=journal)
    (journal/'after.bin').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='identity'):
        inspect_interrupted_restore(path, journal_dir=journal)
