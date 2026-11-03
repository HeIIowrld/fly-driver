"""Windows-exclusive, journaled wind rollback primitive; no engine lifecycle actions."""

import ctypes
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .wind_stimulus_config import _wind_entries, isolated_wind_copy, restore_wind_values

MAX_CONFIG_BYTES = 1024 * 1024


def digest(content):
    return hashlib.sha256(content).hexdigest()


@contextmanager
def exclusive_existing_file(path):
    """Open existing Windows file without any read/write/delete sharing."""
    if os.name != 'nt':
        raise OSError('Windows sharing contract required')
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(Path(path).resolve()), 0x80000000 | 0x40000000, 0, None, 3,
                    0x80000000, None)  # GENERIC_READ|WRITE, OPEN_EXISTING, WRITE_THROUGH
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    try:
        yield fd
    finally:
        os.close(fd)


def _durable_new(path, content):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_BINARY, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def _write_target(fd, content):
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(fd, 'wb', closefd=False) as stream:
        stream.write(content)
        stream.flush()
        os.ftruncate(fd, len(content))
        os.fsync(fd)


def prepare_wind_transaction(path, *, expected_sha256, direction_deg, speed_kmh, journal_dir):
    """Save a pinned snapshot and proposed stimulus under lock, without changing the target.

    The lock is released on return; callers must recheck before any lifecycle request.
    The snapshot does not establish whether an engine merge touches the file.
    """
    if (type(direction_deg) is not int or direction_deg not in (0, 90, 180, 270)
            or type(speed_kmh) is not int or speed_kmh not in (0, 36)
            or speed_kmh == 0 and direction_deg != 0):
        raise ValueError('Only the diagnostic candidate stimulus contract is allowed')
    journal = Path(journal_dir)
    journal.mkdir(exist_ok=False)
    with exclusive_existing_file(path) as fd:
        original = os.read(fd, MAX_CONFIG_BYTES + 1)
        if len(original) > MAX_CONFIG_BYTES or digest(original) != expected_sha256:
            raise ValueError('Bounded exact original configuration required')
        expected = isolated_wind_copy(original, direction_deg=direction_deg, speed_kmh=speed_kmh)
        _wind_entries(original)
        _durable_new(journal/'original.bin', original)
        _durable_new(journal/'stimulus.bin', expected)
        manifest = {'schema': 'flybrain.wind_transaction_prepare.v1',
                    'target': str(Path(path).resolve()), 'original_sha256': digest(original),
                    'stimulus_sha256': digest(expected), 'direction_deg': direction_deg,
                    'speed_kmh': speed_kmh, 'target_modified': False,
                    'lock_held_after_return': False, 'engine_state_verified': False}
        _durable_new(journal/'manifest.json', json.dumps(manifest, sort_keys=True).encode('utf8'))
    return manifest


def inspect_wind_transaction(path, *, journal_dir):
    """Classify wind values under lock; this observation never authorizes a write."""
    journal = Path(journal_dir)
    manifest = json.loads((journal/'manifest.json').read_text(encoding='utf8'))
    original = (journal/'original.bin').read_bytes()
    expected = (journal/'stimulus.bin').read_bytes()
    if (manifest.get('schema') != 'flybrain.wind_transaction_prepare.v1'
            or manifest.get('target') != str(Path(path).resolve())
            or manifest.get('original_sha256') != digest(original)
            or manifest.get('stimulus_sha256') != digest(expected)):
        raise ValueError('Transaction identity or saved bytes differ')
    with exclusive_existing_file(path) as fd:
        current = os.read(fd, MAX_CONFIG_BYTES + 1)
    if len(current) > MAX_CONFIG_BYTES:
        raise ValueError('Config exceeds bounded inspection budget')
    def values(data):
        return {key: entry[2] for key, entry in _wind_entries(data)[1].items()}
    now, baseline, stimulus = values(current), values(original), values(expected)
    if now == baseline and now == stimulus:
        state = 'ambiguous_original_equals_stimulus'
    elif now == baseline:
        state = 'original_wind'
    elif now == stimulus:
        state = 'stimulus_wind'
    else:
        state = 'conflict_or_partial'
    return {'wind_state': state, 'current_sha256': digest(current),
            'entire_file_matches_original': current == original,
            'target_modified': False, 'automatic_recovery_allowed': False,
            'engine_state_verified': False}


def finalize_wind_transaction(path, *, journal_dir, finalization_dir, restart_outcome):
    """Resolve a known restart without overwriting unrelated configuration changes.

    The outcome is a caller attestation, not an independently verified engine receipt.
    Unknown outcomes cannot cause a write. A failed in-place write must be inspected
    using the saved before/after images; it must never be repeated automatically.
    """
    if restart_outcome != 'new_engine_confirmed':
        raise ValueError('Confirmed new engine required; unknown outcomes are never retried')
    journal = Path(journal_dir)
    manifest = json.loads((journal/'manifest.json').read_text(encoding='utf8'))
    original = (journal/'original.bin').read_bytes()
    stimulus = (journal/'stimulus.bin').read_bytes()
    if (manifest.get('schema') != 'flybrain.wind_transaction_prepare.v1'
            or manifest.get('target') != str(Path(path).resolve())
            or manifest.get('original_sha256') != digest(original)
            or manifest.get('stimulus_sha256') != digest(stimulus)
            or stimulus != isolated_wind_copy(original, direction_deg=manifest['direction_deg'],
                                               speed_kmh=manifest['speed_kmh'])):
        raise ValueError('Transaction identity or saved bytes differ')
    final = Path(finalization_dir)
    final.mkdir(exist_ok=False)
    with exclusive_existing_file(path) as fd:
        current = os.read(fd, MAX_CONFIG_BYTES + 1)
        if len(current) > MAX_CONFIG_BYTES:
            raise ValueError('Config exceeds bounded finalization budget')
        current_values = {k: v[2] for k, v in _wind_entries(current)[1].items()}
        original_values = {k: v[2] for k, v in _wind_entries(original)[1].items()}
        stimulus_values = {k: v[2] for k, v in _wind_entries(stimulus)[1].items()}
        if original_values == stimulus_values:
            raise ValueError('Original and stimulus wind are indistinguishable')
        if current_values == original_values:
            after = current
            action = 'already_original_no_write'
        elif current_values == stimulus_values:
            after = restore_wind_values(current, expected_applied=stimulus, original=original)
            action = 'restore_owned_wind'
        else:
            raise ValueError('Wind conflict or partial change; no write')
        _durable_new(final/'before.bin', current)
        _durable_new(final/'after.bin', after)
        intent = {'schema': 'flybrain.wind_restore_intent.v1',
                  'target': str(Path(path).resolve()), 'before_sha256': digest(current),
                  'after_sha256': digest(after), 'write_atomic': False,
                  'engine_state_verified': False, 'restart_outcome_attested': restart_outcome,
                  'action': action}
        _durable_new(final/'intent.json', json.dumps(intent, sort_keys=True).encode('utf8'))
        if action == 'restore_owned_wind':
            _write_target(fd, after)
            os.lseek(fd, 0, os.SEEK_SET)
            if os.read(fd, MAX_CONFIG_BYTES + 1) != after:
                raise OSError('Readback differs; saved journal requires inspection')
        receipt = {'status': 'completed_wind_finalization', 'action': action,
                   'after_sha256': digest(after),
                   'target_modified': action != 'already_original_no_write',
                   'write_atomic': False, 'engine_state_verified': False}
        _durable_new(final/'receipt.json', json.dumps(receipt, sort_keys=True).encode('utf8'))
    return receipt


def locked_restore(path, *, expected_applied, original, journal_dir):
    """Restore only owned wind values under one exclusive handle.

    Journal has the complete pre-write bytes and durable intent before mutation.
    An interrupted in-place write is not atomic: the saved pre-write bytes and intent
    must be inspected manually before recovery. No engine restart is performed.
    """
    journal = Path(journal_dir)
    journal.mkdir(exist_ok=False)
    with exclusive_existing_file(path) as fd:
        current = os.read(fd, MAX_CONFIG_BYTES + 1)
        if len(current) > MAX_CONFIG_BYTES:
            raise ValueError('Config exceeds bounded rollback budget')
        restored = restore_wind_values(current, expected_applied=expected_applied,
                                       original=original)
        _durable_new(journal/'before.bin', current)
        _durable_new(journal/'after.bin', restored)
        intent = {'schema': 'flybrain.wind_restore_intent.v1', 'before_sha256': digest(current),
                  'after_sha256': digest(restored), 'target': str(Path(path).resolve()),
                  'write_atomic': False, 'engine_state_verified': False}
        _durable_new(journal/'intent.json', json.dumps(intent, sort_keys=True).encode('utf8'))
        _write_target(fd, restored)
        os.lseek(fd, 0, os.SEEK_SET)
        if os.read(fd, MAX_CONFIG_BYTES + 1) != restored:
            raise OSError('Readback differs; original journal retained, outcome unknown')
        receipt = {'status': 'completed_locked_wind_restore', 'after_sha256': digest(restored),
                   'write_atomic': False, 'engine_state_verified': False}
        _durable_new(journal/'receipt.json', json.dumps(receipt, sort_keys=True).encode('utf8'))
        return receipt


def inspect_interrupted_restore(path, *, journal_dir):
    """Read-only assessment under lock; never infer ownership of partial current bytes."""
    journal = Path(journal_dir)
    intent = json.loads((journal/'intent.json').read_text(encoding='utf8'))
    before, after = (journal/'before.bin').read_bytes(), (journal/'after.bin').read_bytes()
    if (intent.get('schema') != 'flybrain.wind_restore_intent.v1'
            or intent.get('target') != str(Path(path).resolve())
            or intent.get('before_sha256') != digest(before)
            or intent.get('after_sha256') != digest(after)):
        raise ValueError('Journal identity or saved content differs')
    with exclusive_existing_file(path) as fd:
        current = os.read(fd, MAX_CONFIG_BYTES + 1)
    if len(current) > MAX_CONFIG_BYTES:
        raise ValueError('Config exceeds bounded inspection budget')
    if current == before:
        state = 'matches_prewrite'
    elif current == after:
        state = 'matches_intended_result'
    else:
        state = 'unknown_or_external_change'
    return {'status': 'completed_read_only_restore_inspection', 'current_state': state,
            'current_sha256': digest(current), 'receipt_present': (journal/'receipt.json').exists(),
            'target_modified': False, 'automatic_recovery_allowed': False}
