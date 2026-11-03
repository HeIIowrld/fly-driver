import pytest

from scripts.prepare_teacher_reproduction_settings import edit_ini


def test_ini_settings_are_scoped_and_preserve_other_sections_and_line_endings():
    original = b'[OTHER]\r\nFF_GAIN=7\r\n[STEER]\r\nFF_GAIN=0\r\nLOCK=900\r\n'
    updated = edit_ini(original, {('STEER', 'FF_GAIN'): 1})
    assert updated == original.replace(b'FF_GAIN=0', b'FF_GAIN=1')


def test_missing_or_duplicate_keys_are_rejected_before_any_write():
    with pytest.raises(ValueError, match='missing'):
        edit_ini(b'[OTHER]\nFF_GAIN=0\n', {('STEER', 'FF_GAIN'): 1})
    with pytest.raises(ValueError, match='Duplicate'):
        edit_ini(b'[STEER]\nFF_GAIN=0\nFF_GAIN=1\n', {('STEER', 'FF_GAIN'): 1})
