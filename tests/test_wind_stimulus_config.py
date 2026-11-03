import pytest

from flybrain_drive.wind_stimulus_config import (
    isolated_wind_copy,
    restore_wind_values,
    wind_restart_merge,
)


def test_only_existing_wind_values_change_and_comments_survive():
    raw = (b'[RACE]\r\nVALUE=unchanged\r\n[WIND]\r\nDIRECTION_DEG=-1 ; note\r\n'
           b'SPEED_KMH_MIN=0\r\nSPEED_KMH_MAX=0\r\nOTHER=x\r\n[NEXT]\r\nZ=1\r\n')
    result = isolated_wind_copy(raw, direction_deg=90, speed_kmh=36)
    assert result == (b'[RACE]\r\nVALUE=unchanged\r\n[WIND]\r\nDIRECTION_DEG=90; note\r\n'
                      b'SPEED_KMH_MIN=36\r\nSPEED_KMH_MAX=36\r\nOTHER=x\r\n[NEXT]\r\nZ=1\r\n')


def test_restart_merge_cannot_include_unrelated_race_settings():
    assert wind_restart_merge(direction_deg=90, speed_kmh=36) == (
        '[WIND]\nDIRECTION_DEG=90\nSPEED_KMH_MIN=36\nSPEED_KMH_MAX=36\n')
    with pytest.raises(ValueError):
        wind_restart_merge(direction_deg=0, speed_kmh=float('inf'))


def test_wind_rollback_preserves_unrelated_concurrent_edits():
    original = (b'[WIND]\r\nDIRECTION_DEG=-1\r\nSPEED_KMH_MIN=0\r\n'
                b'SPEED_KMH_MAX=0\r\n[RACE]\r\nCARS=1\r\n')
    applied = isolated_wind_copy(original, direction_deg=90, speed_kmh=36)
    concurrent = applied.replace(b'CARS=1', b'CARS=4') + b'EXTRA=ok\r\n'
    restored = restore_wind_values(concurrent, expected_applied=applied, original=original)
    assert restored == original.replace(b'CARS=1', b'CARS=4') + b'EXTRA=ok\r\n'


def test_wind_rollback_refuses_changed_wind_or_ambiguous_config():
    original = b'[WIND]\nDIRECTION_DEG=-1\nSPEED_KMH_MIN=0\nSPEED_KMH_MAX=0\n'
    applied = isolated_wind_copy(original, direction_deg=90, speed_kmh=36)
    for current in (applied.replace(b'SPEED_KMH_MIN=36', b'SPEED_KMH_MIN=20'),
                    applied + b'DIRECTION_DEG=90\n', applied.replace(b'36', b'nan')):
        with pytest.raises(ValueError):
            restore_wind_values(current, expected_applied=applied, original=original)


@pytest.mark.parametrize('source', [
    b'[RACE]\nA=1\n',
    b'[WIND]\nDIRECTION_DEG=0\nSPEED_KMH_MIN=0\n',
    b'[WIND]\nDIRECTION_DEG=0\nDIRECTION_DEG=1\nSPEED_KMH_MIN=0\nSPEED_KMH_MAX=0\n',
    b'[WIND]\nDIRECTION_DEG=0\nSPEED_KMH_MIN=0\nSPEED_KMH_MAX=0\n[WIND]\n',
])
def test_reject_missing_or_ambiguous_source(source):
    with pytest.raises(ValueError):
        isolated_wind_copy(source, direction_deg=0, speed_kmh=10)


@pytest.mark.parametrize('direction,speed', [(-1, 1), (360, 1), (0, -1), (0, 37),
                                              (float('nan'), 1), (True, 1)])
def test_reject_unbounded_or_invalid_stimulus(direction, speed):
    with pytest.raises(ValueError):
        isolated_wind_copy(b'[WIND]\n', direction_deg=direction, speed_kmh=speed)
