"""Prepare isolated race.ini wind stimuli without touching simulator configuration."""

import math
import re

WIND_KEYS = frozenset((b'DIRECTION_DEG', b'SPEED_KMH_MIN', b'SPEED_KMH_MAX'))


def _wind_entries(source):
    if not isinstance(source, bytes) or not source:
        raise ValueError('Original INI bytes required')
    lines = source.splitlines(keepends=True)
    sections = []
    for index, line in enumerate(lines):
        match = re.fullmatch(rb'\s*\[([^\]\r\n]+)\]\s*', line.rstrip(b'\r\n'))
        if match:
            sections.append((index, match.group(1).strip().upper()))
    wind = [index for index, name in sections if name == b'WIND']
    if len(wind) != 1:
        raise ValueError('Exactly one WIND section required')
    end = next((index for index, _ in sections if index > wind[0]), len(lines))
    entries = {}
    pattern = rb'([ \t]*[A-Za-z_]+[ \t]*=[ \t]*)([^;#\r\n]*)(.*)'
    for index in range(wind[0]+1, end):
        match = re.fullmatch(pattern, lines[index], flags=re.DOTALL)
        if not match:
            continue
        key = match.group(1).split(b'=', 1)[0].strip().upper()
        if key not in WIND_KEYS:
            continue
        if key in entries:
            raise ValueError('Duplicate WIND key')
        value = match.group(2).strip()
        if not re.fullmatch(rb'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value):
            raise ValueError('Finite numeric WIND value required')
        if not math.isfinite(float(value)):
            raise ValueError('Finite numeric WIND value required')
        entries[key] = (index, match, value)
    if set(entries) != WIND_KEYS:
        raise ValueError('All three WIND keys required')
    return lines, entries


def restore_wind_values(current, *, expected_applied, original):
    """Prepare wind-only rollback if current wind still equals the owned stimulus.

    Returns bytes only. Caller must independently protect against concurrent file writes;
    this function performs neither file locking nor simulator changes.
    """
    lines, active = _wind_entries(current)
    _, expected = _wind_entries(expected_applied)
    _, saved = _wind_entries(original)
    if any(active[key][2] != expected[key][2] for key in WIND_KEYS):
        raise ValueError('Current wind differs from owned stimulus; no rollback prepared')
    for key in WIND_KEYS:
        index, match, _ = active[key]
        value = saved[key][2]
        lines[index] = match.group(1)+value+match.group(3)
    return b''.join(lines)


def wind_restart_merge(*, direction_deg, speed_kmh):
    """Create a strictly bounded wind-only candidate for CSP's optional INI merge.

    This is preparation only: no engine invocation, status admission or restoration.
    """
    source = b'[WIND]\nDIRECTION_DEG=0\nSPEED_KMH_MIN=0\nSPEED_KMH_MAX=0\n'
    result = isolated_wind_copy(source, direction_deg=direction_deg, speed_kmh=speed_kmh)
    return result.decode('ascii')


def isolated_wind_copy(source, *, direction_deg, speed_kmh):
    """Modify exactly three existing WIND value lines; preserve all other bytes."""
    if (not isinstance(source, bytes) or not source
            or not isinstance(direction_deg, (int, float)) or isinstance(direction_deg, bool)
            or not isinstance(speed_kmh, (int, float)) or isinstance(speed_kmh, bool)
            or not math.isfinite(direction_deg) or not 0 <= direction_deg < 360
            or not math.isfinite(speed_kmh) or not 0 <= speed_kmh <= 36):
        raise ValueError('Finite bounded wind stimulus and source bytes required')
    lines = source.splitlines(keepends=True)
    sections = []
    for i, line in enumerate(lines):
        match = re.fullmatch(rb'\s*\[([^\]\r\n]+)\]\s*', line.rstrip(b'\r\n'))
        if match:
            sections.append((i, match.group(1).strip().upper()))
    selected = [i for i, name in sections if name == b'WIND']
    if len(selected) != 1:
        raise ValueError('Exactly one original WIND section required')
    start = selected[0]
    end = next((i for i, _ in sections if i > start), len(lines))
    replacements = {b'DIRECTION_DEG': format(direction_deg, '.8g').encode('ascii'),
                    b'SPEED_KMH_MIN': format(speed_kmh, '.8g').encode('ascii'),
                    b'SPEED_KMH_MAX': format(speed_kmh, '.8g').encode('ascii')}
    seen = set()
    for i in range(start+1, end):
        match = re.fullmatch(rb'([ \t]*)([A-Za-z_]+)([ \t]*=[ \t]*)([^;#\r\n]*)(.*)',
                             lines[i], flags=re.DOTALL)
        if not match:
            continue
        key = match.group(2).upper()
        if key not in replacements:
            continue
        if key in seen:
            raise ValueError('Duplicate wind parameter cannot be changed safely')
        seen.add(key)
        lines[i] = (match.group(1)+match.group(2)+match.group(3)+replacements[key]
                    + match.group(5))
    if seen != set(replacements):
        raise ValueError('All three original WIND values must exist')
    return b''.join(lines)
