# Adapted from the reviewed MECar-Simulation operational utility (2026-10-05).
"""Read SSH settings without copying credentials into files, arguments, or logs."""

from __future__ import annotations

import getpass
import os
import re
import sys


def _nonempty(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("SSH settings must be nonempty strings")
    return value


def read_setting(name: str, default: str | None = None) -> str | None:
    """Prefer process settings, then Windows user settings, then the caller default.

    Reading HKCU also supports a running application whose process environment
    predates an explicitly saved Windows user setting. Values are returned
    literally, including REG_EXPAND_SZ values, so passwords are never expanded.
    """
    if not isinstance(name, str) or re.fullmatch(r"MECAR_SSH_[A-Z0-9_]+", name) is None:
        raise ValueError("Only MECAR_SSH_ setting names are supported")
    if name in os.environ:
        return _nonempty(os.environ[name])
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, kind = winreg.QueryValueEx(key, name)
        except FileNotFoundError:
            return default
        if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
            raise ValueError("Windows SSH settings must contain text values")
        return _nonempty(value)
    return default


def ssh_password() -> str:
    """Read the configured secret, or request one interactively when absent."""
    value = read_setting("MECAR_SSH_PASSWORD")
    if value is None:
        value = getpass.getpass("SSH password: ")
    return _nonempty(value)
