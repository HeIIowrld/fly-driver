"""Read-only local installation discovery; never starts or controls the game."""

import os
import re
from pathlib import Path


def steam_roots():
    roots = []
    if os.name == "nt":
        import winreg

        for hive, key, value in (
            (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
        ):
            try:
                with winreg.OpenKey(hive, key) as handle:
                    roots.append(Path(winreg.QueryValueEx(handle, value)[0]))
            except OSError:
                pass
        if os.environ.get("ProgramFiles(x86)"):
            roots.append(Path(os.environ["ProgramFiles(x86)"]) / "Steam")
    else:
        roots.extend([Path.home() / ".steam/steam", Path.home() / ".local/share/Steam"])
    return list(dict.fromkeys(roots))


def inspect_installation(roots=None, game_paths=()):
    game_paths = tuple(game_paths)
    roots = steam_roots() if roots is None else [Path(p) for p in roots]
    libraries, candidates, errors = set(roots), set(map(Path, game_paths)), []
    manifests = []
    for root in roots:
        folders = root / "steamapps/libraryfolders.vdf"
        try:
            text = folders.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as error:
            errors.append({"path": str(folders), "error": type(error).__name__})
            continue
        libraries.update(Path(p.replace("\\\\", "\\")) for p in
                         re.findall(r'"path"\s+"([^"]+)"', text))
    for library in sorted(libraries, key=str):
        manifest = library / "steamapps/appmanifest_244210.acf"
        try:
            text = manifest.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as error:
            errors.append({"path": str(manifest), "error": type(error).__name__})
            continue
        manifests.append(str(manifest))
        directory = re.search(r'"installdir"\s+"([^"]+)"', text)
        if directory:
            name = directory.group(1)
            # A manifest is data, not permission to inspect arbitrary traversal paths.
            if Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
                errors.append({"path": str(manifest), "error": "invalid_installdir"})
                continue
            candidates.add(library / "steamapps/common" / name)
    installations = []
    for candidate in sorted(candidates, key=str):
        try:
            engines = [n for n in ("acs.exe", "acs_x86.exe") if (candidate / n).is_file()]
            installations.append({
                "path": str(candidate), "engine_files": engines,
                "engine_present": bool(engines),
                "plugin_directory_present": (candidate / "apps/python").is_dir(),
                "content_directory_present": (candidate / "content").is_dir(),
            })
        except OSError as error:
            errors.append({"path": str(candidate), "error": type(error).__name__})
    found = any(i["engine_present"] for i in installations)
    return {
        "probe": "read_only_installation_discovery",
        "status": "engine_files_found" if found else "no_engine_in_inspected_locations",
        "searched_steam_roots": list(map(str, roots)),
        "searched_libraries": sorted(map(str, libraries)),
        "explicit_game_paths": list(map(str, game_paths)),
        "manifests": manifests, "candidates": installations, "errors": errors,
        "coverage": "registered/default Steam libraries and explicitly supplied paths only",
        "game_launched": False, "actions_sent": False,
        "runtime_capabilities": {
            key: "not_probed" for key in (
                "image_capture", "synchronized_wide_eye_views", "telemetry_fields",
                "action_response", "reset", "capture_and_control_latency",
            )
        },
        "next_action": "runtime bridge probe" if found else "provide installed AC game location",
    }
