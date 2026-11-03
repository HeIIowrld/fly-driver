"""Compare old and current AC controls as INI mappings; no device or game mutation."""

import argparse
import configparser
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def parse(path):
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(path.read_text(encoding="utf8"))
    if parser.defaults():
        raise ValueError("Unexpected inherited INI defaults")
    return {section: dict(parser.items(section, raw=True)) for section in parser.sections()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "current", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    previous, current = Path(args.previous), Path(args.current)
    old_sha, new_sha = sha256(previous), sha256(current)
    old, new = parse(previous), parse(current)
    if sha256(previous) != old_sha or sha256(current) != new_sha:
        raise ValueError("Input profile changed while reading")
    keys = sorted((section, key) for section in old.keys() | new.keys()
                  for key in old.get(section, {}).keys() | new.get(section, {}).keys())
    differences = [{"section": section, "key": key,
                    "previous": old.get(section, {}).get(key),
                    "current": new.get(section, {}).get(key)}
                   for section, key in keys
                   if old.get(section, {}).get(key) != new.get(section, {}).get(key)]
    missing_sections = sorted(old.keys() ^ new.keys())
    run = begin_run(args.output, {"program_sha256": sha256(__file__),
        "previous_sha256": old_sha, "current_sha256": new_sha, "read_only": True})
    (run / "executed-program.py").write_bytes(Path(__file__).read_bytes())
    result = {"status": "completed_ini_mapping_comparison", "previous_sha256": old_sha,
        "current_sha256": new_sha, "previous_sections": len(old), "current_sections": len(new),
        "section_symmetric_difference": missing_sections, "field_differences": differences,
        "ini_fields_equal": not differences and not missing_sections,
        "runtime_binding_or_axis_response_verified": False, "collector_qualified": False,
        "controls_modified": False, "actor_deployed": False}
    write_json(run / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
