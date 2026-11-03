"""Exercise wind-only rollback on copies of a race config; never modify the source."""

import argparse
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.wind_stimulus_config import isolated_wind_copy, restore_wind_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.source)
    original = source.read_bytes()
    source_hash = sha256(source)
    applied = isolated_wind_copy(original, direction_deg=90, speed_kmh=36)
    # Non-WIND addition represents a change by another owner and must survive.
    external = b'\r\n[FLYBRAIN_DRY_RUN_ONLY]\r\nEXTERNAL_CHANGE=preserve\r\n'
    current = applied + external
    restored = restore_wind_values(current, expected_applied=applied, original=original)
    if restored != original + external:
        raise ValueError('Wind-only rollback changed unrelated bytes')
    conflict = isolated_wind_copy(original, direction_deg=180, speed_kmh=36)
    try:
        restore_wind_values(conflict, expected_applied=applied, original=original)
    except ValueError:
        conflict_rejected = True
    else:
        raise ValueError('Conflicting wind edit must never be overwritten')
    if sha256(source) != source_hash:
        raise ValueError('Original changed during offline diagnostic')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'source_sha256': source_hash, 'scope': __doc__, 'simulation_only': True})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'completed_wind_rollback_dry_run', 'source_sha256': source_hash,
        'other_owner_edit_preserved': True, 'different_wind_edit_rejected': conflict_rejected,
        'source_changed': False, 'file_lock_implemented': False,
        'engine_restart_or_restore_tested': False, 'actor_deployed': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
