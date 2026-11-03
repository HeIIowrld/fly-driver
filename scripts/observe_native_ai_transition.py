"""Read-only native/CSP controller-flag history; repeated packets are never fresh steps."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_status import NativeTeacherStatusReader
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-seconds', type=int, default=60)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.duration_seconds <= 90:
        raise ValueError('Bounded read-only observation required')
    path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
    initial, _ = read_published_json(path)
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        generation=initial['generation'], actor_input=False,
        scope='Observe existing owned collection, never change controls, pause or lifecycle'))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reader = NativeTeacherStatusReader()
    rows, previous = [], None
    started = time.monotonic()
    try:
        while time.monotonic()-started < args.duration_seconds:
            row = dict(read_started_at=time.monotonic())
            try:
                status, modified = read_published_json(path)
                if status['generation'] != initial['generation']:
                    raise ValueError('Original observed generation changed')
                native = reader.snapshot()
                row.update(native=native, csp={k: status.get(k) for k in (
                    'generation', 'script_clock', 'paused', 'ai_controlled',
                    'expert_probe_active', 'camera_active', 'speed_kmh')},
                    csp_publication_age_seconds=time.time()-modified,
                    native_packet_advanced=(previous is not None
                                            and native['packet_id'] > previous),
                    repeated_packet_is_not_new_physics=True)
                if previous is not None and native['packet_id'] < previous:
                    raise ValueError('Native sequence reset during same-generation observation')
                previous = native['packet_id']
            except Exception as error:
                row.update(error=type(error).__name__, reason=str(error))
            rows.append(row)
            time.sleep(.02)
    finally:
        reader.close()
        output = run/'history.jsonl'
        output.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows),
                          encoding='utf8')
        write_json(run/'result.json', dict(status='completed_read_only_ai_flag_observation',
            rows=len(rows), errors=sum('error' in r for r in rows),
            history_sha256=sha256(output), ui_actions_sent=0, actuator_messages_sent=0,
            control_ownership_qualified=False, actor_input=False, goal_achieved=False))


if __name__ == '__main__':
    main()
