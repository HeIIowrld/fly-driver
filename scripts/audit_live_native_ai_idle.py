"""Read advancing same-copy native AI/body packets while CSP reports AI-off.

This is a diagnostic of reported states, not proof of exclusive control, exact
sample time, or engine memory atomicity. No lifecycle or actuator calls are made.
"""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_physics import NativeTeacherPhysicsReader
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--baseline-packet', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-session/status.json'
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'generation': args.generation, 'baseline_packet': args.baseline_packet,
        'scope': 'same-copy native flag and bounded CSP read; no actuation',
        'actor_input': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    rows, result = [], {'status': 'failed', 'actor_deployed': False,
                        'actuator_messages_sent': 0, 'lifecycle_requests_sent': 0}
    try:
        deadline = time.perf_counter() + 2.0
        with NativeTeacherPhysicsReader(timeout=.2) as reader:
            while time.perf_counter() < deadline and len(rows) < 300:
                before, modified = read_published_json(path)
                if (not 0 <= time.time()-modified <= 2
                        or before.get('generation') != args.generation
                        or before.get('paused') is not False
                        or before.get('ai_controlled') is not False
                        or before.get('camera_active') or before.get('expert_probe_active')
                        or before.get('is_online') or before.get('in_menu')):
                    raise ValueError('Fresh same-generation live CSP AI-off state required')
                coupled = reader.receive()
                fields = coupled.physics.fields
                if (coupled.physics.packet_id <= args.baseline_packet
                        or abs(fields['speedKmh']) >= .1 or fields['gas'] >= .02
                        or fields['brake'] <= .98):
                    raise ValueError('Advanced native packet and stationary full-brake required')
                after, modified = read_published_json(path)
                if (not 0 <= time.time()-modified <= 2
                        or after.get('generation') != args.generation
                        or after.get('paused') is not False
                        or after.get('ai_controlled') is not False):
                    raise ValueError('CSP AI-off state changed around native receipt')
                rows.append({'authority': coupled.authority_record(),
                             'csp_before_script_clock': before['script_clock'],
                             'csp_after_script_clock': after['script_clock'],
                             'speed_kmh': fields['speedKmh'], 'gas': fields['gas'],
                             'brake': fields['brake']})
        if len(rows) < 2:
            raise ValueError('At least two advancing packets required')
        write_json(run / 'rows.json', rows)
        result.update(status='completed_live_native_ai_idle_audit', samples=len(rows),
            first_packet=rows[0]['authority']['packet_id'],
            last_packet=rows[-1]['authority']['packet_id'],
            native_true=sum(row['authority']['builtin_ai_controlled'] for row in rows),
            native_false=sum(not row['authority']['builtin_ai_controlled'] for row in rows),
            rows_sha256=sha256(run / 'rows.json'), exact_physical_time_verified=False,
            engine_atomicity_verified=False, exclusive_control_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
