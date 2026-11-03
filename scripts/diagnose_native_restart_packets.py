"""Bounded read-only diagnostic; repeated IDs are observations, not fresh physics."""

import argparse
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_physics import repeated_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
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
        'baseline_packet': args.baseline_packet, 'generation': args.generation,
        'duration_seconds': 3, 'actor_input': False})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    result = {'status': 'failed', 'actuator_messages_sent': 0,
              'lifecycle_requests_sent': 0, 'actor_deployed': False}
    try:
        mapping = ExistingNamedMapping('Local\\acpmf_physics', AI_STATUS_PREFIX.size)
        try:
            deadline = time.perf_counter() + 3
            while time.perf_counter() < deadline:
                state, modified = read_published_json(path)
                age = time.time() - modified
                sample = repeated_teacher_snapshot(mapping)
                fields = sample.physics.fields
                checks = {'fresh_csp': 0 <= age <= 2,
                          'same_generation': state.get('generation') == args.generation,
                          'live_unpaused': (state.get('paused') is False
                                            and not state.get('in_menu')),
                          'csp_ai_off': state.get('ai_controlled') is False,
                          'camera_off': not state.get('camera_active'),
                          'probe_off': not state.get('expert_probe_active'),
                          'offline': not state.get('is_online'),
                          'packet_after_baseline': sample.physics.packet_id > args.baseline_packet,
                          'stationary': abs(fields['speedKmh']) < .1,
                          'gas_low': fields['gas'] < .02,
                          'full_brake': fields['brake'] > .98}
                rows.append({'authority': sample.authority_record(), 'checks': checks,
                             'csp_age_seconds': age, 'csp_script_clock': state.get('script_clock'),
                             'speed_kmh': fields['speedKmh'], 'gas': fields['gas'],
                             'brake': fields['brake']})
                if not all(checks[k] for k in ('fresh_csp', 'same_generation', 'live_unpaused',
                       'csp_ai_off', 'camera_off', 'probe_off', 'offline', 'stationary',
                       'gas_low', 'full_brake')):
                    raise ValueError('Safety or CSP state condition failed; inspect saved checks')
                time.sleep(.05)
        finally:
            mapping.close()
        ids = [row['authority']['packet_id'] for row in rows]
        result.update(status='completed_diagnostic', samples=len(rows),
                      unique_packet_ids=len(set(ids)), min_packet=min(ids), max_packet=max(ids),
                      after_baseline=sum(row['checks']['packet_after_baseline'] for row in rows),
                      native_true=sum(row['authority']['builtin_ai_controlled'] for row in rows),
                      source_time_verified=False, engine_atomicity_verified=False,
                      exclusive_control_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run / 'rows.json', rows)
        result.update(rows_sha256=sha256(run / 'rows.json'), samples_saved=len(rows))
        write_json(run / 'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
