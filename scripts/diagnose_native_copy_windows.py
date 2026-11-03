"""Measure read-copy windows during a bounded stationary live session; no control."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_published_state import read_published_json
from flybrain_drive.ac_teacher_physics import timed_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
    refuse_leased_transport('native copy window diagnostic', status_path=path)
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'generation': args.generation,
        'duration_seconds': 3, 'nominal_sleep_seconds': .002, 'actor_input': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    result = {'status': 'failed', 'ui_actions_sent': 0, 'actuator_messages_sent': 0,
              'lifecycle_requests_sent': 0, 'actor_deployed': False}
    try:
        mapping = ExistingNamedMapping('Local\\acpmf_physics', AI_STATUS_PREFIX.size)
        try:
            deadline, next_status = time.perf_counter()+3, 0.
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                if now >= next_status:
                    state, modified = read_published_json(path)
                    age = time.time()-modified
                    if (not 0 <= age <= 2 or state.get('generation') != args.generation
                            or state.get('paused') is not False or state.get('in_menu') is not False
                            or state.get('is_online') is not False
                            or state.get('ai_controlled') is not False
                            or state.get('camera_active') is not False
                            or state.get('expert_probe_active') is not False
                            or state.get('pending_receipt_id') or state.get('error')):
                        raise ValueError('Live same-generation idle CSP state required')
                    next_status = now+.1
                sample, timing = timed_teacher_snapshot(mapping)
                fields = sample.physics.fields
                rows.append({'authority': sample.authority_record(), 'copy': timing,
                             'speed_kmh': fields['speedKmh'], 'gas': fields['gas'],
                             'brake': fields['brake']})
                if (sample.builtin_ai_controlled or abs(fields['speedKmh']) >= .1
                        or fields['gas'] >= .02 or fields['brake'] <= .98):
                    raise ValueError('Native AI off, stationary and full brake required')
                time.sleep(.002)
        finally:
            mapping.close()
        if len(rows) < 2:
            raise ValueError('At least two original observations required')
        intervals, same_id_changed = [], 0
        for prior, current in zip(rows, rows[1:], strict=False):
            a, b = prior['authority'], current['authority']
            if b['packet_id'] < a['packet_id']:
                raise ValueError('Packet reset; explicit new episode required')
            if b['packet_id'] == a['packet_id']:
                same_id_changed += a['prefix_sha256'] != b['prefix_sha256']
            else:
                intervals.append(1000*(current['copy']['second_copy_end']
                                       - prior['copy']['first_copy_start']))
        if not intervals:
            raise ValueError('No observed packet transition')
        result.update(status='completed_native_copy_window_diagnostic', samples=len(rows),
            observed_packet_transitions=len(intervals), same_id_changed_bytes=same_id_changed,
            copy_window_ms_p50_p95_max=np.percentile([
                1000*(r['copy']['second_copy_end']-r['copy']['first_copy_start']) for r in rows],
                [50, 95, 100]).tolist(),
            conditional_transition_window_ms_p50_p95_max=np.percentile(intervals,
                [50, 95, 100]).tolist(),
            conditional_assumption='coherent monotonic mapping publication, not engine seqlock',
            sample_generation_time_verified=False, physical_time_verified=False,
            camera_exposure_time_verified=False, training_admissible=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'rows.json', rows)
        result.update(rows_sha256=sha256(run/'rows.json'), samples_saved=len(rows))
        write_json(run/'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
