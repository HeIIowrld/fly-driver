"""Replay preserved native bytes through a bounded process queue; no simulator I/O."""
import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def producer(records, queue, period):
    try:
        for seq, record in enumerate(records):
            raw = base64.b64decode(record['prefix_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != record['prefix_sha256']:
                raise ValueError('recorded bytes differ')
            stamp = time.perf_counter()
            decoded = decode_teacher_physics(raw, stamp)
            if decoded.physics.packet_id != record['packet_id']:
                raise ValueError('recorded packet differs')
            queue.put(('sample', seq, stamp, raw), timeout=2)
            time.sleep(period)
        queue.put(('done', len(records)), timeout=2)
    except Exception as exc:
        queue.put(('error', type(exc).__name__, str(exc)), timeout=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    source = Path(plan['source'])
    if sha256(__file__) != plan['program_sha256'] or sha256(source) != plan['source_sha256']:
        raise ValueError('Plan or source mismatch')
    run = begin_run(args.output, dict(plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), source_sha256=sha256(source), actor=False))
    result = {'status':'failed','vehicle_control_sent':False,'actor_deployed':False}
    try:
        records = json.loads(source.read_text(encoding='utf8'))[:plan['samples']]
        records = [row['authority'] for row in records]
        if len(records) != plan['samples']:
            raise ValueError('insufficient preserved records')
        ctx = mp.get_context('spawn')
        channel = ctx.Queue(maxsize=plan['queue_capacity'])
        child = ctx.Process(target=producer, args=(records, channel, plan['period']))
        child.start()
        rows = []
        stall_start = stall_end = None
        while True:
            item = channel.get(timeout=10)
            if item[0] == 'done':
                if item[1] != len(records):
                    raise ValueError('producer count mismatch')
                break
            if item[0] == 'error':
                raise RuntimeError(f'producer {item[1]}: {item[2]}')
            tag, seq, stamp, raw = item
            received = time.perf_counter()
            if tag != 'sample' or seq != len(rows) or received < stamp:
                raise ValueError('missing, duplicate, or future sample')
            expected = records[seq]
            if hashlib.sha256(raw).hexdigest() != expected['prefix_sha256']:
                raise ValueError('IPC payload corrupted')
            decoded = decode_teacher_physics(raw, stamp)
            if decoded.physics.packet_id != expected['packet_id']:
                raise ValueError('IPC packet identity mismatch')
            age = received-stamp
            rows.append({'sequence':seq, 'packet_id':decoded.physics.packet_id,
                'produced':stamp, 'received':received, 'age_ms':age*1000,
                'accepted_fresh':age <= plan['max_age']})
            if len(rows) == plan['stall_after']:
                stall_start = time.perf_counter()
                deadline = stall_start+plan['stall_duration']
                while time.perf_counter() < deadline:
                    pass
                stall_end = time.perf_counter()
        child.join(5)
        if child.is_alive() or child.exitcode != 0:
            raise RuntimeError('producer did not terminate cleanly')
        if len(rows) != len(records):
            raise ValueError('incomplete queue delivery')
        ages = np.array([row['age_ms'] for row in rows])
        stamps = np.array([row['produced'] for row in rows])
        if not np.all(np.diff(stamps) > 0):
            raise ValueError('producer receipt times not ordered')
        write_json(run/'rows.json', rows)
        result.update(status='completed', delivered=len(rows),
            stale_rejected=sum(not row['accepted_fresh'] for row in rows),
            max_delivery_age_ms=float(ages.max()), p95_delivery_age_ms=float(np.percentile(ages,95)),
            producer_max_gap_ms=float(np.diff(stamps).max()*1000),
            stall_ms=(stall_end-stall_start)*1000, child_exitcode=child.exitcode,
            rows_sha256=sha256(run/'rows.json'),
            scope='Offline replay and synthetic receipt times; no live mapping, no physical timing, no actuator, no production integration.')
    except Exception as exc:
        result.update(error=type(exc).__name__,reason=str(exc))
    write_json(run/'result.json',result)
    print(json.dumps(result,allow_nan=False))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    mp.freeze_support()
    main()
