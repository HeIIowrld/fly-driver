"""Bounded read-only native mapping probe; no camera, actuator or simulator control."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_process_sampler import NativeProcessSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    if (sha256(__file__) != plan['program_sha256'] or
            sha256('src/flybrain_drive/native_process_sampler.py') != plan['sampler_sha256']):
        raise ValueError('declared program differs')
    run = begin_run(args.output, dict(plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), actor=False, vehicle_control_sent=False))
    result = {'status':'failed','actor_deployed':False,'vehicle_control_sent':False}
    sampler = NativeProcessSampler(period=.005, max_samples=plan['samples'], capacity=128)
    try:
        sampler.start()
        rows = []
        deadline = time.perf_counter()+15
        while len(rows) < plan['samples']:
            if time.perf_counter() > deadline:
                raise TimeoutError('bounded read-only collection expired')
            row = sampler.poll(timeout=.1)
            if row is not None:
                rows.append(row)
        sampler.stop()
        receipts = np.array([r['authority']['received_monotonic_seconds'] for r in rows])
        packets = [r['authority']['packet_id'] for r in rows]
        if not np.all(np.diff(receipts) > 0) or any(a > b for a,b in zip(packets,packets[1:])):
            raise ValueError('receipt ordering or packet reset')
        write_json(run/'rows.json', rows)
        result.update(status='completed',samples=len(rows),terminal=list(sampler.terminal),
            process_exitcode=sampler.process.exitcode, distinct_packets=len(set(packets)),
            max_host_receipt_gap_ms=float(np.diff(receipts).max()*1000),
            rows_sha256=sha256(run/'rows.json'),
            scope='Read-only native same-copy host receipts. No camera, actuation, physical sample time or actor qualification.')
    except Exception as exc:
        result.update(error=type(exc).__name__,reason=str(exc))
        if sampler.started and sampler.process.is_alive():
            try:
                sampler.stop()
            except Exception as stop_exc:
                result['cleanup_error']=type(stop_exc).__name__+': '+str(stop_exc)
    write_json(run/'result.json',result)
    print(json.dumps(result,allow_nan=False))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
