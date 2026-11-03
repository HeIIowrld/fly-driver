"""Exercise experimental sampler with preserved bytes; no live mapping or control."""
import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_process_sampler import NativeProcessSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    source = Path(plan['source'])
    if (sha256(__file__) != plan['program_sha256'] or sha256(source) != plan['source_sha256']
            or sha256('src/flybrain_drive/native_process_sampler.py') != plan['module_sha256']):
        raise ValueError('source, program, or module mismatch')
    run = begin_run(args.output, dict(plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), source_sha256=sha256(source), actor=False))
    result = {'status':'failed', 'actor_deployed':False, 'vehicle_control_sent':False}
    sampler = None
    try:
        raw_rows = json.loads(source.read_text(encoding='utf8'))[:plan['samples']]
        recorded = tuple(base64.b64decode(row['authority']['prefix_base64'], validate=True)
                         for row in raw_rows)
        if len(recorded) != plan['samples'] or any(hashlib.sha256(raw).hexdigest()
                != row['authority']['prefix_sha256'] for raw, row in zip(recorded, raw_rows)):
            raise ValueError('recorded payload provenance')
        sampler = NativeProcessSampler(recorded=recorded, period=plan['period'],
            max_samples=len(recorded), capacity=plan['capacity'])
        sampler.start()
        received = []
        deadline = time.perf_counter()+15
        while len(received) < len(recorded):
            if time.perf_counter() > deadline:
                raise TimeoutError('incomplete recorded replay')
            row = sampler.poll(timeout=.1)
            if row is not None:
                expected = raw_rows[len(received)]['authority']
                if row['authority']['prefix_sha256'] != expected['prefix_sha256']:
                    raise ValueError('payload order or identity mismatch')
                received.append(row['authority']['packet_id'])
        sampler.stop()
        result.update(status='completed', delivered=len(received),
            process_exitcode=sampler.process.exitcode, terminal=list(sampler.terminal),
            scope='Recorded prefixes, synthetic receipt times, no live mapping or actor.')
    except Exception as exc:
        result.update(error=type(exc).__name__,reason=str(exc))
        if sampler is not None and sampler.started and sampler.process.is_alive():
            try:
                sampler.stop()
            except Exception as stop_exc:
                result['cleanup_error'] = type(stop_exc).__name__+': '+str(stop_exc)
    write_json(run/'result.json', result)
    print(json.dumps(result, allow_nan=False))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
