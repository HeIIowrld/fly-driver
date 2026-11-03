"""Compare synthetic sampler gaps with an artificial GIL-holding Python workload.

This does not connect to AC and cannot identify the cause of an earlier stall.
"""
import argparse
import json
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def process_sampler(start, ready, duration, period, result):
    stamps = []
    ready.set()
    start.wait(5)
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        stamps.append(time.perf_counter())
        time.sleep(period)
    result.put(stamps)


def run_case(switch_interval, duration, period, hog_seconds):
    stop = threading.Event()
    ready = threading.Event()
    stamps = []
    def sampler():
        ready.set()
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(period)
    ctx = mp.get_context('spawn')
    start = ctx.Event()
    child_ready = ctx.Event()
    results = ctx.Queue()
    child = ctx.Process(target=process_sampler, args=(start, child_ready, duration, period, results))
    child.start()
    thread = threading.Thread(target=sampler, daemon=True)
    old = sys.getswitchinterval()
    try:
        sys.setswitchinterval(switch_interval)
        thread.start()
        if not ready.wait(3) or not child_ready.wait(15):
            raise RuntimeError('samplers did not start')
        start.set()
        time.sleep(.2)
        hog_start = time.perf_counter()
        deadline = hog_start + hog_seconds
        n = 0
        while time.perf_counter() < deadline:
            n += 1
        hog_end = time.perf_counter()
        independent = results.get(timeout=5)
    finally:
        sys.setswitchinterval(old)
        stop.set()
        thread.join(3)
        child.join(5)
    if child.is_alive() or child.exitcode != 0 or thread.is_alive():
        raise RuntimeError('sampler failed to exit')
    def inspect(values):
        a = np.asarray(values, dtype=np.float64)
        if len(a) < 3 or not np.all(np.diff(a) > 0):
            raise ValueError('insufficient or unordered samples')
        gaps = np.diff(a)
        overlapping = gaps[(a[:-1] < hog_end) & (a[1:] > hog_start)]
        if not len(overlapping):
            raise ValueError('no samples bracket the workload')
        return {'samples': len(a), 'max_gap_ms': float(gaps.max()*1000),
                'max_hog_overlapping_gap_ms': float(overlapping.max()*1000)}
    return {'switch_interval_seconds': switch_interval,
            'hog_duration_ms': (hog_end-hog_start)*1000, 'iterations': n,
            'same_process_thread': inspect(stamps), 'separate_process': inspect(independent)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    if sha256(__file__) != plan['program_sha256']:
        raise ValueError('Plan program hash mismatch')
    run = begin_run(args.output, dict(plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), actor=False, optimizer_updates=0))
    result = {'status':'failed', 'vehicle_control_sent':False, 'actor_deployed':False}
    try:
        cases = [run_case(interval, plan['duration'], plan['period'], plan['hog_seconds'])
                 for interval in plan['switch_intervals']]
        result.update(status='completed', cases=cases,
            scope='Artificial single-process GIL workload; no simulator, sensor, native copy, GPU, or claim about prior live-stall cause.')
    except Exception as exc:
        result.update(error=type(exc).__name__, reason=str(exc))
    write_json(run/'result.json', result)
    print(json.dumps(result, allow_nan=False))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    mp.freeze_support()
    main()
