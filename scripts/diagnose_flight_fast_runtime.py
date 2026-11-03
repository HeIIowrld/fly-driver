"""Observe GC/CPU overlap on one declared full replay; never a live-control certificate."""

import argparse
import gc
import json
import runpy
import sys
import threading
import time
from pathlib import Path

from flybrain_drive.flight_fast_stream import FastFlightStream
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-audit', required=True)
    parser.add_argument('--mode', choices=('full', 'zero_yaw'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    events, calls = [], []
    originally_enabled = gc.isenabled()
    original = FastFlightStream.advance

    def gc_event(phase, info):
        events.append(dict(phase=phase, clock=time.perf_counter(),
                           thread=threading.get_ident(), **info))

    def observed(self, **kwargs):
        start, cpu_start = time.perf_counter(), time.thread_time()
        first_event = len(events)
        response = None
        try:
            response = original(self, **kwargs)
            return response
        finally:
            end, cpu_end = time.perf_counter(), time.thread_time()
            calls.append(dict(start=start, end=end, wall=end-start,
                              cpu=cpu_end-cpu_start, gc_events=events[first_event:].copy(),
                              original_clock=kwargs['clock'],
                              integration_seconds=(response or {}).get('integration_seconds'),
                              processing_seconds=(response or {}).get('processing_seconds')))

    gc.callbacks.append(gc_event)
    FastFlightStream.advance = observed
    saved_argv = sys.argv
    try:
        sys.argv = ['verify_flight_fast_stream_trial', '--config', args.config,
                    '--source-audit', args.source_audit, '--mode', args.mode,
                    '--output', args.output]
        runpy.run_module('scripts.verify_flight_fast_stream_trial', run_name='__main__')
    finally:
        FastFlightStream.advance = original
        gc.callbacks.remove(gc_event)
        sys.argv = saved_argv
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        evidence = dict(scope=__doc__, gc_policy_changed=False, certificate_eligible=False,
                        gc_originally_enabled=originally_enabled,
                        gc_finally_enabled=gc.isenabled(),
                        observer_source_sha256=sha256(__file__),
                        actuator_messages_sent=0, calls=calls, gc_events=events,
                        slow_calls=[dict(index=i, **call) for i, call in enumerate(calls)
                                    if call['wall'] > .04])
        (root/'runtime-observer.json').write_text(json.dumps(evidence, indent=2), encoding='utf8')
        result_path = root/'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf8'))
            result['underlying_status'] = result['status']
            if result['status'] == 'completed_fast_stream_trial_replay':
                result['status'] = 'completed_fast_stream_runtime_diagnostic'
            result.update(certificate_eligible=False,
                          runtime_observer_sha256=sha256(root/'runtime-observer.json'))
            result_path.write_text(json.dumps(result, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
