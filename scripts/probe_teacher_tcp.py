"""Read-only source cadence and native reproduction-setting observations."""
import argparse
import json
import time

import numpy as np

from flybrain_drive.ac_latest_tcp import LatestTcpTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.source_native_snapshot import WindowsQpc, decode_source_native, source_age_seconds
from scripts.record_ac_wgc import physics_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=12.)
    p.add_argument('--source-native', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if not 2 <= a.seconds <= 60:
        raise ValueError('Bounded 2..60 second read-only probe required')
    run = begin_run(a.output, dict(arguments=vars(a), seed=780102, actuator_messages_sent=0))
    client = None
    qpc = WindowsQpc() if a.source_native else None
    source_ages = []
    result = dict(status='failed', actuator_messages_sent=0, ui_actions_sent=0)
    try:
        client = LatestTcpTelemetryClient()
        start = time.perf_counter()
        with NativePhysicsReader(timeout=.15) as reader:
            initial = reader.receive()
            with (run/'samples.jsonl').open('x', encoding='utf8') as stream:
                while time.perf_counter()-start < a.seconds:
                    packet = client.receive()
                    sample = reader.receive()
                    if qpc is not None:
                        decode_source_native(packet, qpc)
                        source_ages.append(source_age_seconds(packet, qpc)*1000)
                    stream.write(json.dumps(dict(sdk=record_packet(packet), native=physics_record(sample)))+'\n')
                    if abs(sample.fields['speedKmh']) > .1 or sample.fields['gas'] > .02:
                        raise ValueError('Read-only stationary probe observed unexpected motion')
                    time.sleep(.04)
        result.update(status='completed_readonly', initial_native=physics_record(initial),
            final_native=physics_record(sample), source_native_checked=bool(qpc),
            source_capture_age_p50_p95_max_ms=np.percentile(source_ages,[50,95,100]).tolist()
                if source_ages else None)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if client:
            client.close()
            with (run/'tcp-source-packets.jsonl').open('x', encoding='utf8') as stream:
                for packet in client.packets:
                    stream.write(json.dumps(record_packet(packet))+'\n')
            sources = np.array([x.source_monotonic_seconds for x in client.packets])
            receipts = np.array([x.received_monotonic_seconds for x in client.packets])
            def stats(values):
                return dict(median_ms=float(np.median(values)*1000),
                    p99_ms=float(np.quantile(values, .99)*1000), max_ms=float(max(values)*1000),
                    over_80ms=int(sum(values > .08))) if len(values) else None
            result.update(source_packets=len(sources), source_intervals=stats(np.diff(sources)),
                host_receipt_intervals=stats(np.diff(receipts)),
                raw_sha256=sha256(run/'tcp-source-packets.jsonl'),
                receiver_terminated=not client.thread.is_alive())
        write_json(run/'result.json', result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
