"""Independent read-only verification of actual stop and vJoy release after a recording."""

import argparse
import json
import os
from pathlib import Path

from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_vjoy import load_vjoy_sdk
from flybrain_drive.evidence import begin_run, write_json
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.recording)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf8'))
    recorded = json.loads((source / 'result.json').read_text(encoding='utf8'))
    if recorded['status'] != 'segment_completed':
        raise ValueError('A completed recording must precede this stop check')
    config = manifest['config']['axis_configuration']
    preflight = json.loads((source / 'preflight.json').read_text(encoding='utf8'))
    packet = FileTelemetryClient(Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry',
                                 timeout=.15).receive()
    with NativePhysicsReader(timeout=.15) as reader:
        sample = reader.receive()
    sdk = load_vjoy_sdk(config['dll_path'], config['dll_sha256'])
    status = sdk.GetVJDStatus(config['device_id'])
    valid = (abs(sample.fields['speedKmh']) <= .1 and sample.fields['gas'] <= .001
        and sample.fields['brake'] >= .98 and status == 1
        and packet.raw['producer_session_id'] == preflight['producer_session_id'])
    run = begin_run(args.output, dict(scope=__doc__, source_run=str(source), seed=862815,
                                     actuator_messages_sent=0))
    write_json(run / 'result.json', dict(status='completed' if valid else 'failed',
        post_stop_verified=valid, native=physics_record(sample), vjoy_status=status,
        sdk_producer=packet.raw['producer_session_id'], actuator_messages_sent=0,
        scope='observed after completed recording, not measured stopping duration',
        goal_achieved=False))
    if not valid:
        raise ValueError('Actual stationary neutral/release check failed')


if __name__ == '__main__':
    main()
