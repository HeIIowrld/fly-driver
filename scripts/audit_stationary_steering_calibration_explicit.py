"""Verify stationary steering calibration from recorded native bytes; no actuation."""

import argparse
import base64
import hashlib
import json
import statistics
from pathlib import Path

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--amplitude', required=True, type=float)
    args = parser.parse_args()
    source = Path(args.calibration)
    source_result = json.loads((source / 'result.json').read_text(encoding='utf8'))
    rows = [json.loads(line) for line in (source / 'samples.jsonl').read_text(
        encoding='utf8').splitlines()]
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'source_manifest_sha256': sha256(source / 'manifest.json'),
        'source_result_sha256': sha256(source / 'result.json'),
        'samples_sha256': sha256(source / 'samples.jsonl'), 'scope': __doc__})
    (run / 'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'actuator_messages_sent': 0, 'training_admissible': False}
    try:
        samples = [row for row in rows if 'stage' in row]
        amplitude = source_result['request_amplitude']
        if source_result['status'] != 'completed' or len(samples) != 32 or amplitude != args.amplitude or not 0 < amplitude <= 1:
            raise ValueError('Expected completed 32-sample explicit-amplitude calibration')
        previous_packet, previous_clock = -1, -1.
        for number, row in enumerate(samples):
            if (row['stage'], row['index']) != divmod(number, 8):
                raise ValueError('Stage/index order differs')
            request=row['request']['requested_control']
            if request != dict(steering=[0.,amplitude,-amplitude,0.][row['stage']],throttle=0.,brake=1.):
                raise ValueError('Original applied request schedule differs')
            record = row['native']
            raw = base64.b64decode(record['prefix_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != record['prefix_sha256']:
                raise ValueError('Native byte hash differs')
            sample = decode_physics_prefix(raw, record['received_monotonic_seconds'])
            if (sample.packet_id != record['packet_id'] or sample.packet_id <= previous_packet
                    or sample.received_monotonic_seconds <= previous_clock
                    or json.loads(json.dumps(dict(sample.fields))) != record['fields']):
                raise ValueError('Native bytes, decoded fields or receipt order differ')
            previous_packet, previous_clock = sample.packet_id, sample.received_monotonic_seconds
            fields = sample.fields
            if (abs(fields['speedKmh']) >= .1 or fields['gas'] >= .02 or fields['brake'] <= .98
                    or row['native_ai_before']['builtin_ai_controlled'] is not False
                    or row['native_ai_after']['builtin_ai_controlled'] is not False
                    or row['was_foreground_at_record']):
                raise ValueError('Stationary/background/AI-off condition differs')
        native = [statistics.median(row['native']['fields']['steerAngle'] for row in samples
                  if row['stage'] == stage and row['index'] >= 2) for stage in range(4)]
        sdk = [statistics.median(row['sdk_steer_raw'] for row in samples
               if row['stage'] == stage and row['index'] >= 2) for stage in range(4)]
        if (abs(native[0]) > .001 or abs(native[3]) > .001
                or native[1] <= 0 or native[2] >= 0):
            raise ValueError('Native center or sign differs')
        if sdk != source_result['sdk_steering_stage_medians']:
            raise ValueError('SDK stage median differs from recorded result')
        result.update(status='completed_raw_calibration_audit', samples=len(samples),
            first_packet=samples[0]['native']['packet_id'], last_packet=previous_packet,
            native_stage_medians=native, sdk_stage_medians=sdk,
            native_per_request=(native[1]-native[2])/(2*amplitude),
            max_speed_kmh=max(abs(row['native']['fields']['speedKmh']) for row in samples),
            tested_requests=[0, amplitude, -amplitude, 0],
            intermediate_linearity_verified=False, moving_transfer_verified=False,
            independent_physical_units_verified=False, past_recording_codec_verified=False,
            exact_physical_time_verified=False, actor_deployed=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run / 'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
