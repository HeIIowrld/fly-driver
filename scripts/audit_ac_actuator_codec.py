"""Independent offline audit of ordered stationary codec measurements."""

import argparse
import base64
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.actuator_codec import project_pedals
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'executed-audit.py').write_bytes(Path(__file__).read_bytes())
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))
    result = json.loads((source/'result.json').read_text(encoding='utf8'))
    plan = manifest['config']['plan']
    assert plan['mode'] == 'stationary_actuator_codec'
    assert result['events_sha256'] == sha256(source/'events.jsonl')
    assert manifest['config']['program_sha256'] == sha256(source/'executed-program.py')
    assert result['status'] == 'completed_low_speed_maneuver_measurement'
    assert result['stopping_native_acknowledged'] and result['vjoy_free_after_close']
    rows = [json.loads(line) for line in (source/'events.jsonl').read_text().splitlines()]
    groups, group, previous, encodings, pairs, native_count = [], None, None, [], [], 0
    pending_encoding = None
    for row in rows:
        if row['kind'] in ('sample', 'final_native'):
            raw = row['native']
            prefix = base64.b64decode(raw['prefix_base64'], validate=True)
            assert hashlib.sha256(prefix).hexdigest() == raw['prefix_sha256']
            decoded = decode_physics_prefix(prefix, raw['received_monotonic_seconds'])
            assert decoded.packet_id == raw['packet_id']
            assert dict(decoded.fields) == {k: tuple(v) if isinstance(v, list) else v
                                            for k, v in raw['fields'].items()}
            native_count += 1
            applied = [decoded.fields[k] for k in ('steerAngle', 'gas', 'brake')]
            if previous is not None:
                pairs.append(dict(previous_request=previous, native_packet=decoded.packet_id,
                    native_receipt=raw['received_monotonic_seconds'], applied_native=applied))
            if row['kind'] == 'sample':
                assert row['native_ai']['builtin_ai_controlled'] is False
                assert row['native_ai_before']['builtin_ai_controlled'] is False
                if group is None or group['phase'] != row['phase']:
                    group = dict(phase=row['phase'], samples=[])
                    groups.append(group)
                group['samples'].append(applied)
        elif row['kind'] == 'actuator_encoding':
            assert pending_encoding is None
            stage = plan['program'][row['stage_index']]
            assert stage[0] == row['phase']
            assert row['canonical_request'] == stage[1:4]
            expected = project_pedals(stage[1:4])
            for k, value in expected.items():
                assert row[k] == value
            encodings.append(row)
            pending_encoding = row
        elif row['kind'] == 'request':
            receipt = row['receipt']
            assert receipt['driver_accepted'] is True and receipt['game_applied'] is False
            previous = dict(phase=row['phase'], **receipt)
            if pending_encoding is not None:
                assert row['phase'] == pending_encoding['phase']
                assert list(receipt['requested_control'].values()) == (
                    pending_encoding['encoded_request'])
                pending_encoding = None
    assert pending_encoding is None
    stages, max_error, cutoff = [], 0., {}
    program_groups = [g for g in groups if g['phase'].startswith(('raw_', 'encoded_'))]
    assert len(program_groups) == len(plan['program'])
    for index, (group, stage) in enumerate(zip(program_groups, plan['program'], strict=True)):
        assert group['phase'] == stage[0]
        steady = group['samples'][2:]
        assert len(steady) >= plan['analysis_contract']['minimum_steady_samples_per_stage']
        median = [statistics.median(v[i] for v in steady) for i in range(3)]
        canonical = stage[1:4]
        encoded = project_pedals(canonical)['encoded_request'] if stage[0].startswith(
            'encoded_') else canonical
        row = dict(stage_index=index, phase=stage[0], sample_count=len(group['samples']),
            settling_samples=2, steady_samples=len(steady), canonical_request=canonical,
            encoded_request=encoded, applied_native_steady_median=median,
            all_applied_native=group['samples'])
        if stage[0].startswith('encoded_'):
            errors = np.abs(np.asarray(steady)-np.asarray(encoded))
            row['maximum_steady_error'] = float(errors.max())
            max_error = max(max_error, row['maximum_steady_error'])
            assert row['maximum_steady_error'] <= .0001
        else:
            channel = 2 if stage[0].startswith('raw_brake_') else 1
            key = 'brake' if channel == 2 else 'throttle'
            cutoff.setdefault(key, dict(zero_requests=[], positive_requests=[]))
            category = 'zero_requests' if abs(median[channel]) < .0001 else 'positive_requests'
            cutoff[key][category].append(canonical[channel])
            if category == 'positive_requests':
                assert abs(median[channel]-canonical[channel]) < .0001
        stages.append(row)
    summary = dict(status='passed_independent_stationary_codec_audit',
        source_run=str(source), events_sha256=sha256(source/'events.jsonl'),
        native_prefixes_exact=native_count, encoding_records_exact=len(encodings),
        contiguous_stages=len(stages), cutoff=cutoff, maximum_encoded_native_error=max_error,
        samples=result['samples'], maximum_speed_kmh=result['max_speed_kmh'],
        maximum_displacement_m=result['max_displacement_m'],
        stopping_native_acknowledged=True, vjoy_free_after_close=True,
        optimizer_updates=0, ui_actions_sent=0, visual_driving=False, stages=stages,
        limitations=[
            'Steady agreement is not continuous application or exact latency proof.',
            'Neutral stationary capability is not moving deceleration or road following.',
            'Gap projection is an explicit engineering codec; optimizer updates remain zero.'])
    write_json(output/'ordered-request-native-pairs.json', pairs)
    write_json(output/'result.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'stages'}))


if __name__ == '__main__':
    main()
