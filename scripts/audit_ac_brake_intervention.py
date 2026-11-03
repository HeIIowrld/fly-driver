"""Independent native response to a forced AI brake intervention; no fitting/actor."""

import argparse
import base64
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--raw-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, raw_audit = Path(args.recording), Path(args.raw_audit)
    audit = read(raw_audit/'result.json')
    assert audit['status'] == 'completed' and not any(audit['actor_replay_max_errors'].values())
    audit_config = read(raw_audit/'manifest.json')['config']
    assert audit_config['events_sha256'] == sha256(root/'events.jsonl')
    assert audit_config['bundle_sha256'] == sha256(root/'vision/camera.zip')
    receipt = read(root/'control/api-receipt.json')
    request = read(root/'control/request.json')
    assert request['command'] == receipt['command'] == 'expert-brake-pulse30'
    assert request['brake_pulse_seconds'] == .4 and request['brake_pulse_after_seconds'] == 3
    calls = [x['name'] for x in receipt['api_calls']]
    assert calls.count('force_ai_brake_pulse') == calls.count('clear_ai_brake_pulse') == 1
    assert all(x['returned'] is True for x in receipt['api_calls'])
    assert read(root/'control/result.json')['native_motion_receipt_clock'] == (
        'host_monotonic_QPC_time.perf_counter')
    with zipfile.ZipFile(root/'vision/camera.zip') as archive:
        initial = json.loads(archive.read('clock-initial.json'))
    clock = CSPClockBounds(initial['generation'],
        maximum_width=initial['maximum_width_seconds'],
        host_quantization=initial.get('host_clock_quantization_seconds', 0))
    for ping in initial['exchanges']:
        clock.observe(ping['sent'], ping['received'], ping['producer_clock'], initial['generation'])
    assert clock.record() == initial and receipt['before']['generation'] == clock.generation
    pulse = receipt['brake_pulse_issued_clock']
    lower, upper = pulse+clock.low, pulse+clock.high
    samples, seen = [], set()
    for row in read(root/'control/native-motion-samples.json'):
        raw = base64.b64decode(row['prefix_base64'], validate=True)
        assert hashlib.sha256(raw).hexdigest() == row['prefix_sha256']
        sample = decode_physics_prefix(raw, row['received_monotonic_seconds'])
        assert sample.packet_id == row['packet_id']
        if sample.packet_id in seen:
            continue
        seen.add(sample.packet_id)
        fields = sample.fields
        samples.append(dict(packet_id=sample.packet_id, time=sample.received_monotonic_seconds,
            speed_kmh=fields['speedKmh'], brake=fields['brake'], gas=fields['gas'],
            wheel_load=list(fields['wheelLoad'])))
    before = [s for s in samples if lower-.5 <= s['time'] < lower]
    after = [s for s in samples if upper <= s['time'] <= lower+1.5]
    assert before and after and max(s['packet_id'] for s in before) < min(
        s['packet_id'] for s in after)
    baseline = before[-1]['speed_kmh']
    braking = [s for s in after if s['brake'] > .05]
    minimum = min(s['speed_kmh'] for s in after)
    passed = bool(braking and baseline-minimum >= 1)
    run = begin_run(args.output, dict(arguments=vars(args), program_sha256=sha256(__file__),
        request_sha256=sha256(root/'control/request.json'),
        receipt_sha256=sha256(root/'control/api-receipt.json'),
        native_samples_sha256=sha256(root/'control/native-motion-samples.json'),
        raw_audit_sha256=sha256(raw_audit/'result.json'), role='development_diagnostic',
        fit_allowed=False, physical_command_or_exposure_time_verified=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'native-response.json', dict(before=before, after=after))
    def front_share(rows):
        values = np.array([s['wheel_load'] for s in rows])
        return float(np.mean(values[:, :2].sum(1)/values.sum(1)))
    write_json(run/'result.json', dict(status='completed_independent_native_brake_response_audit',
        native_response_observed=passed, request_attempts=1, clear_attempts=1,
        requested_api_time_argument_seconds=.4, actual_brake_duration_assumed=False,
        pulse_host_receipt_bounds=[lower, upper], clock_uncertainty_seconds=upper-lower,
        native_pre_pulse_speed_kmh=baseline, minimum_native_speed_within1_5s_kmh=minimum,
        native_peak_brake=max(s['brake'] for s in after), braking_native_samples=len(braking),
        sampled_braking_span_seconds=braking[-1]['time']-braking[0]['time'] if braking else 0,
        sampled_span_not_exact_physical_duration=True,
        mean_front_load_share_before=front_share(before),
        mean_front_load_share_braking=front_share(braking) if braking else None,
        source_raw_audit_continuous_training_admissible=audit['continuous_40ms_training_admissible'],
        role='development_diagnostic', fit_allowed=False, natural_ai_labels=False,
        physiological_or_closed_loop_student_claim=False, actuator_messages_sent=0,
        ui_actions_sent=0, goal_achieved=False))
    assert passed, 'No measured native brake/slowdown response; retain diagnostic failure'


if __name__ == '__main__':
    main()
