"""Offline raw420/body308 authority audit; never grants fitting or control permission."""

import argparse
import base64
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'executed-audit.py').write_bytes(Path(__file__).read_bytes())
    checks, rows = 0, []
    result = dict(status='failed', fit_permission_granted=False, actor_deployed=False,
                  goal_achieved=False, live_api_calls=0, gpu_queries=0,
                  actuator_messages_sent=0, optimizer_updates=0)
    try:
        original, manifest = read(source/'result.json'), read(source/'manifest.json')
        arguments = manifest['config']['arguments']
        assert arguments['same_copy_native_authority'] is True
        checks += 1
        assert original['training_admissible'] is False
        assert original['expert_labels_available'] is False
        checks += 2
        proof = read(source/'native-authority-source-proof.json')
        assert proof['schema'] == 'flybrain.native.same_copy_recorder_source_proof.v1'
        assert sha256(source/'executed-program.py') == proof['executed_program_sha256']
        assert proof['before_camera_activation'] is True
        assert proof['before_control_activation'] is True
        assert proof['actor_fields_changed'] is False and proof['fit_permission_granted'] is False
        checks += 5
        for path, reference in proof['source_files'].items():
            data = Path(path).read_bytes()
            assert len(data) == reference['bytes']
            assert hashlib.sha256(data).hexdigest() == reference['sha256']
            checks += 2
        baseline = read(source/'native-before-activation.json')
        baseline_authority = read(source/'native-before-activation-authority.json')
        events = [json.loads(line) for line in (source/'events.jsonl').read_text(
            encoding='utf8').splitlines()]
        assert len(events) == original['decisions']
        checks += 1
        previous = baseline['packet_id']
        for index, (body, authority, decision, is_baseline) in enumerate([
            (baseline, baseline_authority, None, True),
            *((event['native'], event['native_teacher_authority'],
               event['decision_monotonic_seconds'], False) for event in events),
        ]):
            raw = base64.b64decode(authority['prefix_base64'], validate=True)
            actor = base64.b64decode(body['prefix_base64'], validate=True)
            assert len(raw) == 420 and len(actor) == 308 and raw[:308] == actor
            assert hashlib.sha256(raw).hexdigest() == authority['prefix_sha256']
            assert hashlib.sha256(actor).hexdigest() == body['prefix_sha256']
            assert authority['body_prefix_sha256'] == body['prefix_sha256']
            packet = struct.unpack_from('<i', raw, 0)[0]
            flag = struct.unpack_from('<i', raw, 416)[0]
            assert flag in (0, 1) and packet == body['packet_id'] == authority['packet_id']
            assert authority['builtin_ai_controlled'] is bool(flag)
            assert authority['received_monotonic_seconds'] == body['received_monotonic_seconds']
            checks += 7
            decoded = decode_physics_prefix(actor, body['received_monotonic_seconds'])
            fields = {k: tuple(v) if isinstance(v, list) else v for k, v in body['fields'].items()}
            assert dict(decoded.fields) == fields and decoded.packet_id == packet
            assert authority['schema'] == 'flybrain.native.same_copy_teacher_authority.v1'
            assert authority['same_native_copy'] is True and authority['actor_input'] is False
            assert authority['atomicity_verified'] is False
            assert authority['source_time_seconds'] is None
            assert authority['optimal_teacher_control_verified'] is False
            assert authority['fit_permission_granted'] is False
            assert authority['camera_teacher_authority_synchronized'] is False
            checks += 8
            eligible = authority['teacher_authority_eligible']
            if is_baseline:
                assert eligible is False
                checks += 1
                continue
            assert packet > previous and packet > baseline['packet_id']
            previous = packet
            age = decision-body['received_monotonic_seconds']
            assert math.isfinite(age) and age >= 0
            assert authority['decision_monotonic_seconds'] == decision
            assert authority['receipt_age_at_decision_seconds'] == age
            maximum_age = authority['maximum_body_age_seconds']
            conditions = authority['eligibility_checks']
            assert math.isfinite(maximum_age) and maximum_age > 0
            assert set(conditions) == {
                'native_AI_true', 'packet_advanced_since_baseline',
                'CSP_AI_running_at_loop_start', 'receipt_age_within_existing_retina_bound'}
            assert all(type(v) is bool for v in conditions.values())
            assert conditions['native_AI_true'] is bool(flag)
            assert conditions['packet_advanced_since_baseline'] is True
            assert conditions['receipt_age_within_existing_retina_bound'] is (age <= maximum_age)
            assert eligible is all(conditions.values())
            assert authority['eligibility_scope'] == 'same_copy_native_AI_flag_at_body_packet_only'
            assert authority['ineligible_reasons'] == [k for k, v in conditions.items() if not v]
            checks += 13
            rows.append(dict(index=index-1, packet_id=packet, builtin_ai=bool(flag),
                             eligible=eligible, receipt_age_seconds=age,
                             camera_synchronized=False))
        with np.load(source/'observations.npz', allow_pickle=False) as arrays:
            names = list(arrays.files)
            assert not any('authority' in k.lower() or 'builtin_ai' in k.lower() for k in names)
            np.testing.assert_array_equal(arrays['observation_time'], [
                e['decision_monotonic_seconds'] for e in events])
            checks += 2
        result.update(status='completed_offline_same_copy_teacher_authority_audit',
            source_run=str(source), source_status=original['status'], observations=len(events),
            source_result_sha256=sha256(source/'result.json'),
            source_manifest_sha256=sha256(source/'manifest.json'),
            source_events_sha256=sha256(source/'events.jsonl'),
            source_observation_array_sha256=sha256(source/'observations.npz'),
            source_proof_sha256=sha256(source/'native-authority-source-proof.json'),
            executed_recorder_sha256=sha256(source/'executed-program.py'),
            helper_source_references_match_current_files=True,
            original_baseline_packet=baseline['packet_id'],
            original_actor_arrays_unchanged_by_audit=True,
            native_ai_true=sum(r['builtin_ai'] for r in rows),
            native_ai_false=sum(not r['builtin_ai'] for r in rows),
            narrow_eligible=sum(r['eligible'] for r in rows),
            maximum_body_receipt_age_seconds=max(
                (r['receipt_age_seconds'] for r in rows), default=None),
            csp_loop_phase_evidence=(
                'Implementation-backed loop admission; no per-event raw CSP state.'),
            native_camera_source_time_synchronized=False,
            exact_native_source_clock_verified=False,
            engine_atomicity_verified=False, optimal_teacher_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        result['checks_passed'] = checks
        result['program_sha256'] = sha256(__file__)
        write_json(output/'result.json', result)
        write_json(output/'receipt.json', dict(result_sha256=sha256(output/'result.json'),
                                              program_sha256=sha256(__file__)))
    if result['status'] == 'failed':
        raise RuntimeError(result.get('reason', 'Authority audit failed'))


if __name__ == '__main__':
    main()
