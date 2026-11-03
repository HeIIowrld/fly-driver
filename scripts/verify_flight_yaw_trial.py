"""Verify both experimental drive boundaries against all225 original moving observations."""

import argparse
import base64
import gc
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_scratch_warmup import prime_flight_operators
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.flight_yaw_linear import apply_linear_yaw_mode
from flybrain_drive.flight_yaw_trial import apply_yaw_mode
from flybrain_drive.flyvis_frozen_tail import install_frozen_visual_tail
from flybrain_drive.flyvis_graphed_tail import install_graphed_visual_tail
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
from scripts.record_ac_engine_ai_probe import verified_shadow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verified_yaw_trial(plan):
    _, base = verified_shadow(plan['inference_config'], plan['inference_verification'])
    root = Path(plan['yaw_adapter_verification'])
    result, manifest = read(root / 'result.json'), read(root / 'manifest.json')
    if (result['status'] != 'completed_both_yaw_boundaries'
            or result['observations_per_mode'] != 225
            or manifest['config']['base_verification'] != base
            or manifest['source_sha256']['src/flybrain_drive/flight_yaw_trial.py']
            != sha256('src/flybrain_drive/flight_yaw_trial.py')):
        raise ValueError('Same successful full-history yaw-adapter replay required')
    proof = dict(path=str(root), result_sha256=sha256(root / 'result.json'),
                 adapter_sha256=sha256('src/flybrain_drive/flight_yaw_trial.py'))
    if plan.get('scratch_warmup_verification'):
        warm = Path(plan['scratch_warmup_verification'])
        receipt, config = read(warm/'result.json'), read(warm/'manifest.json')
        assert receipt['status'] == 'completed_both_yaw_boundaries'
        assert receipt['scratch_warmup_verified'] and receipt['observations_per_mode'] == 225
        assert config['config']['base_verification'] == base
        file = 'src/flybrain_drive/flight_scratch_warmup.py'
        assert config['source_sha256'][file] == sha256(file)
        for mode in ('full', 'zero_yaw'):
            assert sha256(warm/mode/'result.json') == receipt[mode+'_receipt_sha256']
            assert receipt['maximum_intent_errors'][mode] < 3e-6
        proof['scratch_warmup'] = dict(path=str(warm), result_sha256=sha256(warm/'result.json'),
                                      implementation_sha256=sha256(file))
    if plan.get('visual_backend'):
        backend = plan['visual_backend']
        if backend not in ('frozen_visual_tail', 'graphed_visual_tail'):
            raise ValueError('Unknown visual backend')
        stage = Path(plan['visual_backend_stage_verification'])
        coupled = Path(plan['scratch_warmup_verification'])
        stage_result, stage_manifest = read(stage/'result.json'), read(stage/'manifest.json')
        coupled_result = read(coupled/'result.json')
        file = 'src/flybrain_drive/flyvis_frozen_tail.py'
        assert stage_result['status'] == 'completed_frozen_visual_tail_stage_comparison'
        assert stage_result['observations'] == 225
        assert stage_result['compared_integration_batches'] == 448
        assert stage_result['maximum_all_node_error'] < 3e-6
        for key in ('original_clocks_and_holds_preserved', 'exact_parameter_values_preserved',
                    'original_step_function_used', 'intermediate_dt_unchanged'):
            assert stage_result[key]
        assert stage_result['hidden_neural_resets'] == 0
        assert stage_manifest['config']['base_verification'] == base
        assert stage_result.get('visual_backend', 'frozen_visual_tail') == backend
        flag = ('graphed_visual_tail_verified' if backend == 'graphed_visual_tail'
                else 'frozen_visual_tail_verified')
        assert coupled_result[flag]
        files = [file]
        if backend == 'graphed_visual_tail':
            files.append('src/flybrain_drive/flyvis_graphed_tail.py')
            assert stage_result['graphed_steps'] > 0
            assert stage_result['original_fractional_steps'] > 0
        for receipt_root in (stage, coupled, coupled/'full', coupled/'zero_yaw'):
            for module in files:
                assert read(receipt_root/'manifest.json')['source_sha256'][module] == sha256(module)
        for mode in ('full', 'zero_yaw'):
            child = read(coupled/mode/'result.json')
            assert child[flag] and child['scratch_warmup_verified']
            assert child['original_full_history_replayed'] and child['hidden_resets'] == 0
        proof['visual_backend'] = dict(name=backend,
            implementation_sha256=(sha256(file) if backend == 'frozen_visual_tail'
                                   else {module: sha256(module) for module in files}),
            stage_path=str(stage),
            stage_result_sha256=sha256(stage/'result.json'),
            coupled_result_sha256=sha256(coupled/'result.json'))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', required=True, choices=['full', 'zero_yaw'])
    parser.add_argument('--scratch-warmup', action='store_true')
    parser.add_argument('--frozen-visual-tail', action='store_true')
    parser.add_argument('--graphed-visual-tail', action='store_true')
    parser.add_argument('--linear-yaw', action='store_true')
    args = parser.parse_args()
    assert not (args.frozen_visual_tail and args.graphed_visual_tail)
    root, audit = Path(args.recording), Path(args.audit)
    plan = read(root / 'manifest.json')['config']['plan']
    config, proof = verified_shadow(plan['inference_config'], plan['inference_verification'])
    audit_result = read(audit / 'result.json')
    assert audit_result['status'] == 'completed_moving_body_replay_audit'
    assert read(audit / 'manifest.json')['config']['events_sha256'] == sha256(root / 'events.jsonl')
    run = begin_run(args.output, dict(arguments=vars(args), base_verification=proof,
        events_sha256=sha256(root / 'events.jsonl'),
        reference_sha256=sha256(audit / 'yaw-replay.npz'), actuator_messages_sent=0))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0)
    try:
        rows = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()
                if json.loads(s)['kind'] == 'flight_axis_observation']
        assert len(rows) == 225
        with load_array(audit / 'yaw-replay.npz', allow_pickle=False) as data:
            expected = {mode: data[key].copy() for mode, key in
                        [('full', 'full'), ('zero_yaw', 'zero_yaw')]}
        errors = {}
        torch.set_num_threads(1)
        with CameraArchive(root / 'vision') as camera:
            for mode in (args.mode,):
                torch.manual_seed(proof['initialization_seed'])
                yaw_adapter = apply_linear_yaw_mode if args.linear_yaw else apply_yaw_mode
                stream = FlightStream(plan['inference_config'],
                    native_identity=read(root / 'identity.json'), episode_id=root.name)
                reference_flight = None
                if args.linear_yaw:
                    # Sparse CSR caches cannot be deep-copied. Build the same original
                    # model and load its frozen state, without advancing a reference episode.
                    reference_flight = type(stream.flight)(stream.spec_file,
                        glutamate_sign=stream.flight.glutamate_sign).to(stream.device)
                    reference_flight.load_state_dict(stream.flight.state_dict(), strict=True)
                    reference_flight.eval().requires_grad_(False)
                stream = yaw_adapter(stream, mode)
                stream.goals[0, 1] = .2
                if args.frozen_visual_tail:
                    install_frozen_visual_tail(stream)
                if args.graphed_visual_tail:
                    install_graphed_visual_tail(stream)
                if args.scratch_warmup:
                    gc.collect()
                    torch.cuda.empty_cache()
                    write_json(run / 'scratch-warmup.json', prime_flight_operators(stream))
                maximum, predictions, drive_maximum = 0., [], 0.
                for index, row in enumerate(rows):
                    stream.goals[0, 0] = row['task_turn_bias']
                    pixels = {}
                    for view in row['camera']['views']:
                        raw = camera.read(Path(view['file']).name)
                        assert hashlib.sha256(raw).hexdigest() == view['file_sha256']
                        rgb = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
                        assert hashlib.sha256(rgb.tobytes()).hexdigest() == view['rgb_sha256']
                        pixels[view['view_id']] = rgb
                    n = row['native']
                    sample = decode_virtual_body_prefix(base64.b64decode(n['prefix_base64'],
                        validate=True), n['received_monotonic_seconds'])
                    assert sample.prefix_sha256 == n['prefix_sha256']
                    response = stream.advance(clock=row['decision_monotonic_seconds'],
                        camera=row['camera'], pixels=pixels, native_sample=sample)
                    np.testing.assert_allclose(response['intent'], expected[mode][index],
                                               atol=3e-6, rtol=2e-5)
                    np.testing.assert_array_equal(stream.last_body.values, row['body_values'])
                    np.testing.assert_array_equal(stream.last_body.valid, row['body_valid'])
                    if reference_flight is not None:
                        body_values = np.array(stream.last_body.values, dtype=np.float32)[None]
                        body = torch.tensor(body_values, device=stream.device)
                        valid = torch.tensor(np.array(stream.last_body.valid)[None],
                                             device=stream.device)
                        if mode == 'zero_yaw':
                            body[:, 4] = 0.
                        expected_drive = reference_flight._drive_tensor(stream.visual_zero,
                            stream.visual_valid, body, valid, stream.goals, stream.last_features)
                        torch.testing.assert_close(stream.flight_held, expected_drive,
                                                   atol=3e-6, rtol=2e-5)
                        drive_maximum = max(drive_maximum, float(
                            (stream.flight_held-expected_drive).abs().max().cpu()))
                    maximum = max(maximum, float(np.abs(np.array(response['intent'])
                                                       - expected[mode][index]).max()))
                    predictions.append(response['intent'])
                errors[mode] = maximum
                np.save(run / (mode+'.npy'), np.array(predictions))
                # Explicitly completed separate replay episodes, never resets within one.
                del stream
                gc.collect()
                torch.cuda.empty_cache()
                write_json(run / 'progress.json', dict(completed_modes=list(errors), errors=errors))
        result.update(status='completed_one_yaw_boundary', observations_per_mode=225,
                      scratch_warmup_verified=args.scratch_warmup,
                      frozen_visual_tail_verified=args.frozen_visual_tail,
                      graphed_visual_tail_verified=args.graphed_visual_tail,
                      linear_yaw_verified=args.linear_yaw,
                      original_drive_verified=args.linear_yaw,
                      maximum_drive_error=drive_maximum if args.linear_yaw else None,
                      maximum_intent_errors=errors, original_full_history_replayed=True,
                      original_sensor_values_and_masks_preserved=True, hidden_resets=0)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run / 'result.json', result)
        raise
    write_json(run / 'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
