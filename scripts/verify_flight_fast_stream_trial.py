"""One complete original history in the actual fast stream, with persistent fine-state reference."""

import argparse
import base64
import hashlib
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_vision import validate_frame_aid_state
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_shared_texture import verify_shared_frame
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_fast_haltere import FastHaltereDynamics
from flybrain_drive.flight_fast_stream import FastFlightStream, fast_definition
from flybrain_drive.flight_scratch_warmup import prime_flight_operators
from flybrain_drive.flight_yaw_trial import apply_yaw_mode
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-audit', required=True)
    parser.add_argument('--mode', choices=('full', 'zero_yaw'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config, parent, proof, prototype, _, candidate = fast_definition(args.config)
    source = Path(args.source_audit)
    source_result, source_manifest = read(source/'result.json'), read(source/'manifest.json')
    assert source_result['status'] == 'completed_physical_yaw_trial_audit'
    source_proof = source_result['fast_original_replay']
    assert source_proof['contract']['plan_sha256'] == config['prototype_plan_sha256']
    assert source_proof['contract']['reference_sha256'] == config['motor_reference_sha256']
    assert sha256(source/'fast-hypotheses.npz') == source_proof['archive_sha256']
    root = Path(source_manifest['config']['arguments']['recording'])
    manifest = read(root/'manifest.json')
    plan = manifest['config']['plan']
    assert plan['inference_config'] == config['parent_config']
    assert sha256(root/'events.jsonl') == source_manifest['config']['events_sha256']
    assert sha256(root/'vision/camera.zip') == source_manifest['config']['bundle_sha256']
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    rows = [row for row in events if row['kind'] == 'flight_axis_observation']
    with load_array(source/'fast-hypotheses.npz', allow_pickle=False) as data:
        index = [h['name'] for h in source_proof['hypotheses']].index(config['hypothesis'])
        mode_index = source_proof['scenarios'].index(args.mode)
        target = data['intents'][:, index, mode_index].copy()
        clocks = data['original_clock'].copy()
    assert len(rows) == len(target) == len(clocks) == 257
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        seed=proof['initialization_seed'], fast_config_sha256=sha256(args.config),
        source_audit_result_sha256=sha256(source/'result.json'),
        target_archive_sha256=source_proof['archive_sha256'],
        fine_microstep_seconds=.0005, fine_neuron_tolerance=config['fine_neuron_tolerance'],
        maximum_wall_seconds=.08, percentile95_wall_seconds=.04))
    result = dict(status='failed', goal_achieved=False, actor_qualified=False,
                  actuator_messages_sent=0, ui_actions_sent=0, optimizer_updates=0)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(proof['initialization_seed'])
        stream = FastFlightStream(args.config, native_identity=read(root/'identity.json'),
                                  episode_id=root.name)
        apply_yaw_mode(stream, args.mode)
        write_json(run/'scratch-warmup.json', prime_flight_operators(stream))
        fine = FastHaltereDynamics(stream.flight, prototype['inventory'], maximum_microstep=.0005)
        fine_state, fine_held, previous = stream.flight.initial_state(1), None, None
        exact_state = stream.flight.initial_state(1)
        tracked = []
        for prefix, module in (('flight', stream.flight), ('visual', stream.network)):
            persistent = set(module.state_dict())
            tracked.extend((prefix+'.'+name, value, id(value), value._version) for name, value in
                list(module.named_parameters())+list(module.named_buffers()) if name in persistent)
        maximum_intent, maximum_fine, maximum_exact = 0., 0., 0.
        output_rows, wall_seconds = [], []
        with torch.no_grad(), CameraArchive(root/'vision') as camera:
            camera.verify_all()
            for name in camera.by_name:
                if re.fullmatch(r'frame-\d{4}\.json', name):
                    validate_frame_aid_state(camera.json(name))
            for cursor, row in enumerate(rows):
                clock = row['decision_monotonic_seconds']
                assert clock == clocks[cursor]
                stream.goals[0, 0] = row['task_turn_bias']
                raw = base64.b64decode(row['native']['prefix_base64'], validate=True)
                assert hashlib.sha256(raw).hexdigest() == row['native']['prefix_sha256']
                sample = decode_physics_prefix(raw, row['native']['received_monotonic_seconds'])
                assert json.loads(json.dumps(dict(sample.fields))) == row['native']['fields']
                candidate_frame, pixels = row['camera'], {}
                validate_frame_aid_state(candidate_frame)
                frame_raw = camera.read(Path(candidate_frame['file']).name)
                assert hashlib.sha256(frame_raw).hexdigest() == candidate_frame['file_sha256']
                frame = json.loads(frame_raw)
                verify_shared_frame(frame, candidate_frame,
                    camera.json(f"ack-{candidate_frame['capture_id']:04d}.json"))
                for view in candidate_frame['views']:
                    image_bytes = camera.read(Path(view['file']).name)
                    assert hashlib.sha256(image_bytes).hexdigest() == view['file_sha256']
                    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
                    assert hashlib.sha256(rgb.tobytes()).hexdigest() == view['rgb_sha256']
                    pixels[view['view_id']] = rgb
                prior_actual = stream.flight_state
                prior_snapshot = prior_actual.clone()
                started = time.perf_counter()
                response = stream.advance(clock=clock, camera=candidate_frame, pixels=pixels,
                    native_sample=decode_virtual_body_prefix(
                        raw, sample.received_monotonic_seconds))
                wall_seconds.append(time.perf_counter()-started)
                assert torch.equal(prior_actual, prior_snapshot)
                np.testing.assert_allclose(response['intent'], target[cursor], atol=3e-6, rtol=2e-5)
                maximum_intent = max(maximum_intent,
                    float(np.max(np.abs(np.array(response['intent'])-target[cursor]))))
                assert response['fast_hypothesis'] == config['hypothesis']
                assert response['fast_stream_config_sha256'] == sha256(args.config)
                assert response['fast_yaw_mode'] == args.mode
                assert not response['recurrent_state_reset'] and response['steps'] == cursor+1
                for key in ('visual_ticks', 'flight_ticks', 'capture_id', 'generation'):
                    assert response[key] == row['inference'][key]
                np.testing.assert_array_equal(stream.last_body.values, row['body_values'])
                np.testing.assert_array_equal(stream.last_body.valid, row['body_valid'])
                if previous is not None:
                    elapsed, nominal = clock-previous, stream.flight.spec['dynamics']['dt']
                    count = int(elapsed//nominal)
                    intervals = [nominal]*count
                    if elapsed-count*nominal > 1e-12:
                        intervals.append(elapsed-count*nominal)
                    for dt in intervals:
                        exact_state = stream.dynamics.advance(fine_held, exact_state, dt,
                            coupling=candidate['coupling'], fast_tau=candidate['fast_tau_seconds'])
                        fine_state = fine.advance(fine_held, fine_state, dt,
                            coupling=candidate['coupling'], fast_tau=candidate['fast_tau_seconds'])
                assert torch.isfinite(fine_state).all() and torch.isfinite(exact_state).all()
                assert torch.isfinite(stream.flight_state).all()
                maximum_exact = max(maximum_exact,
                    float((exact_state-stream.flight_state).abs().max()))
                assert maximum_exact <= 3e-6
                maximum_fine = max(maximum_fine,
                    float((fine_state-stream.flight_state).abs().max()))
                assert maximum_fine <= config['fine_neuron_tolerance']
                fine_held = stream.flight_held.clone()
                output_rows.append(response['intent'])
                previous = clock
        assert len(output_rows) == 257
        current = {}
        for prefix, module in (('flight', stream.flight), ('visual', stream.network)):
            persistent = set(module.state_dict())
            current.update({prefix+'.'+name: (id(value), value._version) for name, value in
                list(module.named_parameters())+list(module.named_buffers()) if name in persistent})
        assert current == {name: (identity, version) for name, _, identity, version in tracked}
        stream.flight.assert_body_boundary_protected()
        assert sha256(parent['training']+'/checkpoint.pt') == parent['checkpoint_sha256']
        percentiles = np.percentile(wall_seconds, [50, 95, 100]).tolist()
        np.savez_compressed(run/'outputs.npz', intents=np.array(output_rows), original_clock=clocks,
                            wall_seconds=np.array(wall_seconds))
        result.update(status='completed_fast_stream_trial_replay', original_observations=257,
            mode=args.mode, trial_index=source_result['trial_index'],
            maximum_intent_error=maximum_intent,
            maximum_exact_neuron_error=maximum_exact,
            maximum_fine_neuron_error=maximum_fine, original_clocks_pixels_native_preserved=True,
            fine_reference_persistent=True, fine_reference_hidden_resets=0, hidden_actual_resets=0,
            configured_parameters_unchanged=True, wall_seconds_percentiles=percentiles,
            real_time_budget_passed=percentiles[1] <= .04 and percentiles[2] <= .08,
            original_maximum_interval_seconds=float(np.diff(clocks).max()),
            archive_sha256=sha256(run/'outputs.npz'))
        if config.get('fast_neural_backend') == 'graphed_fast_haltere':
            assert (stream.fast_stepper.nominal_steps > 0
                    and (stream.fast_stepper.fractional_steps
                         + stream.fast_stepper.fractional_graph_steps) > 0)
            if config.get('graph_fractional_intervals'):
                assert stream.fast_stepper.fractional_graph_steps > 0
            result['nominal_graph_calls'] = stream.fast_stepper.nominal_steps
            result['fractional_eager_calls'] = stream.fast_stepper.fractional_steps
            result['fractional_graph_calls'] = stream.fast_stepper.fractional_graph_steps
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
