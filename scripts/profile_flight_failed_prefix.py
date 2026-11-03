"""Replay every preserved failed-prefix observation, without actuators or new source times."""

import argparse
import base64
import gc
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_scratch_warmup import prime_flight_operators
from flybrain_drive.flight_stream import FlightStream
from flybrain_drive.flight_yaw_linear import apply_linear_yaw_mode
from flybrain_drive.flight_yaw_trial import apply_yaw_mode
from flybrain_drive.flyvis_frozen_tail import install_frozen_visual_tail
from flybrain_drive.flyvis_graphed_tail import install_graphed_visual_tail
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_body import decode_virtual_body_prefix
from scripts.verify_flight_yaw_trial import verified_yaw_trial


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--linear-yaw', action='store_true')
    a = p.parse_args()
    root = Path(a.recording)
    manifest, source = read(root/'manifest.json'), read(root/'result.json')
    assert source['status'] == 'failed' and source['released_neutral_and_pause_verified']
    plan = manifest['config']['plan']
    proof = verified_yaw_trial(plan)
    assert proof == manifest['config']['yaw_verification']
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    inputs = [e for e in events if e['kind'] == 'actor_input']
    references = [e for e in events if e['kind'] == 'inference_complete']
    terminal_guard = len(inputs) == len(references)+1
    assert inputs and (len(inputs) == len(references) or terminal_guard)
    if terminal_guard:
        assert source['error'] == 'ValueError'
        assert source['reason'] == 'Original positive interval exceeds stream budget; no gap repair'
    run = begin_run(a.output, dict(arguments=vars(a), verification=proof,
        events_sha256=sha256(root/'events.jsonl'),
        source_manifest_sha256=sha256(root/'manifest.json'),
        actuator_messages_sent=0, timings='instrumented offline, no live timing qualification'))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0)
    write_json(run/'process.json', dict(pid=os.getpid()))
    try:
        torch.set_num_threads(1)
        torch.manual_seed(manifest['config']['seed'])
        stream = FlightStream(plan['inference_config'], native_identity=read(root/'identity.json'),
                              episode_id=root.name)
        trial = manifest['config']['perturbation_trial']
        yaw_adapter = apply_linear_yaw_mode if a.linear_yaw else apply_yaw_mode
        yaw_adapter(stream, trial['mode'])
        stream.goals[0, 1] = .2
        if plan.get('visual_backend') == 'frozen_visual_tail':
            install_frozen_visual_tail(stream)
        if plan.get('visual_backend') == 'graphed_visual_tail':
            install_graphed_visual_tail(stream)
        gc.collect()
        torch.cuda.empty_cache()
        write_json(run/'scratch-warmup.json', prime_flight_operators(stream))
        calls, rows = [], []

        def instrument(owner, name, label):
            original = getattr(owner, name)

            def measured(*args, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                host = time.perf_counter()
                start.record()
                value = original(*args, **kwargs)
                end.record()
                calls.append((label, start, end, time.perf_counter()-host))
                return value

            setattr(owner, name, measured)

        instrument(stream.network, 'simulate', 'visual')
        instrument(stream.flight, '_advance_elapsed_tensor', 'flight')
        with CameraArchive(root/'vision') as camera:
            camera.verify_all()
            for index, row in enumerate(inputs):
                expected = references[index] if index < len(references) else None
                assert row['phase_index'] >= 0
                if expected is not None:
                    assert row['phase_sample'] == expected['phase_sample']
                stream.goals[0, 0] = row['task_turn_bias']
                pixels = {}
                for view in row['camera']['views']:
                    raw = camera.read(Path(view['file']).name)
                    assert hashlib.sha256(raw).hexdigest() == view['file_sha256']
                    rgb = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
                    assert hashlib.sha256(rgb.tobytes()).hexdigest() == view['rgb_sha256']
                    pixels[view['view_id']] = rgb
                native = row['native']
                sample = decode_virtual_body_prefix(base64.b64decode(native['prefix_base64'],
                    validate=True), native['received_monotonic_seconds'])
                assert sample.prefix_sha256 == native['prefix_sha256']
                calls.clear()
                try:
                    response = stream.advance(clock=row['decision_monotonic_seconds'],
                        camera=row['camera'], pixels=pixels, native_sample=sample)
                except ValueError as error:
                    if expected is not None or not terminal_guard:
                        raise
                    assert str(error) == source['reason'] and stream.failed
                    rows.append(dict(index=index, source_clock=row['decision_monotonic_seconds'],
                        expected_terminal_guard_reproduced=True, error=str(error),
                        observation_removed=False, numerical_integration_attempted=False))
                    break
                # A rejected original observation cannot become accepted.
                assert expected is not None
                torch.cuda.synchronize()
                np.testing.assert_allclose(response['intent'], expected['inference']['intent'],
                                           atol=3e-6, rtol=2e-5)
                np.testing.assert_array_equal(stream.last_body.values, expected['body_values'])
                np.testing.assert_array_equal(stream.last_body.valid, expected['body_valid'])
                rows.append(dict(index=index, source_clock=row['decision_monotonic_seconds'],
                    original_inference=expected['inference'], replay=response,
                    operator_calls=[dict(label=label, gpu_milliseconds=start.elapsed_time(end),
                                         host_enqueue_seconds=host)
                                    for label, start, end, host in calls]))
        write_json(run/'profile.json', rows)
        result.update(status='completed_failed_prefix_profile', observations=len(rows),
            original_full_prefix_replayed=True, hidden_resets=0, numerical_match=True,
            expected_terminal_guard_reproduced=terminal_guard,
            compared_successful_inferences=len(references),
            actual_live_timing_verified=False, allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved())
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
