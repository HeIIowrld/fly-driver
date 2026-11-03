"""Full-neuron original225-frame comparison of unchanged upstream steps and frozen-tail calls."""

import argparse
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
from flybrain_drive.flight_stream import FlightStream, held_input_batches
from flybrain_drive.flyvis_fractional_tail import FractionalVisualTail
from flybrain_drive.flyvis_frozen_tail import FrozenVisualTail
from flybrain_drive.flyvis_fullbatch_tail import FullBatchVisualTail
from flybrain_drive.flyvis_graphed_tail import GraphedVisualTail
from flybrain_drive.graph import sha256
from scripts.record_ac_engine_ai_probe import verified_shadow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--base-verification', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--graphed', action='store_true')
    p.add_argument('--fullbatch', action='store_true')
    p.add_argument('--reference-graphed', action='store_true')
    p.add_argument('--fractional', action='store_true')
    a = p.parse_args()
    assert sum((a.graphed, a.fullbatch, a.fractional)) <= 1
    assert not a.reference_graphed or a.fullbatch or a.fractional
    root = Path(a.recording)
    _, proof = verified_shadow(a.config, a.base_verification)
    run = begin_run(a.output, dict(arguments=vars(a), scope=__doc__, base_verification=proof,
        events_sha256=sha256(root/'events.jsonl'), atol=3e-6, rtol=2e-5,
        comparison='every one of4x45669 node activities at every original integration batch',
        actuator_messages_sent=0))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0,
                  actual_live_timing_verified=False, actor_qualified=False)
    write_json(run/'process.json', dict(pid=os.getpid()))
    try:
        torch.set_num_threads(1)
        torch.manual_seed(proof['initialization_seed'])
        stream = FlightStream(a.config, native_identity=read(root/'identity.json'),
                              episode_id=root.name)
        backend = (FractionalVisualTail(stream.network, stream.visual_state) if a.fractional
                   else FullBatchVisualTail(stream.network, stream.visual_state) if a.fullbatch
                   else GraphedVisualTail(stream.network, stream.visual_state) if a.graphed
                   else FrozenVisualTail(stream.network))
        reference_backend = (GraphedVisualTail(stream.network, stream.visual_state)
                             if a.reference_graphed else None)
        reference_simulate = reference_backend.simulate if reference_backend else backend.original
        reference = stream.visual_state
        # Identical canonical initial activity; independent state APIs/caches.
        from flyvis.utils.tensor_utils import AutoDeref

        candidate = stream.network._state_api(AutoDeref(
            nodes=AutoDeref(**{key: value.clone() for key, value in reference.nodes.items()}),
            edges=AutoDeref(**{key: value.clone() for key, value in reference.edges.items()})))
        parameters = {name: value.detach().clone() for name, value in
                      [*stream.network.named_parameters(), *stream.network.named_buffers()]}
        gc.collect()
        torch.cuda.empty_cache()
        rows = [json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()
                if json.loads(s)['kind'] == 'flight_axis_observation']
        # t1154 format is one successful actor observation event per original point.
        assert len(rows) == 225
        previous, held, maximum, batch_count = None, None, 0., 0
        reference_time, candidate_time, reference_peak, candidate_peak = [], [], [], []
        operator_rows = []
        with torch.no_grad(), CameraArchive(root/'vision') as camera:
            for index, row in enumerate(rows):
                clock = row['decision_monotonic_seconds']
                if previous is not None:
                    with torch.device('cuda'):
                        for dt, count in held_input_batches(clock-previous):
                            movie = held[:, None, None].expand(-1, count, -1, -1)
                            states = dict(reference=reference, candidate=candidate)
                            order = ('candidate', 'reference') if batch_count % 2 else (
                                'reference', 'candidate')
                            timings, peaks = {}, {}
                            for label in order:
                                runner = reference_simulate if label == 'reference' else (
                                    backend.simulate)
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                                started = time.perf_counter()
                                states[label] = runner(movie, dt=dt,
                                    initial_state=states[label], as_states=True)[-1]
                                states[label].nodes.activity.cpu()
                                timings[label] = time.perf_counter()-started
                                peaks[label] = torch.cuda.max_memory_allocated()
                            reference, candidate = states['reference'], states['candidate']
                            reference_time.append(timings['reference'])
                            candidate_time.append(timings['candidate'])
                            reference_peak.append(peaks['reference'])
                            candidate_peak.append(peaks['candidate'])
                            operator_rows.append(dict(observation=index, original_clock=clock,
                                dt=dt, steps=count, order=order, seconds=timings))
                            actual, expected = candidate.nodes.activity, reference.nodes.activity
                            error = float((actual-expected).abs().max().cpu())
                            maximum = max(maximum, error)
                            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
                            batch_count += 1
                pixels = {}
                for view in row['camera']['views']:
                    raw = camera.read(Path(view['file']).name)
                    assert hashlib.sha256(raw).hexdigest() == view['file_sha256']
                    rgb = np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
                    assert hashlib.sha256(rgb.tobytes()).hexdigest() == view['rgb_sha256']
                    pixels[view['view_id']] = rgb
                elapsed, signature = stream.order.preview(clock, row['camera'], pixels)
                retinal = np.stack([stream.projector.sample(pixels[name])
                                    for name in ('front', 'left', 'right', 'rear')])
                held = torch.tensor(retinal, device='cuda')
                stream.order.commit(clock, row['camera'], signature)
                previous = clock
                if index % 50 == 0:
                    write_json(run/'progress.json', dict(observations=index+1,
                        maximum_all_node_error=maximum, compared_batches=batch_count))
        for name, value in [*stream.network.named_parameters(), *stream.network.named_buffers()]:
            assert torch.equal(value, parameters[name]), name
        np.savez_compressed(run/'operator-comparison.npz', reference_seconds=reference_time,
            candidate_seconds=candidate_time, reference_peak=reference_peak,
            candidate_peak=candidate_peak)
        write_json(run/'operator-rows.json', operator_rows)
        result.update(status='completed_frozen_visual_tail_stage_comparison', observations=225,
            compared_integration_batches=batch_count, maximum_all_node_error=maximum,
            original_clocks_and_holds_preserved=True, exact_parameter_values_preserved=True,
            original_step_function_used=True, intermediate_dt_unchanged=True,
            reference_seconds_total=float(np.sum(reference_time)),
            candidate_seconds_total=float(np.sum(candidate_time)),
            reference_peak_allocated_bytes=int(max(reference_peak)),
            candidate_peak_allocated_bytes=int(max(candidate_peak)),
            hidden_neural_resets=0, full_coupled_inference_verified=False)
        result.update(reference_backend='graphed_visual_tail' if a.reference_graphed else 'author',
                      timing_order_alternated=True)
        result['visual_backend'] = ('fractional_visual_tail' if a.fractional
                                    else 'fullbatch_visual_tail' if a.fullbatch
                                    else 'graphed_visual_tail' if a.graphed
                                    else 'frozen_visual_tail')
        if a.graphed or a.fullbatch or a.fractional:
            result.update(graphed_steps=backend.graphed_steps,
                          original_fractional_steps=backend.fractional_steps)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
