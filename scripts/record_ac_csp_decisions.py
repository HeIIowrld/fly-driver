"""Read-only nominal 40ms decisions from an asynchronous camera cache and native organs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', type=int, default=100)
    parser.add_argument('--config', default='configs/ac-csp-retina.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--training-run', help='Optional hash-verified model, read-only inference')
    parser.add_argument('--require-background', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.decisions <= 500:
        raise ValueError('Finite 1..500 decisions required')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    retina = CSPCameraRetina(config)
    model, state, model_hash = None, None, None
    if args.training_run:
        training = Path(args.training_run)
        report = json.loads((training / 'result.json').read_text(encoding='utf8'))
        checkpoint = training / 'checkpoint.pt'
        if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
            raise ValueError('Completed hash-verified checkpoint required')
        learned = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (not learned.get('csp_input_registration')
                or learned['constructor']['sensory_calibration'] != retina.calibration
                or learned['training_configuration']['brain_ticks_per_decision'] != 4
                or learned['training_configuration']['control_dt_seconds'] != .04):
            raise ValueError('New camera calibration and declared model tick timing required')
        graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                        'weights': np.float32}.get(k))
                 for k, v in learned['graph'].items()}
        torch.set_num_threads(1)
        model = MultiViewCircuitController(graph, **learned['constructor'])
        model.load_state_dict(learned['model'])
        model.eval().requires_grad_(False)
        with torch.no_grad():
            warm = model.initial_state()
            for _ in range(4):
                _, warm = model.advance(torch.zeros_like(warm), warm)
        state, model_hash = model.initial_state(), sha256(checkpoint)
    native = WindowsACWindow()
    windows = native.discover()
    if len(windows) != 1:
        raise ValueError('Exactly one engine required')
    window = windows[0]
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862810,
        sensory_configuration=config, engine_pid=window.pid, nominal_period_seconds=.04,
        model_checkpoint_sha256=model_hash,
        ui_actions_sent=0, actuator_messages_sent=0))
    vision, reader = None, None
    observations, times, ids, costs, precise_times, predictions = [], [], [], [], [], []
    result = dict(status='failed', actuator_messages_sent=0, ui_actions_sent=0,
        model_inferences=0, training_admissible=False, goal_achieved=False)
    try:
        vision = CSPVisionRecorder(window.hwnd, config,
            max_frames=int(np.ceil(args.decisions*.04/.03))+40,
            pixel_preprocessor=retina.precompute_pixels)
        vision.wait_ready()
        reader = NativePhysicsReader(timeout=.15, clock=vision.host_clock)
        with (run / 'events.jsonl').open('x', encoding='utf8') as stream:
            for index in range(args.decisions):
                if args.require_background and native.user.GetForegroundWindow() == window.hwnd:
                    raise RuntimeError('Background-only profile; no window focus action is sent')
                started = time.perf_counter()
                body = reader.receive()
                body_done = time.perf_counter()
                candidate = vision.candidate_before(body.received_monotonic_seconds)
                samples = vision.samples_for(candidate)
                decision = vision.host_clock()
                decision_precise = time.perf_counter()
                fields = {key: body.fields.get(key) for key in retina.required_native_fields}
                cache_done = time.perf_counter()
                packed = retina.pack(samples, decision_time=decision, native_scalars=fields,
                    native_receipt=body.received_monotonic_seconds)
                retina_done = time.perf_counter()
                drive_done, brain_done = retina_done, retina_done
                prediction = None
                if model is not None:
                    with torch.no_grad():
                        observation = {k: torch.from_numpy(v)[None] for k, v in packed.items()}
                        drive = model.prepare_drive(observation)
                        drive_done = time.perf_counter()
                        for _ in range(4):
                            value, state = model.advance(drive, state)
                        prediction = value[0].numpy().copy()
                        brain_done = time.perf_counter()
                        predictions.append(prediction)
                observations.append(packed)
                times.append(decision)
                precise_times.append(decision_precise)
                ids.append(candidate['capture_id'])
                cost = time.perf_counter()-started
                costs.append(cost)
                stream.write(json.dumps(dict(index=index, native=physics_record(body),
                    camera=candidate, decision_monotonic_seconds=decision,
                    decision_perf_counter_seconds=decision_precise,
                    student_prediction=prediction.tolist() if prediction is not None else None,
                    processing_seconds=cost,
                    phase_seconds=dict(native_read=body_done-started,
                        camera_cache=cache_done-body_done, retina=retina_done-cache_done,
                        sensory_drive=drive_done-retina_done, brain_ticks=brain_done-drive_done),
                    foreground=bool(native.user.GetForegroundWindow() == window.hwnd)),
                    allow_nan=False)+'\n')
                stream.flush()
                time.sleep(max(0., .04-(time.perf_counter()-started)))
        np.savez_compressed(run / 'observations.npz',
            **{'obs_'+k: np.stack([o[k] for o in observations]) for k in observations[0]},
            observation_time=np.array(times))
        result.update(status='completed', decisions=len(times), unique_camera_groups=len(set(ids)),
            repeated_camera_decisions=len(ids)-len(set(ids)),
            camera_groups_at_decisions=ids,
            actual_decision_interval_percentiles=np.percentile(
                np.diff(precise_times), [50, 95, 99]).tolist() if len(times) > 1 else [],
            processing_seconds_percentiles=np.percentile(costs, [50, 95, 99]).tolist(),
            all_four_views_valid_decisions=sum(bool(o['view_valid'].all()) for o in observations),
            pixel_cache_hits=retina.pixel_cache_hits, pixel_cache_misses=retina.pixel_cache_misses,
            actor_input_pose_free=True)
        if predictions:
            np.savez_compressed(run / 'predictions.npz', prediction=np.array(predictions))
            result.update(model_inferences=len(predictions),
                prediction_min=np.min(predictions, axis=0).tolist(),
                prediction_max=np.max(predictions, axis=0).tolist(),
                final_prediction=predictions[-1].tolist(), expert_labels_available=False,
                recurrent_state='initialized once, four ticks per decision, never reset in episode')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), decisions=len(times))
    finally:
        if reader is not None:
            reader.close()
        if vision is not None:
            result['vision'] = vision.close()
            result['vision_archive'] = vision.archive_to(run / 'vision')
        write_json(run / 'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()
