"""Read-only native pixel/body fly inference; no SDK teacher, driver or GUI input API."""

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import NativePhysicsReader, decode_static_identity_prefix
from flybrain_drive.ac_pixel_retina import NativePixelRetina
from flybrain_drive.ac_teacher_vision import TeacherVisionRecorder
from flybrain_drive.ac_view_contract import validate_declared_live_view
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from scripts.record_ac_wgc import physics_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--retina-config', default='configs/ac-native-pixel-retina.json')
    parser.add_argument('--decisions', type=int, default=25)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.decisions <= 250:
        raise ValueError('Finite 1..250 shadow decisions required')
    training = Path(args.training_run)
    report = json.loads((training / 'result.json').read_text(encoding='utf8'))
    checkpoint = training / 'checkpoint.pt'
    if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified training checkpoint required')
    torch.set_num_threads(1)
    learned = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = learned['training_configuration']
    if (config['action_profile'] != 'normalized_ac_absolute'
            or config['brain_ticks_per_decision'] != 4
            or config['control_dt_seconds'] != .04):
        raise ValueError('Reviewed native codec and four 10ms neural ticks required')
    retina = NativePixelRetina(json.loads(Path(args.retina_config).read_text(encoding='utf8')))
    live_view_contract = validate_declared_live_view(retina.config)
    if retina.calibration != learned['constructor']['sensory_calibration']:
        raise ValueError('Live pixel/body calibration differs from checkpoint')
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in learned['graph'].items()}
    model = MultiViewCircuitController(graph, **learned['constructor'])
    model.load_state_dict(learned['model'])
    model.eval().requires_grad_(False)
    state = model.initial_state()
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        checkpoint_sha256=sha256(checkpoint), pixel_configuration=retina.config,
        declared_live_view=live_view_contract,
        seed=config['seed'], nominal_period_seconds=.04,
        recurrent_state='reset once at run start; preserve between decisions',
        actuator_messages_sent=0, ui_actions_sent=0))
    reader, vision = None, None
    timings, predictions, sources = [], [], []
    result = dict(status='not_ready', actuator_messages_sent=0, ui_actions_sent=0,
                  student_controls_applied=False, full_lap_verified=False, goal_achieved=False)
    try:
        native = WindowsACWindow()
        windows = native.discover()
        if len(windows) != 1:
            raise ValueError('Exactly one original AC engine window required')
        window = windows[0]
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        write_json(run / 'identity.json', identity)
        if identity['car_model'] != 'bmw_z4_gt3' or identity['track'] != 'monza':
            raise ValueError('Reviewed native BMW Z4 GT3 / Monza required')
        spool = (Path(os.environ['LOCALAPPDATA']) / 'flybrain-review-20261027'
                 / 'student-shadow-spool' / uuid.uuid4().hex / 'vision')
        write_json(run / 'vision-spool.json', dict(directory=str(spool), source_retained=True))
        vision = TeacherVisionRecorder(spool, window.hwnd, max_frames=args.decisions + 40)
        vision.wait_ready()
        reader = NativePhysicsReader(timeout=.15)
        cached_key, rgb = None, None
        with (run / 'shadow.jsonl').open('x', encoding='utf8') as stream, torch.no_grad():
            for index in range(args.decisions):
                started = time.monotonic()
                if (run / 'stop').exists():
                    raise RuntimeError('Owned shadow stop file detected')
                if native.probe(window.hwnd, require_foreground=False) != window:
                    raise ValueError('Original engine/window changed')
                sample = reader.receive()
                candidate = vision.candidate_before(sample.received_monotonic_seconds,
                                                     max_age=retina.max_age)
                if cached_key != candidate['file_sha256']:
                    file = spool / Path(candidate['file']).name
                    if sha256(file) != candidate['file_sha256']:
                        raise ValueError('Committed native image file hash changed')
                    rgb = load_array(file, allow_pickle=False)
                    if hashlib.sha256(rgb.tobytes()).hexdigest() != candidate['rgb_sha256']:
                        raise ValueError('Committed native image pixels changed')
                    cached_key = candidate['file_sha256']
                decision_time = time.monotonic()
                # Only the explicit pose-free body whitelist crosses the actor boundary.
                fields = {key: sample.fields.get(key) for key in retina.required_native_fields}
                packed = retina.pack(rgb, capture_id=candidate['capture_id'],
                    callback_time=candidate['callback_finished_monotonic_seconds'],
                    decision_time=decision_time, native_scalars=fields,
                    native_receipt=sample.received_monotonic_seconds)
                observation = {key: torch.from_numpy(value)[None] for key, value in packed.items()}
                drive = model.prepare_drive(observation)
                for _ in range(config['brain_ticks_per_decision']):
                    prediction, state = model.advance(drive, state)
                value = prediction[0].numpy()
                if (not np.isfinite(value).all() or abs(value[0]) > 1
                        or np.any((value[1:] < 0) | (value[1:] > 1))):
                    raise ValueError('Native student prediction outside action contract')
                elapsed = time.monotonic() - started
                timings.append(elapsed)
                predictions.append(value.copy())
                sources.append(candidate['capture_id'])
                stream.write(json.dumps(dict(index=index, native=physics_record(sample),
                    image_candidate=candidate, actor_field_names=sorted(packed),
                    actor_body_field_names=sorted(fields), actor_prediction=value.tolist(),
                    actual_native_control={k: sample.fields[k] for k in ('gas', 'brake')},
                    decision_monotonic_seconds=decision_time, processing_seconds=elapsed,
                    was_foreground=bool(native.user.GetForegroundWindow() == window.hwnd),
                    controls_applied=False), allow_nan=False) + '\n')
                stream.flush()
                time.sleep(max(0., .04 - (time.monotonic() - started)))
        result.update(status='completed', predictions_min=np.min(predictions, axis=0).tolist(),
                      predictions_max=np.max(predictions, axis=0).tolist())
    except Exception as error:
        result.update(status='aborted', error=type(error).__name__, reason=str(error))
    finally:
        if reader is not None:
            reader.close()
        if vision is not None:
            try:
                result['vision'] = vision.close()
                result['vision_archive'] = vision.archive_to(run / 'vision')
            except Exception as error:
                result['vision_cleanup_error'] = str(error)
        result.update(decisions=len(predictions), unique_images=len(set(sources)),
            processing_seconds_percentiles=np.percentile(timings, [50, 95, 99]).tolist()
                if timings else [], processing_over_40ms=sum(t > .04 for t in timings),
            latency_scope='window probe, native receive, image read/hash, retina and model; '
                          'excludes asynchronous capture latency and JSON logging',
            physical_exposure_sync_verified=False, calibrated_optics=False,
            full_multiview_ready=False)
        write_json(run / 'result.json', result)


if __name__ == '__main__':
    main()
