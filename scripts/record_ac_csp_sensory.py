"""Bounded four-camera/native-body recording, interval-clock alignment and actor packing."""

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_clock import CSPClockBounds, clock_ping
from flybrain_drive.ac_csp_geometry import camera_geometry, matrix
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_file import FileTelemetryClient
from flybrain_drive.ac_native_physics import NativePhysicsReader
from flybrain_drive.ac_tcp import record_packet
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.record_ac_wgc import physics_record


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--period', type=float, default=.1)
    parser.add_argument('--config', default='configs/ac-csp-retina.json')
    parser.add_argument('--native-only', action='store_true', help='Diagnostic: omit SDK entirely; never synthesize or reuse missing SDK values')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if (not 1 <= args.frames <= 100 or not np.isfinite(args.period)
            or not .04 <= args.period <= .25):
        raise ValueError('Finite 1..100 frame and .04..25s period budgets required')
    config = read_json(Path(args.config))
    retina = CSPCameraRetina(config)
    root = Path(os.environ['LOCALAPPDATA']) / 'flybrain/csp-camera'
    status = read_json(root / 'status.json')
    if (time.time()-(root / 'status.json').stat().st_mtime > 3
            or status['patch_version'] != '0.2.11' or status['state'] == 'capturing'
            or not status.get('generation')):
        raise ValueError('Current reviewed idle producer with generation required')
    native = WindowsACWindow()
    windows = native.discover()
    if len(windows) != 1:
        raise ValueError('Exactly one original AC engine required')
    window = windows[0]
    deployed = Path(window.executable).parent / 'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    if sha256(deployed) != sha256('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua'):
        raise ValueError('Deployed camera source differs')
    request_id = 'sensory-' + uuid.uuid4().hex
    directory = root / request_id
    spool = root / 'client-recordings' / request_id
    spool.mkdir(parents=True, exist_ok=False)
    timeout = min(60., max(10., args.frames*args.period+5))
    request = dict(id=request_id, frames=args.frames, period_seconds=args.period,
        deadline_seconds=timeout, exposure=.35, expected_generation=status['generation'])
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862808,
        request=request, engine_pid=window.pid, hwnd=window.hwnd,
        camera_source_sha256=sha256(deployed), sensory_configuration=config,
        source_retained=True, spool=str(spool), ui_actions_sent=0, actuator_messages_sent=0))
    shutil.copy2(deployed, run / 'deployed-camera.lua')
    clock = CSPClockBounds(status['generation'],
        host_quantization=time.get_clock_info('monotonic').resolution)
    result = dict(status='failed', actor_stream_created=False, training_admissible=False,
        actuator_messages_sent=0, ui_actions_sent=0, goal_achieved=False)
    observations, decisions, summaries = [], [], []
    reader = None
    triggered = False
    try:
        for _ in range(4):
            clock_ping(root, clock, window.hwnd)
            time.sleep(.04)
        frozen_clock = clock.record()
        if frozen_clock['uncertainty_seconds'] > .1:
            raise ValueError('Clock uncertainty exceeds declared budget')
        write_json(run / 'initial-clock.json', frozen_clock)
        reader = NativePhysicsReader(timeout=.15)
        sdk = None if args.native_only else FileTelemetryClient(
            Path(os.environ['LOCALAPPDATA']) / 'flybrain/telemetry', timeout=.15)
        producer = None if sdk is None else sdk.session_id
        deadline = time.monotonic()+timeout+2
        write_json(root / 'request.json', request)
        triggered = True
        with (spool / 'events.jsonl').open('x', encoding='utf8') as stream:
            for index in range(args.frames):
                path = directory / f'frame-{index:04d}.json'
                while not path.exists():
                    if (directory / 'result.json').exists():
                        terminal = read_json(directory / 'result.json')
                        if terminal['status'] != 'completed':
                            raise ValueError('Camera producer failed: ' + str(terminal))
                    if time.monotonic() > deadline:
                        raise TimeoutError('Camera recording deadline exceeded')
                    time.sleep(.002)
                frame = read_json(path)
                received = time.monotonic()
                if (frame['request_id'] != request_id or frame['index'] != index
                        or frame['generation'] != clock.generation
                        or frame['car_id'] != 'bmw_z4_gt3' or frame['track_id'] != 'monza'
                        or frame.get('camera_layout_revision') != 2):
                    raise ValueError('Camera identity or reviewed layout changed')
                camera_geometry(frame['views'])
                samples, image_rows = {}, []
                for view in frame['views']:
                    name = view['view_id']
                    p = matrix(view['projection_matrix_diagnostic'])
                    expected = config['views'][name]
                    if (not np.isclose(p[0, 0], expected['projection_x'], atol=1e-6)
                            or not np.isclose(p[1, 1], expected['projection_y'], atol=1e-6)
                            or view['requested_exposure'] != .35
                            or view['file'] != f'{index:04d}-{name}.png'):
                        raise ValueError('Runtime optics, exposure or image identity changed')
                    file = directory / view['file']
                    with Image.open(file) as image:
                        pixels = np.array(image.convert('RGB'))
                    samples[name] = dict(rgb=pixels, capture_id=index,
                        generation=clock.generation,
                        render_started_lower=clock.render_start_lower(
                            view['render_started_clock'], clock.generation),
                        received=received)
                    image_rows.append(dict(view_id=name, file=view['file'], sha256=sha256(file),
                        rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
                body = reader.receive()
                packet = None if sdk is None else sdk.receive()
                if ((packet is not None and packet.raw['producer_session_id'] != producer)
                        or native.probe(window.hwnd, require_foreground=False) != window):
                    raise ValueError('Engine or SDK producer changed during recording')
                decision = time.monotonic()
                fields = {k: body.fields.get(k) for k in retina.required_native_fields}
                packed = retina.pack(samples, decision_time=decision,
                    native_scalars=fields, native_receipt=body.received_monotonic_seconds)
                observations.append(packed)
                decisions.append(decision)
                summary = dict(index=index, view_valid_counts=packed['view_valid'].sum(1).tolist(),
                    contrast_valid_counts=packed['contrast_valid'].sum(1).tolist(),
                    organ_valid_count=int(packed['organ_valid'].sum()),
                    view_age_upper_seconds=packed['view_age'].tolist(),
                    foreground=bool(native.user.GetForegroundWindow() == window.hwnd))
                summaries.append(summary)
                stream.write(json.dumps(dict(kind='sensory_pair', index=index,
                    host_image_receipt=received, decision_monotonic_seconds=decision,
                    native=physics_record(body), sdk=None if packet is None else record_packet(packet),
                    camera_metadata_file=path.name, camera_metadata_sha256=sha256(path),
                    image_sources=image_rows, actor_body_field_names=sorted(fields),
                    actor_field_names=sorted(packed), summary=summary), allow_nan=False)+'\n')
                stream.flush()
        while not (directory / 'result.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Camera terminal result missing')
            time.sleep(.01)
        terminal = read_json(directory / 'result.json')
        if terminal['status'] != 'completed' or terminal['frames'] != args.frames:
            raise ValueError('Incomplete producer terminal result')
        # Future clock exchanges validate the fixed interval used online; do not rewrite samples.
        for _ in range(2):
            clock_ping(root, clock, window.hwnd)
            time.sleep(.04)
        write_json(run / 'final-clock.json', clock.record())
        np.savez_compressed(spool / 'observations.npz',
            **{'obs_'+key: np.stack([o[key] for o in observations]) for key in observations[0]},
            observation_time=np.array(decisions))
        result.update(status='completed', frames=len(observations), actor_stream_created=not args.native_only,
            sdk_omitted=args.native_only,
            all_four_views_valid_frames=sum(all(n > 0 for n in s['view_valid_counts'])
                                           for s in summaries), summaries=summaries,
            native_sdk_producer=producer,
            clock_uncertainty_seconds=frozen_clock['uncertainty_seconds'],
            input_whitelist_pose_free=True, model_inferences=0,
            physical_exposure_sync_verified=False, independent_final_test=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), frames=len(observations))
    finally:
        if reader is not None:
            reader.close()
        # A failed consumer does not restart a still-running bounded producer.
        if triggered:
            wait_deadline = time.monotonic()+timeout+2
            while not (directory / 'result.json').exists() and time.monotonic() < wait_deadline:
                time.sleep(.05)
            result['producer_terminal_observed'] = (directory / 'result.json').exists()
        archive = []
        sources = [(spool, 'client'), (directory, 'camera')] if triggered else [(spool, 'client')]
        for source, label in sources:
            if not source.exists():
                continue
            destination = run / label
            destination.mkdir()
            for file in sorted(source.iterdir()):
                if not file.is_file() or file.suffix not in ('.json', '.jsonl', '.npz', '.png'):
                    continue
                target = destination / file.name
                shutil.copy2(file, target)
                if sha256(file) != sha256(target):
                    raise ValueError('Archive hash mismatch')
                archive.append(dict(file=target.relative_to(run).as_posix(), sha256=sha256(target)))
        write_json(run / 'archive.json', archive)
        write_json(run / 'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result.get('reason', 'Sensory recording failed'))


if __name__ == '__main__':
    main()
