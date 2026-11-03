"""One bounded offline engine-AI diagnostic; MMF visual/body shadow never controls."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import decode_static_identity_prefix
from flybrain_drive.ac_teacher_physics import repeated_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_body_shadow import ImmutableBodyShadow
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.record_ac_airflow_timed_diagnostic import NativeCopySampler


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_proofs(bridge_root, split_root, config, training, opponent, body_priority=0):
    bridge_root, split_root = Path(bridge_root), Path(split_root)
    bridge, split = read(bridge_root/'result.json'), read(split_root/'result.json')
    bridge_manifest = read(bridge_root/'manifest.json')
    split_manifest = read(split_root/'manifest.json')
    bm, sm = bridge_manifest['config'], split_manifest['config']
    shadow_hash = sha256('src/flybrain_drive/multirate_live_shadow.py')
    flight_hash = sha256(read(Path(training)/'result.json')['checkpoint'])
    opponent_hash = sha256(Path(opponent)/'temporal-boundary.pt')
    if (bridge.get('status') != 'completed_immutable_shadow_bridge_replay'
            or bridge.get('maximum_state_error', float('inf')) > 2e-6
            or bm.get('shadow_source_sha256') != shadow_hash
            or bm.get('adapter_sha256') != sha256('src/flybrain_drive/immutable_camera_group.py')
            or bm.get('config_sha256') != sha256(config)
            or bridge.get('initialization', {}).get('flight_checkpoint_sha256') != flight_hash
            or bridge.get('initialization', {}).get('opponent_checkpoint_sha256') != opponent_hash
            or sha256(bridge_root/'events.json') != bridge.get('events_sha256')
            or split.get('status') != 'completed_offline_multirate_flight_replay'
            or not split.get('total_updates')
            or split.get('split_cuda_compared_events') != split.get('total_updates')
            or split.get('split_cuda_max_state_drive_trace_intent_abs', float('inf')) > 2e-6
            or sm.get('body_period_ms') != 10.
            or sm.get('body_stream_priority', 0) != body_priority
            or sm.get('live_cuda_source_sha256') != bm['shadow_source_sha256']
            or sha256(split_root/'events.json') != split.get('events_sha256')
            or sha256(split_root/'frame-comparison.json') != split.get('frame_comparison_sha256')
            or sm.get('flight_checkpoint_sha256') != flight_hash
            or sm.get('opponent_checkpoint_sha256') != opponent_hash):
        raise ValueError('Exact immutable bridge, split stream and checkpoint evidence required')
    dependencies = ('src/flybrain_drive/flight_visual_body_temporal_context.py',
                    'src/flybrain_drive/flight_visual_body_airflow.py',
                    'src/flybrain_drive/flight_temporal_motion_boundary.py',
                    'src/flybrain_drive/vehicle_circuit.py',
                    'src/flybrain_drive/packed_view_body.py')
    if any(bridge_manifest['source_sha256'].get(path) != sha256(path)
           or split_manifest['source_sha256'].get(path) != sha256(path) for path in dependencies):
        raise ValueError('Exact body-context source and packing evidence required')
    return {'bridge_result_sha256': sha256(bridge_root/'result.json'),
            'split_result_sha256': sha256(split_root/'result.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'training', 'opponent', 'bridge-proof', 'split-proof', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--diagnostic-cuda-timing', action='store_true',
                   help='Record GPU stream event spans; instrumentation may alter timing')
    p.add_argument('--body-stream-priority', type=int, choices=(-1, 0), default=0)
    a = p.parse_args()
    proofs = verify_proofs(a.bridge_proof, a.split_proof, a.config, a.training, a.opponent,
                           a.body_stream_priority)
    import torch

    from flybrain_drive.multirate_live_shadow import SplitCudaMultirateLiveShadow

    # Prepare all network state before owning a camera or sending any lifecycle request.
    torch.manual_seed(863009)
    neural = SplitCudaMultirateLiveShadow(config_path=a.config, training=a.training,
        opponent=a.opponent, body_stream_priority=a.body_stream_priority)
    if a.diagnostic_cuda_timing:
        neural.enable_diagnostic_cuda_timing()
    cfg = read(a.config)
    camera_config = read(cfg['camera_config'])
    local = Path(os.environ['LOCALAPPDATA'])/'flybrain'
    status_path = local/'csp-session/status.json'
    refuse_leased_transport('bounded MMF body shadow', status_path=status_path)
    windows = WindowsACWindow()
    matches = windows.discover()
    if len(matches) != 1 or windows.user.GetForegroundWindow() == matches[0].hwnd:
        raise ValueError('One original background simulator required')
    from scripts.collect_spatial_flight_teacher import idle

    before = read(status_path)
    preflight = idle(before['generation'])
    with ExistingNamedMapping('Local\\acpmf_physics', AI_STATUS_PREFIX.size) as mapping:
        native_before = repeated_teacher_snapshot(mapping)
    if native_before.builtin_ai_controlled:
        raise ValueError('Independent native AI-off status required')
    with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
        identity = decode_static_identity_prefix(mapping.read())
    run = begin_run(a.output, {'program_sha256': sha256(__file__), 'proofs': proofs,
        'body_shadow_source_sha256': sha256('src/flybrain_drive/immutable_body_shadow.py'),
        'config_sha256': sha256(a.config), 'camera_config_sha256': sha256(cfg['camera_config']),
        'seed': 863009, 'profile': 'single 3s 15kmh engine AI; shadow only',
        'diagnostic_cuda_timing': a.diagnostic_cuda_timing,
        'body_stream_priority': a.body_stream_priority,
        'actor_deployed': False, 'training_admissible': False, 'before': preflight,
        'native_before_packet': native_before.physics.packet_id, 'native_ai_before': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'failed', 'actor_deployed': False, 'training_admissible': False,
              'ui_actions_sent': 0, 'student_actuator_messages_sent': 0}
    camera = sampler = bridge = child = log = None
    try:
        retina = CSPCameraRetina(camera_config)
        camera = CSPVisionRecorder(matches[0].hwnd, camera_config, max_frames=350,
            max_bytes=512*1024**2, period=.04, image_format='bmp', shared_transport=True,
            batched_shared_copy=True, async_pixel_precompute=True, async_frame_persistence=True,
            mmf_metadata=True, source_timing_v1=True, pixel_preprocessor=retina.precompute_pixels)
        camera.wait_ready()
        sampler = NativeCopySampler(lambda: ExistingNamedMapping(
            'Local\\acpmf_physics', AI_STATUS_PREFIX.size))
        sampler.start()
        neural.attach(native_rows=sampler.rows, native_identity=identity,
                      episode_id=camera.request_id, body_period=.01)
        bridge = ImmutableBodyShadow(camera=camera, neural_shadow=neural, native_rows=sampler.rows,
            native_identity=identity, episode_id=camera.request_id, config_path=a.config)
        log = (run/'control.log').open('x', encoding='utf8')
        child = subprocess.Popen([sys.executable, '-m', 'scripts.control_ac_session',
            '--command', 'expert-motion-probe', '--execute', '--camera-request-id',
            camera.request_id, '--output', str(run/'control')], stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        result.update(control_pid=child.pid, camera_request_id=camera.request_id)
        deadline = time.monotonic()+20
        while child.poll() is None and time.monotonic() < deadline:
            if bridge.error is None:
                try:
                    bridge.consume()
                except Exception as error:
                    result['shadow_failure'] = {'type': type(error).__name__, 'reason': str(error)}
            time.sleep(.002)
        if child.poll() is None:
            raise TimeoutError('Original child still live; do not replay or stop its camera')
        result['control_exit_code'] = child.returncode
        if child.returncode != 0:
            raise RuntimeError('Owned AI motion failed; retain the original control receipt')
        result['status'] = 'completed_diagnostic' if bridge.error is None else 'shadow_stopped'
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if child is not None and child.poll() is None:
            result['control_still_live_pid'] = child.pid
        else:
            if camera is not None:
                try:
                    result['camera_close'] = camera.close()
                except Exception as error:
                    result['cleanup_error'] = type(error).__name__+': '+str(error)
        if hasattr(neural, 'ticker'):
            try:
                neural.stop_ticker()
                write_json(run/'body-ticker.json', neural.ticker_events)
                result.update(ticker_sha256=sha256(run/'body-ticker.json'),
                              ticker_error=neural.ticker_error,
                              ticker_stop_reason=neural.ticker_stop_reason)
            except Exception as error:
                result['ticker_cleanup_error'] = type(error).__name__+': '+str(error)
        if sampler is not None:
            try:
                sampler.stop()
                write_json(run/'native-copy-windows.json', sampler.rows)
                result.update(native_sha256=sha256(run/'native-copy-windows.json'),
                              native_error=sampler.error, native_count=len(sampler.rows))
            except Exception as error:
                result['sampler_cleanup_error'] = type(error).__name__+': '+str(error)
        if bridge is not None:
            write_json(run/'bridge-events.json', bridge.events)
            result.update(bridge_sha256=sha256(run/'bridge-events.json'),
                          bridge_count=len(bridge.events), bridge_error=bridge.error)
        if (camera is not None and (child is None or child.poll() is not None)
                and not camera.thread.is_alive() and camera.triggered):
            try:
                result['archive'] = camera.archive_to(run/'vision', bundle=True)
            except Exception as error:
                result['archive_error'] = type(error).__name__+': '+str(error)
        if log:
            log.close()
        if (result['status'] == 'completed_diagnostic' and (
                result.get('ticker_error') or result.get('ticker_stop_reason')
                or result.get('native_error') or result.get('cleanup_error')
                or result.get('ticker_cleanup_error') or result.get('sampler_cleanup_error')
                or result.get('archive_error')
                or result.get('camera_close', {}).get('status') != 'stopped')):
            result['status'] = 'incomplete_diagnostic'
        write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
