"""Bounded owned AI motion with native copy windows and camera timing; diagnostic only."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from bisect import bisect_left, bisect_right
from pathlib import Path

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_clock import CSPClockBounds, clock_ping, initialize_clock
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import decode_static_identity_prefix
from flybrain_drive.ac_teacher_physics import timed_teacher_snapshot
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.control_ac_session import read_session_heartbeat, validate_heartbeat


def diagnostic_profile(name):
    """Fixed, finite contracts; the long profile uses the existing CSP brake gate."""
    profiles = {
        'motion3': dict(command='expert-motion-probe', frames=300, camera_deadline=30,
                        child_deadline=20, maximum_samples=20000, steer=1.),
        'brake30': dict(command='expert-brake-pulse30', frames=600, camera_deadline=55,
                        child_deadline=47, maximum_samples=40000, steer=.9),
    }
    if name not in profiles:
        raise ValueError('Unknown bounded diagnostic profile')
    return profiles[name].copy()


def bracket_render(samples, lower, upper):
    if not samples or lower > upper:
        raise ValueError('Ordered native copies and render interval required')
    ends = [row['copy']['second_copy_end'] for row in samples]
    starts = [row['copy']['first_copy_start'] for row in samples]
    if any(b <= a for a, b in zip(starts, starts[1:], strict=False)):
        raise ValueError('Native copies must have strictly increasing host starts')
    earlier = bisect_right(ends, lower)-1
    later = bisect_left(starts, upper)
    if earlier < 0 or later == len(samples):
        return {'status': 'missing_host_bracket'}
    return {'status': 'bracketed_host_render_call',
            'earlier_packet': samples[earlier]['authority']['packet_id'],
            'later_packet': samples[later]['authority']['packet_id'],
            'earlier_copy_end': ends[earlier], 'later_copy_start': starts[later],
            'copy_gap_seconds': starts[later]-ends[earlier]}


class NativeCopySampler:
    """Read a bounded native stream through the whole owned camera lifetime."""

    def __init__(self, mapping_factory, *, maximum_samples=20000, period=.002):
        if type(maximum_samples) is not int or maximum_samples < 2 or not 0 < period <= .02:
            raise ValueError('Explicit bounded sampler budget required')
        self.mapping_factory = mapping_factory
        self.maximum_samples, self.period = maximum_samples, period
        self.rows, self.error = [], None
        self.stop_event, self.ready_event = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._run, name='native-copy-sampler', daemon=True)

    def _run(self):
        try:
            with self.mapping_factory() as mapping:
                while not self.stop_event.is_set():
                    if len(self.rows) >= self.maximum_samples:
                        raise ValueError('Native sampling budget exceeded')
                    sample, timing = timed_teacher_snapshot(mapping)
                    self.rows.append({'authority': sample.authority_record(), 'copy': timing})
                    self.ready_event.set()
                    # Windows Event.wait can be quantized to ~15.6 ms; sleep uses the
                    # high-resolution timer on the supported Python runtime.
                    time.sleep(self.period)
        except Exception as error:
            self.error = type(error).__name__+': '+str(error)
            self.ready_event.set()

    def start(self):
        self.thread.start()
        if not self.ready_event.wait(3) or self.error or not self.rows:
            raise RuntimeError('Native sampler did not become ready: '+str(self.error))

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise TimeoutError('Original native sampler remains live')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', choices=('motion3', 'brake30'), default='motion3',
                        help='Bounded owned expert motion or 30s brake pulse diagnostic')
    parser.add_argument('--shadow-body-history', action='store_true',
                        help='Record online receipt-ordered body contexts; never controls')
    parser.add_argument('--neural-shadow', action='store_true',
                        help='Prepare and run diagnostic FlyVis/flight shadow, no actuation')
    parser.add_argument('--neural-config')
    parser.add_argument('--neural-training')
    parser.add_argument('--neural-opponent')
    parser.add_argument('--neural-verification')
    parser.add_argument('--late-body', action='store_true',
                        help='Select decision body after completed vision inference')
    parser.add_argument('--multirate-shadow', action='store_true',
                        help='Run separate diagnostic body ticker and FlyVis producer')
    parser.add_argument('--multirate-verification')
    parser.add_argument('--split-cuda-streams', action='store_true',
                        help='Run visual and body CUDA work on separate streams')
    parser.add_argument('--split-verification')
    parser.add_argument('--diagnostic-cuda-timing', action='store_true',
                        help='Record stream event spans and body encoder host timing')
    parser.add_argument('--body-period-ms', type=float, default=20.,
                        help='Original body ticker minimum interval, 10..40ms')
    args = parser.parse_args()
    profile = diagnostic_profile(args.profile)
    if (args.late_body and not args.neural_shadow or args.multirate_shadow and
            (not args.late_body or not args.multirate_verification)
            or args.split_cuda_streams and (not args.multirate_shadow or not args.split_verification)
            or args.diagnostic_cuda_timing and not args.split_cuda_streams
            or not 10 <= args.body_period_ms <= 40
            or args.body_period_ms != 20. and not args.multirate_shadow):
        raise ValueError('Multirate requires verified late-body neural shadow')
    neural = None
    if args.neural_shadow:
        if not args.shadow_body_history or not all((args.neural_config, args.neural_training,
                                                    args.neural_opponent, args.neural_verification)):
            raise ValueError('Explicit body shadow, pinned neural sources and verification required')
        from flybrain_drive.online_neural_shadow import OnlineNeuralShadow

        proof_root = Path(args.neural_verification)
        proof = json.loads((proof_root/'result.json').read_text(encoding='utf8'))
        proof_manifest = json.loads((proof_root/'manifest.json').read_text(encoding='utf8'))
        proof_config = proof_manifest['config']
        neural_dependencies = ('online_body_shadow', 'flight_visual_body_temporal_context', 'temporal_body_context',
            'packed_view_body',
            'flight_opponent_temporal_boundary', 'flight_temporal_motion_boundary',
            'flight_opponent_sensory_stream', 'flight_visual_body_airflow',
            'flight_embodied_navigation', 'flight_navigation', 'flight_commands', 'flight_circuit',
            'vehicle_circuit', 'flyvis_camera', 'flyvis_fractional_tail', 'flyvis_graphed_tail',
            'flyvis_frozen_tail', 'flight_stream', 'relative_airflow_sensor', 'per_view_body_history')
        if (proof.get('status') != 'completed_neural_shadow_offline_original_replay'
                or proof.get('maximum_intent_error', float('inf')) > 2e-6
                or proof_config['shadow_source_sha256'] != sha256(
                    'src/flybrain_drive/online_neural_shadow.py')
                or any(proof_manifest['source_sha256'].get('src/flybrain_drive/'+name+'.py') !=
                       sha256('src/flybrain_drive/'+name+'.py') for name in neural_dependencies)
                or args.late_body and (proof.get('late_body_equal_clock_equivalence') is not True
                    or proof_config.get('late_body_source_sha256') != sha256(
                        'src/flybrain_drive/online_neural_late_body.py'))
                or proof['initialization']['config_sha256'] != sha256(args.neural_config)):
            raise ValueError('Exact successful original-history neural shadow proof required')
        # Complete initialization before reading lifecycle/camera state or requesting capture.
        if args.multirate_shadow:
            mr_root = Path(args.multirate_verification)
            mr = json.loads((mr_root/'result.json').read_text(encoding='utf8'))
            mr_config = json.loads((mr_root/'manifest.json').read_text(encoding='utf8'))['config']
            if (mr.get('status') != 'completed_offline_multirate_flight_replay'
                    or mr.get('training_admissible') is not False
                    or mr_config['multirate_source_sha256'] != sha256(
                        'src/flybrain_drive/multirate_flight_stream.py')
                    or mr_config.get('body_period_ms') != args.body_period_ms
                    or mr['events_sha256'] != sha256(mr_root/'events.json')
                    or mr['frame_comparison_sha256'] != sha256(mr_root/'frame-comparison.json')):
                raise ValueError('Exact completed multirate original-history proof required')
            from flybrain_drive.multirate_live_shadow import (
                MultirateLiveShadow, SplitCudaMultirateLiveShadow,
            )
            if args.split_cuda_streams:
                split_root = Path(args.split_verification)
                split = json.loads((split_root/'result.json').read_text(encoding='utf8'))
                split_config = json.loads((split_root/'manifest.json').read_text(encoding='utf8'))['config']
                if (split.get('status') != 'completed_offline_multirate_flight_replay'
                        or split.get('split_cuda_compared_events') != split.get('total_updates')
                        or split.get('split_cuda_max_state_drive_trace_intent_abs', float('inf')) > 2e-6
                        or split_config.get('body_period_ms') != args.body_period_ms
                        or split_config.get('live_cuda_source_sha256') != sha256(
                            'src/flybrain_drive/multirate_live_shadow.py')
                        or split['events_sha256'] != sha256(split_root/'events.json')
                        or any(split_config.get(key) != mr_config.get(key) for key in
                            ('multirate_source_sha256', 'flight_checkpoint_sha256',
                             'opponent_checkpoint_sha256'))):
                    raise ValueError('Exact original-history split CUDA numerical proof required')
            neural_class = SplitCudaMultirateLiveShadow if args.split_cuda_streams else MultirateLiveShadow
            neural = neural_class(config_path=args.neural_config,
                                  training=args.neural_training, opponent=args.neural_opponent)
            if args.diagnostic_cuda_timing:
                neural.enable_diagnostic_cuda_timing()
            if (mr_config['flight_checkpoint_sha256'] != neural.provenance['flight_checkpoint_sha256']
                    or mr_config['opponent_checkpoint_sha256'] != neural.provenance['opponent_checkpoint_sha256']):
                raise ValueError('Exact multirate model lineage required')
        elif args.late_body:
            from flybrain_drive.online_neural_late_body import LateBodyNeuralShadow
            neural = LateBodyNeuralShadow(config_path=args.neural_config,
                                          training=args.neural_training, opponent=args.neural_opponent)
        else:
            neural = OnlineNeuralShadow(config_path=args.neural_config,
                                        training=args.neural_training, opponent=args.neural_opponent)
        if (neural.provenance['flight_checkpoint_sha256'] != proof['initialization']['flight_checkpoint_sha256']
                or neural.provenance['opponent_checkpoint_sha256'] != proof['initialization']['opponent_checkpoint_sha256']):
            raise ValueError('Verified neural shadow checkpoint lineage changed')
    local = Path(os.environ['LOCALAPPDATA'])/'flybrain'
    camera_root = local/'csp-camera'
    status = json.loads((camera_root/'status.json').read_text(encoding='utf8'))
    before, modified, _ = read_session_heartbeat(local/'csp-session/status.json')
    validate_heartbeat(before, time.time()-modified, mutating=True)
    if (status.get('state') == 'capturing' or before['paused'] is not True
            or before['is_online'] or before['camera_active'] or before['ai_controlled']
            or before['generation'] != status['generation']):
        raise ValueError('Fresh paused offline idle camera and AI required')
    windows = WindowsACWindow().discover()
    if len(windows) != 1:
        raise ValueError('Exactly one reviewed simulator required')
    deployed = Path(windows[0].executable).parent/'apps/lua/FlyBrainCamera/FlyBrainCamera.lua'
    candidate = Path('integrations/ac_csp/candidates'
                      '/FlyBrainCamera-airflow-motion-diagnostic-v1.lua')
    if sha256(deployed) != sha256(candidate):
        raise ValueError('Installed candidate differs')
    request_id = 'airflow-' + uuid.uuid4().hex
    # The AI gate requires at least eight seconds of remaining camera deadline.
    # Reserve room for the child process and its provenance capture before activation.
    request = dict(id=request_id, frames=profile['frames'], period_seconds=.08,
                   deadline_seconds=profile['camera_deadline'],
                   exposure=.35, image_format='png', airflow_diagnostic=True,
                   expected_generation=before['generation'])
    if args.profile == 'brake30':
        request['brake_pulse_capture'] = True
    run = begin_run(args.output, dict(program_sha256=sha256(__file__), request=request,
        candidate_sha256=sha256(candidate), before=before, actor_deployed=False,
        diagnostic_profile=args.profile, profile_contract=profile,
        shadow_body_history=args.shadow_body_history,
        shadow_source_sha256=sha256('src/flybrain_drive/online_body_shadow.py')
        if args.shadow_body_history else None,
        body_history_source_sha256=sha256('src/flybrain_drive/per_view_body_history.py')
        if args.shadow_body_history else None,
        neural_source_sha256=sha256('src/flybrain_drive/online_neural_shadow.py')
        if neural else None, neural_provenance=neural.provenance if neural else None,
        late_body_source_sha256=sha256('src/flybrain_drive/online_neural_late_body.py')
        if args.late_body else None,
        multirate_live_source_sha256=sha256('src/flybrain_drive/multirate_live_shadow.py')
        if args.multirate_shadow else None,
        multirate_verification_sha256=sha256(Path(args.multirate_verification)/'result.json')
        if args.multirate_shadow else None,
        split_verification_sha256=sha256(Path(args.split_verification)/'result.json')
        if args.split_cuda_streams else None,
        diagnostic_cuda_timing=args.diagnostic_cuda_timing,
        body_period_ms=args.body_period_ms if args.multirate_shadow else None,
        neural_verification_sha256=sha256(Path(args.neural_verification)/'result.json')
        if neural else None))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', request_id=request_id, actor_deployed=False,
                  ui_actions_sent=0, student_actuator_messages_sent=0)
    child, log, sampler, shadow = None, None, None, None
    directory = camera_root/request_id
    try:
        clock = CSPClockBounds(before['generation'], maximum_width=.1)
        initialize_clock(camera_root, clock, windows[0].hwnd, host_clock=time.perf_counter)
        write_json(run/'clock-before.json', clock.record())
        sampler = NativeCopySampler(lambda: ExistingNamedMapping(
            'Local\\acpmf_physics', AI_STATUS_PREFIX.size),
            maximum_samples=profile['maximum_samples'])
        sampler.start()
        if args.shadow_body_history:
            from flybrain_drive.online_body_shadow import OnlineBodyShadow

            with ExistingNamedMapping('Local\\acpmf_static', 200) as static:
                identity = decode_static_identity_prefix(static.read())
            if args.multirate_shadow:
                neural.attach(native_rows=sampler.rows, native_identity=identity,
                              episode_id=request_id, body_period=args.body_period_ms/1000.)
            shadow = OnlineBodyShadow(directory=directory, request_id=request_id,
                generation=before['generation'], offset_lower=clock.low, native_rows=sampler.rows,
                native_identity=identity, neural_shadow=neural)
            shadow.start()
        write_json(camera_root/'request.json', request)
        deadline = time.monotonic()+8
        while not (directory/'frame-0000.json').exists():
            if (directory/'result.json').exists():
                raise ValueError('Camera terminated before activation: '+
                                 (directory/'result.json').read_text(encoding='utf8'))
            if time.monotonic() >= deadline:
                raise TimeoutError('Original camera startup timeout')
            time.sleep(.02)
        frame = json.loads((directory/'frame-0000.json').read_text(encoding='utf8'))
        if (frame['request_id'] != request_id or not frame.get('aid_probe_only')
                or not frame.get('airflow_diagnostic')):
            raise ValueError('Exact diagnostic camera activation required')
        while True:
            session, session_modified, _ = read_session_heartbeat(local/'csp-session/status.json')
            validate_heartbeat(session, time.time()-session_modified, mutating=True)
            if (session.get('camera_active') is True
                    and session.get('camera_request_id') == request_id):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned camera did not appear in session heartbeat')
            time.sleep(.02)
        if sampler.error:
            raise ValueError('Native sampler failed before control: '+sampler.error)
        log = (run/'control.log').open('x', encoding='utf8')
        child = subprocess.Popen([sys.executable, '-m', 'scripts.control_ac_airflow_motion',
            '--command', profile['command'], '--execute', '--camera-request-id', request_id,
            '--ai-steer-multiplier', str(profile['steer']),
            '--output', str(run/'control')], stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        result['control_pid'] = child.pid
        sample_deadline = time.monotonic()+profile['child_deadline']
        while child.poll() is None and time.monotonic() < sample_deadline:
            time.sleep(.02)
        if child.poll() is None:
            raise TimeoutError('Original control child remains live; never replay')
        code = child.returncode
        result['control_exit_code'] = code
        if code != 0:
            raise RuntimeError('Owned AI motion failed; preserve control receipt and log')
        (directory/'stop').touch(exist_ok=False)
        deadline = time.monotonic()+5
        while not (directory/'result.json').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('Original camera stop outcome unknown')
            time.sleep(.02)
        terminal = json.loads((directory/'result.json').read_text(encoding='utf8'))
        if terminal['status'] not in ('completed', 'stopped'):
            raise ValueError('Original camera terminated unsuccessfully')
        if shadow is not None:
            shadow.stop()
            if args.multirate_shadow:
                neural.stop_ticker()
                write_json(run/'body-ticker.json', neural.ticker_events)
                result.update(ticker_events=len(neural.ticker_events), ticker_error=neural.ticker_error,
                    ticker_stop_reason=neural.ticker_stop_reason,
                    ticker_sha256=sha256(run/'body-ticker.json'))
            write_json(run/'online-body-shadow.json', shadow.events)
            result.update(shadow_events=len(shadow.events), shadow_error=shadow.error,
                          shadow_sha256=sha256(run/'online-body-shadow.json'))
            if shadow.error:
                raise RuntimeError('Online body shadow failed: '+str(shadow.error))
        sampler.stop()
        samples = sampler.rows
        write_json(run/'native-copy-windows.json', samples)
        result.update(native_copy_samples=len(samples), native_copy_error=sampler.error,
                      native_copy_sha256=sha256(run/'native-copy-windows.json'))
        if sampler.error or len(samples) < 2:
            raise ValueError('Complete native copy observation required')
        for _ in range(4):
            clock_ping(camera_root, clock, windows[0].hwnd, host_clock=time.perf_counter)
        write_json(run/'clock-after.json', clock.record())
        rows = []
        render_bounds = []
        for index in range(terminal['frames']):
            path = directory/f'frame-{index:04d}.json'
            raw = path.read_bytes()
            frame = json.loads(raw)
            diag = frame.get('airflow_diagnostic', {})
            if (frame.get('request_id') != request_id or frame['index'] != index
                    or not frame.get('aid_probe_only')
                    or diag.get('source_verified') is not False
                    or diag.get('actor_admissible') is not False
                    or diag.get('generation') != before['generation']):
                raise ValueError('Original diagnostic identity or isolation differs')
            rows.append(dict(index=index, source_file=str(path), sha256=sha256(path),
                             diagnostic=diag))
            for view in frame['views']:
                start = view['render_started_clock']
                lo, hi = start+clock.low, start+clock.high
                render_bounds.append({'index': index, 'view': view['view_id'],
                    'render_host_bounds': [lo, hi], **bracket_render(samples, lo, hi)})
        if not rows:
            raise ValueError('No original diagnostic observations')
        if shadow is not None:
            if len(shadow.events) != len(rows) or any(
                    e['index'] != r['index'] or e['metadata_sha256'] != r['sha256']
                    for e, r in zip(shadow.events, rows, strict=True)):
                raise ValueError('Online shadow must cover exact original camera frames')
        write_json(run/'diagnostic-frames.json', rows)
        write_json(run/'camera-terminal.json', terminal)
        write_json(run/'render-copy-brackets.json', render_bounds)
        result.update(status='completed', frames=len(rows), source_retained=True,
                      camera_terminal=terminal['status'],
                      metadata_sha256=sha256(run/'diagnostic-frames.json'),
                      render_copy_brackets_sha256=sha256(run/'render-copy-brackets.json'),
                      source_clock_uncertainty_seconds=clock.high-clock.low,
                      render_views_bracketed=sum(x['status'] == 'bracketed_host_render_call'
                                                 for x in render_bounds),
                      render_views_total=len(render_bounds), training_admissible=False,
                      physical_sample_or_exposure_time_verified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if shadow is not None and shadow.thread.is_alive():
            shadow.stop()
        if args.multirate_shadow and neural is not None and hasattr(neural, 'ticker'):
            if neural.ticker.is_alive():
                neural.stop_ticker()
            if not (run/'body-ticker.json').exists():
                write_json(run/'body-ticker.json', neural.ticker_events)
                result.update(ticker_events=len(neural.ticker_events), ticker_error=neural.ticker_error,
                    ticker_stop_reason=neural.ticker_stop_reason,
                    ticker_sha256=sha256(run/'body-ticker.json'))
        if shadow is not None and not (run/'online-body-shadow.json').exists():
            write_json(run/'online-body-shadow.json', shadow.events)
            result.update(shadow_events=len(shadow.events), shadow_error=shadow.error,
                          shadow_sha256=sha256(run/'online-body-shadow.json'))
        if sampler is not None:
            sampler.stop()
            if not (run/'native-copy-windows.json').exists():
                write_json(run/'native-copy-windows.json', sampler.rows)
                result.update(native_copy_samples=len(sampler.rows),
                              native_copy_error=sampler.error,
                              native_copy_sha256=sha256(run/'native-copy-windows.json'))
        if child is not None and child.poll() is None:
            result['control_still_live_pid'] = child.pid
        elif directory.exists() and not (directory/'result.json').exists():
            (directory/'stop').touch(exist_ok=True)
        if log:
            log.close()
        write_json(run/'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result.get('reason', 'diagnostic failed'))


if __name__ == '__main__':
    main()
