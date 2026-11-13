"""Eight original paused normal-stream frames; camera-only calibration, never fitting data."""

import argparse
from contextlib import nullcontext
import json
import os
import time
from pathlib import Path

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_camera_mount import configured_mount, validate_frame_mount
from flybrain_drive.ac_csp_vision import CSPVisionRecorder, NoFreshCSPFrame, validate_frame_aid_state
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.immutable_frame_writer import verify_persistence_archive
from flybrain_drive.session_heartbeat_admission import refuse_leased_transport
from scripts.collect_spatial_flight_teacher import idle


def make_frozen_processor(pins, cache, config, *, warmup=False):
    import cv2
    import numpy as np
    import torch

    from extract_next_corner_flyvis import PairedFlyVisFrames, setup
    from flybrain_drive.flyvis_camera import VIEW_ORDER
    from flybrain_drive.local_racing_line import plan_local_line, speed_bound_with_corner_preview
    from flybrain_drive.pixel_road_boundary import PixelRoadBoundary
    from infer_frozen_next_corner import FrozenCornerStream
    from plan_frozen_corner_archive import LIMITS
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    net, projector, indices, pool, step, source = setup(cache, batch_size=8)
    visual = PairedFlyVisFrames(net, projector, indices, pool, step, batched=True)
    readout = FrozenCornerStream(pins, visual, source)
    detector = PixelRoadBoundary(config)
    def process(recorder, row, *, speed_mps=0., tilt=None):
        forecast = readout.push_candidate(recorder, row)
        samples = recorder.samples_for(row)
        images = {name: cv2.cvtColor(view["rgb"], cv2.COLOR_RGB2BGR)
                  for name, view in samples.items()}
        sections = detector.detect(images, tilt=tilt)
        line, status = plan_local_line(sections, speed_mps, **LIMITS)
        preview = None
        if line is not None and forecast["alert"]:
            preview = speed_bound_with_corner_preview(
                line, forecast["distance_m"], corner_speed_mps=20/3.6,
                braking_mps2=LIMITS["braking_mps2"]).tolist()
        return dict(frame=row["capture_id"], **forecast, sections=sections, line=line,
                    path_status=status, preview_speed_mps=preview)
    if warmup:
        pixels = np.full((180, 320, 3), 128, dtype=np.uint8)

        class WarmupCache:
            def samples_for(self, row):
                return {v["view_id"]: dict(rgb=pixels, capture_id=row["capture_id"],
                    generation=row["generation"], received=row["received"],
                    render_started_lower=v["render_started_lower"]) for v in row["views"]}

        dummy = WarmupCache()
        for frame in range(3):
            stamp = frame*.08
            process(dummy, dict(capture_id=frame, generation="synthetic-warmup", received=stamp+.02,
                ideal_line_shown=False, aid_probe_only=False,
                views=[dict(view_id=name, render_started_lower=stamp+i*.003)
                       for i, name in enumerate(VIEW_ORDER)]))
        torch.cuda.synchronize()
    return process


def qualify_frozen_readout(recorder, pins, cache, destination, config):
    """Replay retained paused frames through the same cache-to-prediction path."""
    from train_vision_teacher_line import jsonl

    rows = recorder.recorded_candidates()
    if [row["capture_id"] for row in rows] != list(range(8)):
        raise ValueError("Eight ordered paused camera groups required")
    process = make_frozen_processor(pins, cache, config)
    predictions = [process(recorder, row) for row in rows]
    jsonl(destination, predictions)
    available = [row["frame"] for row in predictions if row["available"]]
    return dict(frames=len(rows), available_frames=available, rows_sha256=sha256(destination),
                pins_sha256=sha256(pins), complete=available == list(range(2,8)),
                paths=sum(row["line"] is not None for row in predictions),
                path_reference_speed_mps=0.,
                live_latency_verified=False)


def observe_frozen_readout(recorder, process, *, body_reader=None, paused_body_encoder=None):
    """Measure delivery age with no route or actuator side effects."""
    from flybrain_drive.ac_native_physics import repeated_snapshot
    from flybrain_drive.virtual_navigation_body import decode_navigation_body_prefix
    from scripts.record_ac_wgc import physics_record

    output = []
    frozen_body = None
    frozen_prefix = None
    deadline = recorder.host_clock()+10.
    while recorder.host_clock() < deadline:
        selected = recorder.host_clock()
        try:
            row = recorder.candidate_before(selected, max_age=.15)
        except NoFreshCSPFrame:
            row = None
        if row is not None and (not output or row["capture_id"] > output[-1]["frame"]):
            before = repeated_snapshot(body_reader.mapping, clock=recorder.host_clock) if body_reader else None
            if paused_body_encoder is None:
                result = process(recorder, row)
            else:
                if frozen_body is None:
                    received = before.received_monotonic_seconds
                    frozen_prefix = before.prefix_bytes
                    sample = decode_navigation_body_prefix(frozen_prefix, received)
                    frozen_body = paused_body_encoder.encode(sample, episode_id=row["generation"],
                                                            decision_monotonic_seconds=received)
                if (before.prefix_bytes != frozen_prefix or frozen_body.episode_id != row["generation"]
                        or not frozen_body.valid[[0, 1, 16]].all()):
                    raise ValueError("Valid unchanged paused body attitude and speed required")
                tilt = frozen_body.values[:2].tolist()
                speed = float(frozen_body.values[16])
                result = process(recorder, row, speed_mps=speed, tilt=tilt)
                result.update(paused_body_input=dict(packet_id=frozen_body.packet_id, tilt_rad=tilt,
                    speed_mps=speed, first_copy_s=frozen_body.original_received_monotonic_seconds,
                    physical_tick_time_verified=False))
            finished = recorder.host_clock()
            if before is not None:
                after = repeated_snapshot(body_reader.mapping, clock=recorder.host_clock)
                if paused_body_encoder is not None and after.prefix_bytes != frozen_prefix:
                    raise ValueError("Paused body changed during projection")
                if not selected <= before.received_monotonic_seconds <= finished <= after.received_monotonic_seconds:
                    raise ValueError("Native body receipt clocks do not bracket processing")
                if (after.packet_id < before.packet_id
                        or after.packet_id == before.packet_id and after.prefix_bytes != before.prefix_bytes):
                    raise ValueError("Native body packet regressed or changed under the same ID")
                result.update(body_before=physics_record(before), body_after=physics_record(after),
                              body_packet_advanced_during_processing=after.packet_id > before.packet_id)
            output.append(dict(result, received_s=row["received"], selected_s=selected,
                finished_s=finished, receive_to_output_s=finished-row["received"],
                oldest_render_to_output_s=finished-min(v["render_started_lower"] for v in row["views"])))
        elif not recorder.thread.is_alive():
            return output
        else:
            time.sleep(.001)
    raise TimeoutError("Paused live readout did not reach recorder termination")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-lua-sha256', help='Pin the exact alternate CSP producer')
    parser.add_argument('--expected-track', help='Require this AC track at admission and in every frame')
    parser.add_argument('--frozen-pins', type=Path, help='Also replay the retained paused frames')
    parser.add_argument('--flyvis-cache', type=Path)
    parser.add_argument('--during-capture', action='store_true',
                        help='Measure paused readout while the camera is delivering groups')
    parser.add_argument('--observe-body', action='store_true',
                        help='Bracket paused readout with original native physics receipts')
    parser.add_argument('--use-paused-attitude', action='store_true',
                        help='Use the unchanged paused native body packet for diagnostic path projection')
    args = parser.parse_args()
    if args.use_paused_attitude and not args.observe_body:
        parser.error('--use-paused-attitude requires --observe-body')
    if args.observe_body and not args.during_capture:
        parser.error('--observe-body requires --during-capture')
    if args.during_capture and args.frozen_pins is None:
        parser.error('--during-capture requires frozen readout inputs')
    if (args.frozen_pins is None) != (args.flyvis_cache is None):
        parser.error('--frozen-pins and --flyvis-cache must be supplied together')
    refuse_leased_transport('qualify_exterior_camera_stream')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    if configured_mount(config) is None:
        raise ValueError('Explicit new mount qualification required')
    status_path = Path(os.environ['LOCALAPPDATA'])/'flybrain/csp-session/status.json'
    refuse_leased_transport('qualify_exterior_camera_stream', status_path=status_path)
    generation = json.loads(status_path.read_text(encoding='utf8'))['generation']
    before = idle(generation)
    track = before['session']['track_id']
    expected_track = args.expected_track or 'monza'
    if track != expected_track:
        raise ValueError('Paused AC track differs from explicitly requested track')
    arguments = {name: str(value) if isinstance(value, Path) else value
                 for name, value in vars(args).items()}
    run = begin_run(args.output, dict(arguments=arguments, program_sha256=sha256(__file__),
        sensory_configuration=config, config_sha256=sha256(args.config),
        before=before, role='calibration', fit_allowed=False))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', role='calibration', fit_allowed=False,
        ui_actions_sent=0, actuator_messages_sent=0, student_deployed=False, goal_achieved=False)
    recorder = None
    try:
        from flybrain_drive.ac_capture import WindowsACWindow

        process = None
        if args.during_capture:
            process = make_frozen_processor(args.frozen_pins, args.flyvis_cache, config, warmup=True)
            if idle(generation)['session']['track_id'] != track:
                raise ValueError('AC track changed while preparing frozen readout')
        refuse_leased_transport('qualify_exterior_camera_stream')
        window = WindowsACWindow().discover()[0]
        from flybrain_drive.ac_native_physics import NativePhysicsReader
        paused_body_encoder = None
        if args.use_paused_attitude:
            from flybrain_drive.ac_image_buffer import ExistingNamedMapping
            from flybrain_drive.ac_native_physics import decode_static_identity_prefix
            from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder
            from plan_frozen_corner_archive import BODY_CONTRACT

            with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
                identity = decode_static_identity_prefix(mapping.read())
            if identity.pop('track') != track:
                raise ValueError('Native identity differs from the paused camera track')
            paused_body_encoder = NavigationBodyEncoder(BODY_CONTRACT, native_identity=identity)
            write_json(run/'paused-body-contract.json', dict(identity=identity,
                navigation_contract_sha256=sha256(BODY_CONTRACT)))

        reader = NativePhysicsReader(clock=time.perf_counter) if args.observe_body else nullcontext()
        with reader as body_reader:
            recorder = CSPVisionRecorder(window.hwnd, config, max_frames=8, period=.08,
                max_bytes=32*1024**2, image_format='bmp', shared_transport=True,
                batched_shared_copy=True, async_frame_persistence=True, mmf_metadata=True,
                expected_source_sha256=args.expected_lua_sha256, expected_track=expected_track)
            if process is None:
                recorder.wait_ready()
                live_rows = None
            else:
                live_rows = observe_frozen_readout(recorder, process, body_reader=body_reader,
                                                   paused_body_encoder=paused_body_encoder)
        deadline = time.monotonic()+10
        while recorder.thread.is_alive():
            if time.monotonic() > deadline:
                raise TimeoutError('Same owned normal-source qualification deadline')
            recorder.thread.join(timeout=.02)
        closed = recorder.close()
        if (closed['error'] is not None or closed['frames'] != 8
                or closed['producer_terminal']['status'] != 'completed'):
            raise ValueError('Entire original paused normal stream did not complete')
        recorder.archive_to(run/'vision', bundle=True)
        with CameraArchive(run/'vision') as archive:
            rows = verify_persistence_archive(archive, expected_mmf=True)
            if len(rows) != 8:
                raise ValueError('Every original source frame required')
            for i in range(8):
                frame = archive.json(f'frame-{i:04d}.json')
                if frame.get('track_id') != track:
                    raise ValueError('Captured frame differs from admitted AC track')
                validate_frame_aid_state(frame)
                validate_frame_mount(frame, config)
        # Publication follows cleanup on the next heartbeat; wait only for same generation.
        deadline = time.monotonic()+1
        while True:
            try:
                after = idle(generation)
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.01)
        if after['session']['track_id'] != track:
            raise ValueError('AC track changed during paused qualification')
        if before['native'] != after['native']:
            raise ValueError('Paused native physical values changed')
        if live_rows is not None:
            from train_vision_teacher_line import jsonl
            trace = run/'live-readout.jsonl'
            jsonl(trace, live_rows)
            result['paused_live_readout'] = dict(
                selected_frames=[r['frame'] for r in live_rows],
                available_frames=[r['frame'] for r in live_rows if r['available']],
                missed_frames=sorted(set(range(8))-{r['frame'] for r in live_rows}),
                rows_sha256=sha256(trace), pins_sha256=sha256(args.frozen_pins),
                moving_stream_qualified=False)
            if args.observe_body:
                result['paused_live_readout']['body_receipts'] = dict(
                    distinct_packet_ids=sorted({row[key]['packet_id'] for row in live_rows
                                                for key in ('body_before', 'body_after')}),
                    advanced_during_processing=sum(row['body_packet_advanced_during_processing']
                                                   for row in live_rows),
                    actor_input=False, physical_tick_time_verified=False,
                    paused_projection_frames=[row['frame'] for row in live_rows
                                              if 'paused_body_input' in row])
        elif args.frozen_pins is not None:
            result['frozen_readout'] = qualify_frozen_readout(
                recorder, args.frozen_pins, args.flyvis_cache, run/'frozen-readout.jsonl', config)
            if not result['frozen_readout']['complete']:
                raise ValueError('Paused camera readout did not produce all six causal predictions')
        result.update(status='completed_paused_exterior_normal_stream', frames=8, track_id=track,
            closed=closed, after=after, native_values_unchanged=True,
            all_raw_slots_and_persistence_verified=True, moving_stream_qualified=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        if recorder is not None and recorder.thread.is_alive():
            result['cleanup'] = recorder.close()
        write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
