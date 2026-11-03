# ruff: noqa: E501 -- Exact source-derived nested context is intentional.
"""Replay raw expert-probe pixels/native bytes; verify actor packing, not teacher labels."""

import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_camera_mount import validate_frame_mount
from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_csp_vision import validate_frame_aid_state
from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.ac_shared_texture import verify_shared_frame
from flybrain_drive.actor_cadence_trace_v1 import ActorCadenceProfile, verify_actor_trace
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.frame_persistence_profile_v4 import (
    HostCopyProfile,
    verify_host_persistence_archive,
)
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--actor-trace', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repack-config',
                        help='Optional registered encoding; preserve original audit')
    parser.add_argument('--require-aid-off', action='store_true',
                        help='Verify explicit OFF/non-diagnostic metadata in every raw frame')
    args = parser.parse_args()
    if args.repack_config or not args.require_aid_off:
        raise ValueError('Profile qualification forbids repack and requires aidOFF')
    source = Path(args.recording)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))
    ActorCadenceProfile.from_record(manifest['config']['actor_cadence_profile'])
    profile = HostCopyProfile.from_record(
        manifest['config']['frame_persistence_profile'])
    if not all(manifest['config']['arguments'].get(k) for k in
               ('async_frame_persistence', 'mmf_metadata',
                'same_copy_native_authority')):
        raise ValueError('Explicit bounded MMF same-copy source required')
    bundle = json.loads((source/'vision/bundle.json').read_text(encoding='utf8'))
    archive = json.loads((source/'vision/archive.json').read_text(encoding='utf8'))
    if (sha256(source/'vision/camera.zip') != bundle['sha256']
            or sha256(source/'vision/archive.json') != bundle['archive_manifest_sha256']):
        raise ValueError('Camera archive hashes differ')
    repack_config = (json.loads(Path(args.repack_config).read_text(encoding='utf8'))
                     if args.repack_config else None)
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862953,
        entry_script_sha256=sha256(__file__),
        recording_manifest_sha256=sha256(source/'manifest.json'),
        observation_sha256=sha256(source/'observations.npz'),
        events_sha256=sha256(source/'events.jsonl'), bundle_sha256=bundle['sha256'],
        repack_configuration=repack_config,
        repack_config_sha256=sha256(args.repack_config) if args.repack_config else None))
    events = [json.loads(line) for line in
              (source/'events.jsonl').read_text(encoding='utf8').splitlines()]
    baseline_path = source/'native-before-activation.json'
    baseline_advance_verified = False
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf8'))
        raw = base64.b64decode(baseline['prefix_base64'], validate=True)
        decoded = decode_physics_prefix(raw, baseline['received_monotonic_seconds'])
        if (hashlib.sha256(raw).hexdigest() != baseline['prefix_sha256']
                or decoded.packet_id != baseline['packet_id'] or not events
                or events[0]['native']['packet_id'] <= decoded.packet_id
                or events[0]['native']['received_monotonic_seconds'] <= (
                    decoded.received_monotonic_seconds)):
            raise ValueError('First motion observation must advance the original paused baseline')
        baseline_advance_verified = True
    retina = CSPCameraRetina(manifest['config']['sensory_configuration'])
    errors = {}
    aid_frames = 0
    repacker = CSPCameraRetina(repack_config) if repack_config else None
    repacked = []
    # NpzFile.__getitem__ decompresses the entire selected array each time.
    # Read each array once instead of repeating that work for every observation.
    with load_array(source/'observations.npz') as packed:
        saved = {name: packed[name] for name in packed.files}
    # Stage the verified bundle locally once, avoiding thousands of NAS seeks.
    with CameraArchive(source/'vision') as camera:
        deferred = manifest['config']['arguments'].get('async_frame_persistence', False)
        mmf = manifest['config']['arguments'].get('mmf_metadata', False)
        if mmf and not deferred:
            raise ValueError('MMF stream requires explicit deferred persistence audit')
        persisted_rows = (verify_host_persistence_archive(
            camera, expected_mmf=mmf) if deferred else None)
        initial = camera.json('clock-initial.json')
        clock = CSPClockBounds(initial['generation'],
            maximum_width=initial['maximum_width_seconds'],
            host_quantization=initial.get('host_clock_quantization_seconds', 0.))
        for ping in initial['exchanges']:
            clock.observe(ping['sent'], ping['received'], ping['producer_clock'],
                          initial['generation'])
        if clock.record() != initial:
            raise ValueError('Original causal camera clock bounds differ')
        if len(events) != len(saved['observation_time']):
            raise ValueError('Observation/event counts differ')
        for entry in archive:
            raw = camera.read(entry['file'])
            if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Archived camera file differs: '+entry['file'])
            if re.fullmatch(r'frame-\d{4}\.json', entry['file']):
                validate_frame_mount(json.loads(raw), manifest['config']['sensory_configuration'])
            if args.require_aid_off and re.fullmatch(r'frame-\d{4}\.json', entry['file']):
                frame = json.loads(raw)
                validate_frame_aid_state(frame)
                if frame.get('generation') != clock.generation:
                    raise ValueError('Aid evidence has a different camera generation')
                aid_frames += 1
        if args.require_aid_off:
            recorded = json.loads((source/'result.json').read_text(encoding='utf8'))
            if not aid_frames or aid_frames != recorded['vision']['frames']:
                raise ValueError('Every produced frame requires explicit aid-state evidence')
        for index, event in enumerate(events):
            raw = base64.b64decode(event['native']['prefix_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != event['native']['prefix_sha256']:
                raise ValueError('Native raw prefix hash differs')
            native = decode_physics_prefix(raw, event['native']['received_monotonic_seconds'])
            if native.packet_id != event['native']['packet_id']:
                raise ValueError('Native packet ID differs')
            samples = {}
            row = event['camera']
            pipeline = row.get('shared_pipeline', {})
            if deferred:
                # Actor adds selection-age diagnostics after receipt publication.
                original = persisted_rows[row['capture_id']]
                if any(row.get(key) != value for key, value in original.items()):
                    raise ValueError('Actor receipt differs from deferred immutable stream')
            elif pipeline.get('revision') in ('copy_all_then_ack_then_async_persist_v2',
                    'copy_all_then_mmf_ack_then_async_persist_v3'):
                raise ValueError('Deferred stream requires an explicitly configured audit')
            if args.require_aid_off:
                validate_frame_aid_state(row)
            frame_bytes = camera.read(Path(row['file']).name)
            frame = json.loads(frame_bytes)
            if (hashlib.sha256(frame_bytes).hexdigest() != row['file_sha256']
                    or frame['index'] != row['capture_id']
                    or frame['generation'] != clock.generation):
                raise ValueError('Original camera metadata identity/hash differs')
            verify_shared_frame(frame, row,
                camera.json(f'ack-{row["capture_id"]:04d}.json')
                if frame.get('transport') == 'shared_immutable_v1' else None)
            source_views = {v['view_id']: v for v in frame['views']}
            for view in row['views']:
                raw = camera.read(Path(view['file']).name)
                if hashlib.sha256(raw).hexdigest() != view['file_sha256']:
                    raise ValueError('Selected view hash differs')
                pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
                if hashlib.sha256(pixels.tobytes()).hexdigest() != view['rgb_sha256']:
                    raise ValueError('Decoded RGB hash differs')
                lower = clock.render_start_lower(
                    source_views[view['view_id']]['render_started_clock'], clock.generation)
                if lower != view['render_started_lower']:
                    raise ValueError('Original causal render start bound differs')
                samples[view['view_id']] = dict(rgb=pixels, capture_id=row['capture_id'],
                    generation=row['generation'], received=row['received'],
                    render_started_lower=view['render_started_lower'])
            packed = retina.pack(samples, decision_time=event['decision_monotonic_seconds'],
                native_scalars={k: native.fields[k] for k in retina.required_native_fields},
                native_receipt=native.received_monotonic_seconds)
            if saved['observation_time'][index] != event['decision_monotonic_seconds']:
                raise ValueError('Saved decision time differs')
            for key, value in packed.items():
                original = saved['obs_'+key][index]
                error = float(np.max(np.abs(value.astype(float)-original.astype(float))))
                errors[key] = max(errors.get(key, 0), error)
                if not np.array_equal(value, original):
                    raise ValueError('Raw pixel/body replay differs: '+key)
            if repacker:
                repacked.append(repacker.pack(samples,
                    decision_time=event['decision_monotonic_seconds'],
                    native_scalars={k: native.fields[k] for k in repacker.required_native_fields},
                    native_receipt=native.received_monotonic_seconds))
    intervals = np.diff([event['decision_monotonic_seconds'] for event in events])
    if repacker:
        np.savez_compressed(run/'repacked-observations.npz',
            **{'obs_'+key: np.stack([row[key] for row in repacked]) for key in repacked[0]},
            observation_time=saved['observation_time'])
    if Path(args.actor_trace).resolve() != source.resolve().parent/'actor-trace':
        raise ValueError('Actor trace must be attached to the same original parent')
    recording_result = json.loads((source/'result.json').read_bytes())
    ActorCadenceProfile.from_record(recording_result['actor_cadence_profile'])
    if recording_result['actor_trace_decisions'] != len(events):
        raise ValueError('Every original actor decision trace required')
    actor_trace = verify_actor_trace(args.actor_trace, events,
        expected_cleanup=recording_result['owned_cleanup'])
    write_json(run/'result.json', dict(status='completed', decisions=len(events),
        actor_cadence_trace_verified=actor_trace,
        bounded_frame_profile_verified=profile.record(),

        archive_files_verified=len(archive), actor_replay_max_errors=errors,
        camera_aid_state_verified=args.require_aid_off and aid_frames > 0,
        explicit_aid_off_frames_verified=aid_frames,
        deferred_persistence_verified=deferred,
        raw_mmf_slots_and_original_metadata_verified=mmf,
        maximum_decision_interval_seconds=float(intervals.max()) if len(intervals) else None,
        intervals_above_80ms=int((intervals > .08).sum()),
        distinct_native_packets=len({event['native']['packet_id'] for event in events}),
        preactivation_baseline_advance_verified=baseline_advance_verified,
        actor_truth_inputs_added=False, expert_action_codec_verified=False,
        continuous_40ms_training_admissible=False, labels_created=False,
        repacked_sensory_calibration=repacker.calibration if repacker else None,
        repacked_observations_sha256=sha256(run/'repacked-observations.npz') if repacker else None,
        ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
