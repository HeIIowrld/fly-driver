"""Seal executed CSP-camera teacher data after exact pixel/body and policy-request replay."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_clock import CSPClockBounds
from flybrain_drive.ac_csp_retina import CSPCameraRetina
from flybrain_drive.ac_geometric_teacher import GeometricRecoveryTeacher
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT, PHYSICS_PREFIX, decode_physics_prefix
from flybrain_drive.ac_sac_teacher import (
    PrivilegedSACTeacher,
    legacy_request_to_normalized_controls,
    relative_request,
)
from flybrain_drive.ac_teacher_geometry import PrivilegedTrackGeometry
from flybrain_drive.ac_teacher_pilot import collection_speed_request
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_learning import load_episodes
from flybrain_drive.multiview_sensory import MultiViewPacker


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--post-stop', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.recording)
    manifest, recorded = load(root / 'manifest.json'), load(root / 'result.json')
    config = manifest['config']
    kind = config['arguments']['teacher_kind']
    if (recorded['status'] != 'segment_completed' or config['arguments']['camera_backend'] != 'csp'
            or kind not in ('sac', 'geometric_recovery')):
        raise ValueError('Completed actual CSP SAC or geometric teacher segment required')
    stop = load(args.post_stop)
    if not stop['post_stop_verified'] or stop['vjoy_status'] != 1:
        raise ValueError('Independent actual stop and driver release evidence required')
    producer = load(root / 'preflight.json')['producer_session_id']
    if stop['sdk_producer'] != producer:
        raise ValueError('Post-stop producer differs')
    stop_manifest = load(Path(args.post_stop).parent / 'manifest.json')
    if Path(stop_manifest['config']['source_run']).resolve() != root.resolve():
        raise ValueError('Stop verification belongs to another recording')
    archive = CameraArchive(root/'vision')
    expected_bundle = recorded['vision'].get('archive_receipt', {}).get('bundle_sha256')
    if expected_bundle is not None and archive.bundle_sha256 != expected_bundle:
        raise ValueError('Camera bundle differs from collection receipt')
    archive.verify_all()
    if sha256(root / 'actor-observations.npz') != recorded['actor_observations_sha256']:
        raise ValueError('Recorded actor observations changed')
    codec = config['teacher_configuration']
    if kind == 'sac' and sha256(codec['checkpoint']) != codec['checkpoint_sha256']:
        raise ValueError('Pinned public SAC teacher changed')
    torch.set_num_threads(1)
    if kind == 'sac':
        teacher = PrivilegedSACTeacher(codec['checkpoint'], codec['checkpoint_sha256'])
    else:
        implementation = 'src/flybrain_drive/ac_geometric_teacher.py'
        if sha256(implementation) != manifest['source_sha256'][implementation]:
            raise ValueError('Geometric teacher implementation changed since collection')
        fit_path = config['arguments']['kinematic_calibration']
        fit = load(fit_path)
        if (fit != config['kinematic_calibration'] or fit['status'] != 'completed'
                or not fit['teacher_only'] or fit['producer_session_id'] != producer
                or fit['profile_sha256'] != config['axis_configuration']['profile_sha256']):
            raise ValueError('Recorded same-session geometric calibration differs')
        geo = config['geometry_configuration']
        geometry = PrivilegedTrackGeometry(geo['track_csv'], geo['reference_csv'],
            track_sha256=geo['track_sha256'], reference_sha256=geo['reference_sha256'])
        teacher = GeometricRecoveryTeacher(geometry.line, fit['curvature_per_canonical_steering'],
            maximum_speed_kmh=config['arguments'].get('geometric_target_speed_kmh', 20.))
    retina = CSPCameraRetina(config['csp_sensory_configuration'])
    initial = archive.json('clock-initial.json')
    final = archive.json('clock-final.json')
    clock = CSPClockBounds(initial['generation'], maximum_width=initial['maximum_width_seconds'],
        host_quantization=initial.get('host_clock_quantization_seconds', 0.))
    for ping in initial['exchanges']:
        clock.observe(ping['sent'], ping['received'], ping['producer_clock'], initial['generation'])
    if clock.record() != initial:
        raise ValueError('Initial camera clock does not reproduce')
    events_path = root / 'observations-and-requests.jsonl'
    if recorded.get('events_archive_sha256') is not None and (
            not recorded.get('events_archive_verified')
            or sha256(events_path) != recorded['events_archive_sha256']):
        raise ValueError('Verified local event spool/NAS archive differs')
    events = [json.loads(x) for x in events_path.read_text(encoding='utf8').splitlines()]
    pairs, requests, feedback = {}, [], []
    for event in events:
        if event['kind'] == 'raw_pair':
            for key, tag in [('sdk', 'wire'), ('native', 'prefix')]:
                raw = base64.b64decode(event[key][tag+'_base64'], validate=True)
                if hashlib.sha256(raw).hexdigest() != event[key][tag+'_sha256']:
                    raise ValueError('Raw sensor source hash changed')
            key = (event['sdk']['sequence'], event['native']['packet_id'])
            if key in pairs:
                raise ValueError('Duplicate native/SDK source pair')
            pairs[key] = event
        elif event['kind'] == 'teacher_request':
            requests.append(event)
        elif event['kind'] == 'feedback':
            feedback.append(event)
    if (len(requests) != recorded['decisions'] or len(feedback) != len(requests)
            or [r['index'] for r in requests] != list(range(len(requests)))):
        raise ValueError('Contiguous executed teacher requests and feedback required')
    stop_raw = base64.b64decode(stop['native']['prefix_base64'], validate=True)
    stop_sample = decode_physics_prefix(stop_raw, stop['native']['received_monotonic_seconds'])
    if (hashlib.sha256(stop_raw).hexdigest() != stop['native']['prefix_sha256']
            or stop_sample.packet_id <= max(k[1] for k in pairs)
            or abs(stop_sample.fields['speedKmh']) > .1 or stop_sample.fields['gas'] > .001
            or stop_sample.fields['brake'] < .98):
        raise ValueError('Actual stop prefix does not verify the later stopped state')
    with load_array(root / 'actor-observations.npz', allow_pickle=False) as data:
        original = {key: data[key].copy() for key in data.files}
    if set(original) != {'obs_'+k for k in MultiViewPacker.fields} | {'observation_time'}:
        raise ValueError('Only pose-free actor arrays accepted')
    previous = np.array([0., -1., 1.])
    packed_rows, labels, times, quality = [], [], [], []
    errors = {k: 0. for k in MultiViewPacker.fields}
    policy_error = 0.
    for index, request in enumerate(requests):
        pair = pairs[(request['sdk_sequence'], request['native_packet_id'])]
        sdk = json.loads(base64.b64decode(pair['sdk']['wire_base64']))
        if sdk['producer_session_id'] != producer:
            raise ValueError('Executed teacher sensor producer changed')
        raw_native = pair['native']
        native = decode_physics_prefix(base64.b64decode(raw_native['prefix_base64']),
                                       raw_native['received_monotonic_seconds'])
        meta = request['image_candidate']
        if (meta['source_kind'] != 'csp_four_camera'
                or meta['generation'] != clock.generation
                or meta['packed_observation_index'] != index
                or meta['actor_native_packet_id'] != native.packet_id
                or meta['native_prefix_sha256'] != raw_native['prefix_sha256']
                or set(meta['actor_body_field_names']) != retina.required_native_fields
                or meta['diagnostic_pose_used_by_actor']):
            raise ValueError('Actor pixel/body separation or observation identity changed')
        decision = meta['actor_decision_monotonic_seconds']
        if decision != original['observation_time'][index]:
            raise ValueError('Recorded actor decision time changed')
        camera_path = Path(meta['file'])
        if camera_path.parts[:-1] != ('vision',):
            raise ValueError('Camera metadata path differs')
        camera_bytes = archive.read(camera_path.name)
        if hashlib.sha256(camera_bytes).hexdigest() != meta['file_sha256']:
            raise ValueError('Camera metadata identity differs')
        frame = json.loads(camera_bytes)
        if frame['generation'] != clock.generation or frame['index'] != meta['capture_id']:
            raise ValueError('CSP frame generation differs')
        records = {v['view_id']: v for v in meta['views']}
        samples = {}
        for view in frame['views']:
            name, source = view['view_id'], records[view['view_id']]
            file = Path(source['file'])
            if file.parts[:-1] != ('vision',):
                raise ValueError('Camera image path differs')
            image_bytes = archive.read(file.name)
            if hashlib.sha256(image_bytes).hexdigest() != source['file_sha256']:
                raise ValueError('Source image changed')
            with Image.open(io.BytesIO(image_bytes)) as image:
                pixels = np.array(image.convert('RGB'))
            lower = clock.render_start_lower(view['render_started_clock'], frame['generation'])
            if (hashlib.sha256(pixels.tobytes()).hexdigest() != source['rgb_sha256']
                    or lower != source['render_started_lower']):
                raise ValueError('Source pixel bytes or conservative image time changed')
            samples[name] = dict(rgb=pixels, capture_id=meta['capture_id'],
                generation=meta['generation'], render_started_lower=lower,
                received=meta['received'])
        packed = retina.pack(samples, decision_time=decision,
            native_scalars={k: native.fields.get(k) for k in retina.required_native_fields},
            native_receipt=native.received_monotonic_seconds)
        for key, value in packed.items():
            errors[key] = max(errors[key], float(np.max(np.abs(value-original['obs_'+key][index]))))
        if kind == 'sac':
            delta = teacher.predict(np.array(request['observation'])[None])[
                'deterministic_policy_action'][0]
            expected = relative_request(delta, previous, codec['legacy_control_max_delta'],
                [codec['legacy_control_bounds'][0]]*3, [codec['legacy_control_bounds'][1]]*3,
                product_dtype=codec['legacy_product_dtype'])
            if request['collection_speed_kmh'] is not None:
                expected = collection_speed_request(expected, sdk['speed']*3.6,
                                                      request['collection_speed_kmh'])
        else:
            truth = sdk['privileged_teacher']
            if (truth['actor_input'] is not False or request['teacher_kind'] != kind
                    or truth.get('protocol') != 'flybrain.ac.privileged_teacher.v1'):
                raise ValueError('Explicit geometric teacher-only state required')
            state = {k: truth['raw'][k] for k in ('world_position_x', 'world_position_y')}
            state['speed'] = sdk['speed']
            values = PHYSICS_PREFIX.unpack(native.prefix_bytes)
            offset = 0
            for name, _, count in PHYSICS_LAYOUT:
                if name == 'heading':
                    state['yaw'] = -values[offset]
                offset += count
            state.update(geometry.features(state['world_position_x'], state['world_position_y'],
                                           state['yaw']))
            expected = teacher.request(state)
        policy_error = max(policy_error,
            float(np.max(np.abs(expected-request['legacy_absolute_request']))))
        previous = expected
        canonical = legacy_request_to_normalized_controls(expected)
        actual = request['request']['requested_control']
        applied_codec = np.array([actual[k] for k in ('steering', 'throttle', 'brake')])
        if not np.allclose(applied_codec,
            canonical*np.array([request['steering_request_scale'], 1, 1]), rtol=0, atol=1e-12):
            raise ValueError('Executed request differs from canonical label/steering transform')
        packed_rows.append(packed)
        labels.append(canonical)
        times.append(decision)
        quality.append(dict(index=index, source_sdk_sequence=request['sdk_sequence'],
            source_native_packet_id=request['native_packet_id'],
            image_sha256=hashlib.sha256(json.dumps(
                [v['rgb_sha256'] for v in meta['views']]).encode()).hexdigest(),
            source_camera_capture_id=meta['capture_id'],
            all_views_valid=bool(packed['view_valid'].all()),
            camera_generation=meta['generation']))
    for ping in final['exchanges'][len(initial['exchanges']):]:
        clock.observe(ping['sent'], ping['received'], ping['producer_clock'], final['generation'])
    if clock.record() != final or any(errors.values()) or policy_error > 1e-7:
        raise ValueError('Clock, actor reconstruction or teacher policy replay failed')
    archive.close()
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862812,
        recording_manifest_sha256=sha256(root / 'manifest.json'),
        events_sha256=sha256(events_path), post_stop_sha256=sha256(args.post_stop)))
    dataset = run / 'dataset'
    dataset.mkdir()
    episode = dataset / 'episode-000.npz'
    arrays = {k: np.stack([row[k] for row in packed_rows]) for k in MultiViewPacker.fields}
    np.savez_compressed(episode, **{'obs_'+k: v for k, v in arrays.items()},
        action=np.array(labels, np.float32), action_valid=np.ones((len(labels), 3), bool),
        observation_time=np.array(times, np.float64))
    contract = dict(kind='executed_sac_speed_governed',
        collection_speed_kmh=config['arguments']['collection_speed_kmh'],
        teacher_checkpoint_sha256=codec['checkpoint_sha256'],
        control_codec_sha256=sha256('src/flybrain_drive/ac_teacher_pilot.py'),
        label_semantics='executed canonical requests; steering device scale excluded') \
        if kind == 'sac' else dict(kind='geometric_recovery_fixed_task',
            maximum_speed_kmh=teacher.maximum_speed_kmh,
            reference_sha256=geo['reference_sha256'], track_sha256=geo['track_sha256'],
            kinematic_calibration_sha256=sha256(fit_path),
            teacher_implementation_sha256=sha256('src/flybrain_drive/ac_geometric_teacher.py'),
            interpretation='teacher-only geometric desired canonical control at recorded state; '
                           'execution status recorded separately')
    sealed = dict(schema='flybrain.multiview.teacher.v1',
        source='actual AC CSP four-camera/native-body observations '
               'and executed '+kind+' requests',
        action_profile='normalized_ac_absolute', sensory_calibration=retina.calibration,
        actor_fields=sorted(MultiViewPacker.fields), decisions=len(labels), decision_dt_seconds=.04,
        brain_ticks_per_decision=4, expert_labels_available=True, labels_are_executed_requests=True,
        teacher_label_contract=contract, raw_events_sha256=sha256(events_path),
        native_AC_training_data=True, physical_optics_calibrated=False,
        exposure_synchronization_verified=False, experimental_offline_replay_only=True,
        episodes=[dict(episode_id='csp-teacher-segment', file=episode.name,
                       sha256=sha256(episode), steps=len(labels))])
    write_json(dataset / 'dataset.json', sealed)
    write_json(run / 'sample-quality.json', quality)
    load_episodes(dataset, retina.calibration)
    write_json(run / 'result.json', dict(status='completed', decisions=len(labels),
        exact_actor_replay_max_error=errors, teacher_policy_replay_max_error=policy_error,
        dataset_manifest_sha256=sha256(dataset / 'dataset.json'),
        view_coverage=arrays['view_valid'].any(2).sum(0).tolist(),
        unique_camera_groups=len({q['source_camera_capture_id'] for q in quality}),
        maximum_speed_kmh=max(f['speed_m_s']*3.6 for f in feedback),
        maximum_abs_gap_m=max(abs(f['gap_m']) for f in feedback),
        teacher_label_contract=contract, post_stop_verified=True,
        fly_actor_driving_verified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
