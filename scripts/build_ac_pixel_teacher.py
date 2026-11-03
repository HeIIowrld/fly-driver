"""Seal actual AC request/image/body labels and initialize native pixel-addressed fly inputs.

Experimental host-paired data, not calibrated optics, simulation synchronization or a
successful driving policy. No game, window, actuator, privileged-state actor or reset API.
"""

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_pixel_retina import NativePixelRetina
from flybrain_drive.ac_sac_teacher import legacy_request_to_normalized_controls
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import load_episodes
from flybrain_drive.multiview_sensory import MultiViewPacker

SOURCE = 'runs/multiview-dagger-fast-20261027-t29b/runs/training/checkpoint.pt'
SOURCE_SHA = '14c8b79ac124ddfe9b704480aae8c6501e8e70d4315437bd0cfe707da7235f87'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--retina-config', default='configs/ac-native-pixel-retina.json')
    parser.add_argument('--source-kind', choices=['teacher', 'student'], default='teacher')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.run).resolve()
    config = json.loads(Path(args.retina_config).read_text(encoding='utf8'))
    front = NativePixelRetina(config)
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError('Original structural student checkpoint changed')
    result_path = root / 'result.json'
    result = json.loads(result_path.read_text())
    if result['status'] != 'segment_completed':
        raise ValueError('Completed actual request segment required')
    run = begin_run(args.output, dict(arguments=vars(args), pixel_configuration=config,
        source_checkpoint=SOURCE, source_sha256=SOURCE_SHA,
        recording_result_sha256=sha256(result_path), seed=861101, scope=__doc__))
    write_json(run / 'pixel-calibration.json', config)
    student = args.source_kind == 'student'
    events_path = root / ('events.jsonl' if student else 'observations-and-requests.jsonl')
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    pairs, native_to_pair, requests = {}, {}, []
    for event in events:
        if event['kind'] == 'raw_pair':
            for name, tag in [('sdk', 'wire'), ('native', 'prefix')]:
                raw = base64.b64decode(event[name][tag+'_base64'], validate=True)
                if hashlib.sha256(raw).hexdigest() != event[name][tag+'_sha256']:
                    raise ValueError('Native input recording hash mismatch')
            key = (event['sdk']['sequence'], event['native']['packet_id'])
            if key in pairs:
                raise ValueError('Duplicate raw sensor identity')
            pairs[key] = event
            native_to_pair[key[1]] = key
        elif event['kind'] == ('student_request' if student else 'teacher_request'):
            event = copy.deepcopy(event)
            if student:
                meta = event['actor_input']
                key = native_to_pair[meta['native_packet_id']]
                if meta['native_prefix_sha256'] != pairs[key]['native']['prefix_sha256']:
                    raise ValueError('Learner input does not match its recorded raw native sample')
                event.update(sdk_sequence=key[0], native_packet_id=key[1],
                    image_candidate=meta['image'], recorded_actor_decision=meta[
                        'decision_monotonic_seconds'])
            requests.append(event)
    expected = result['student_requests'] if student else result['decisions']
    if len(requests) != expected or not requests:
        raise ValueError('Exact teacher request count required')
    features, actions, times, quality, cache = [], [], [], [], {}
    for request in requests:
        pair = pairs[(request['sdk_sequence'], request['native_packet_id'])]
        image = request['image_candidate']
        file = (root / image['file']).resolve()
        if not file.is_relative_to(root) or sha256(file) != image['file_sha256']:
            raise ValueError('Image path/hash differs')
        if str(file) not in cache:
            rgb = load_array(file, allow_pickle=False)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != image['rgb_sha256']:
                raise ValueError('Image pixel hash differs')
            cache[str(file)] = rgb
        rgb = cache[str(file)]
        clock = request.get('recorded_actor_decision',
                            request['request']['requested_monotonic_seconds'])
        # Deliberately pass only pose-free body channels; never the SDK teacher namespace.
        fields = {k: pair['native']['fields'].get(k) for k in front.required_native_fields}
        observation = front.pack(rgb, capture_id=image['capture_id'],
            callback_time=image['callback_finished_monotonic_seconds'], decision_time=clock,
            native_scalars=fields,
            native_receipt=pair['native']['received_monotonic_seconds'])
        features.append(observation)
        actions.append(np.array(request['prediction']) if student else
                       legacy_request_to_normalized_controls(request['legacy_absolute_request']))
        times.append(clock)
        quality.append(dict(index=request['index'], capture_id=image['capture_id'],
            image_sha256=image['file_sha256'], source_sdk_sequence=request['sdk_sequence'],
            source_native_packet_id=request['native_packet_id'],
            host_image_age=clock-image['callback_finished_monotonic_seconds'],
            available_views=observation['view_valid'].any(1).tolist()))
    dataset = run / 'dataset'
    dataset.mkdir()
    episode = dataset / 'native-segment.npz'
    arrays = {k: np.stack([row[k] for row in features]) for k in MultiViewPacker.fields}
    np.savez_compressed(episode, **{'obs_'+k: v for k, v in arrays.items()},
        action=np.array(actions, dtype=np.float32), action_valid=np.ones((len(actions), 3), bool),
        observation_time=np.array(times, dtype=np.float64))
    manifest = dict(schema='flybrain.multiview.teacher.v1',
        source='actual AC learner visits; executed learner requests require expert relabeling'
            if student else 'actual AC teacher requests with host-paired pixel/body inputs',
        action_profile='normalized_ac_absolute', actor_fields=sorted(MultiViewPacker.fields),
        sensory_calibration=front.calibration, decisions=len(actions),
        observation_time_domain='host_monotonic_receipt; not simulation or exposure time',
        decision_dt_seconds=.04, brain_ticks_per_decision=4,
        action_semantics='teacher absolute request, not synchronized applied feedback',
        action_provider='actual fly student policy, no expert labels' if student else
            'privileged geometric recovery teacher'
            if any(r.get('teacher_kind') == 'geometric_recovery' for r in requests) else
            'public SAC plus declared collection-speed governor'
            if any(r.get('collection_speed_kmh') is not None for r in requests) else 'public SAC',
        teacher_source_kinds=['fly_student'] if student else
            sorted({r.get('teacher_kind', 'sac') for r in requests}),
        expert_labels_available=not student,
        source_request_kind=args.source_kind,
        raw_events_file=events_path.name,
        collection_speed_targets_kmh=sorted({r['collection_speed_kmh'] for r in requests
                                            if r.get('collection_speed_kmh') is not None}),
        governor_changed_requests=sum(r.get('speed_governor_changed_request', False)
                                      for r in requests),
        labels_are_executed_requests=True,
        unmodified_public_policy_performance_claimed=False,
        native_AC_training_data=True,
        raw_events_sha256=sha256(events_path), physical_optics_calibrated=False,
        exposure_synchronization_verified=False, experimental_offline_replay_only=True,
        episodes=[dict(episode_id='native-segment', file=episode.name,
                       sha256=sha256(episode), steps=len(actions))])
    write_json(dataset / 'dataset.json', manifest)
    write_json(run / 'sample-quality.json', quality)
    load_episodes(dataset, front.calibration)
    source = torch.load(SOURCE, map_location='cpu', weights_only=True)
    constructor = copy.deepcopy(source['constructor'])
    constructor.update(directions=front.local_addresses.tolist(),
        available=np.ones(front.local_addresses.shape[:2]).tolist(),
        sensory_calibration=front.calibration, pedal_mode='independent')
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in source['graph'].items()}
    torch.manual_seed(861101)
    model = MultiViewCircuitController(graph, **constructor)
    state, fresh = copy.deepcopy(source['model']), model.state_dict()
    for key in state:
        if key.startswith('photometric_') or key in (
                'adapter.visual_adapter.indices', 'adapter.visual_adapter.centers',
                'adapter.visual_adapter.port_sources'):
            state[key] = fresh[key]
    model.load_state_dict(state)
    initialized = copy.deepcopy(source)
    initialized.update(constructor=constructor, model=model.state_dict(),
        model_kind='native_pixel_registered_structural_student',
        driving_trained='source MuJoCo steering only; native weights not trained',
        native_input_registration=dict(configuration=config, source_checkpoint_sha256=SOURCE_SHA,
            anatomical_optics_verified=False, native_driving_verified=False))
    torch.save(initialized, run / 'native-initialized.pt')
    train = dict(source_checkpoint=str(run / 'native-initialized.pt'),
        source_checkpoint_sha256=sha256(run / 'native-initialized.pt'), seed=861201,
        action_profile='normalized_ac_absolute', epochs=12, chunk_length=16,
        learning_rate=.0001, brain_ticks_per_decision=4, control_dt_seconds=.04,
        trainable_groups=['visual', 'organs', 'readout'], pedal_mode='independent',
        pedal_conditioning=dict(target_rms=[.2, .2], max_gain=1e10),
        native_steering_conditioning=dict(target_rms=.15, max_gain=1e10),
        condition_steering_gain=False,
        photometric_normalization=dict(rgb_floor=.03, contrast_floor=.1))
    write_json(run / 'training-config.json', train)
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError('Original checkpoint mutated')
    write_json(run / 'result.json', dict(status='completed', decisions=len(actions),
        view_coverage=arrays['view_valid'].any(2).sum(0).tolist(),
        organ_valid_feature_counts=np.unique(arrays['organ_valid'].sum(1)).tolist(),
        graph_nodes=len(source['graph']['ids']), graph_edges=len(source['graph']['pre']),
        original_checkpoint_unchanged=True, actor_fields=sorted(arrays),
        experimental_host_paired_dataset=True, full_multiview_ready=False,
        calibrated_optics=False, AC_driving_verified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
