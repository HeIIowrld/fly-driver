"""Offline frozen-graph audit, modality ablations and model-only latency; never controls AC."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import load_episodes


def equal_tensor(first, second):
    if first.layout != second.layout or first.shape != second.shape:
        return False
    if first.layout == torch.sparse_csr:
        return (torch.equal(first.crow_indices(), second.crow_indices())
                and torch.equal(first.col_indices(), second.col_indices())
                and torch.equal(first.values(), second.values()))
    if first.layout == torch.sparse_coo:
        a, b = first.coalesce(), second.coalesce()
        return torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values())
    return torch.equal(first, second)


def hide_visual_sources(observation, indices, maximum_age):
    """Mask declared sources in place; body feedback and remaining sources are untouched."""
    for key in ('view_rgb', 'view_contrast', 'view_valid', 'contrast_valid'):
        observation[key][:, indices] = 0
    observation['view_age'][:, indices] = maximum_age*10


def pixel_timing_ablation(observation, model, mode, first_rgb):
    """Offline counterfactual only: keep validity/body fixed while separating pixels and age."""
    if mode == 'pixels_training_mean':
        if not model.photometric_normalization:
            raise ValueError('Training-mean pixel ablation requires declared photometry')
        observation['view_rgb'].copy_(model.photometric_mean[..., :3])
        observation['view_contrast'].copy_(model.photometric_mean[..., 3:6])
    elif mode == 'pixels_first_frame':
        observation['view_rgb'].copy_(first_rgb)
        observation['view_contrast'].zero_()
    elif mode == 'valid_source_age_zero':
        observed = observation['view_valid'].any(-1)
        observation['view_age'][observed] = 0.
    else:
        raise ValueError('Explicit pixel/timing counterfactual required')


def braking_metrics(predictions, targets, validity, threshold=.05):
    """Report rare supervised braking separately from the mostly-zero brake average."""
    selected = validity[:, 2].astype(bool) & (targets[:, 2] > threshold)
    count = int(selected.sum())
    return dict(label_threshold=threshold, supervised_braking_samples=count,
        brake_rmse=float(np.sqrt(np.mean(
            (predictions[selected, 2]-targets[selected, 2])**2))) if count else None,
        mean_target_brake=float(targets[selected, 2].mean()) if count else None,
        mean_predicted_brake=float(predictions[selected, 2].mean()) if count else None,
        predicted_brake_above_threshold=int((predictions[selected, 2] > threshold).sum()),
        scope='offline teacher-label subset; not actual braking response or safety proof')


def split_evidence(learned, dataset, training_dataset=None):
    """Distinguish fitted replay from a new, explicitly disjoint recorded segment."""
    dataset = Path(dataset)
    expected = learned['training_teacher_manifest_sha256']
    actual = sha256(dataset / 'dataset.json')
    if actual == expected:
        return dict(held_out=False, data_used_for_training=True,
                    evaluation_manifest_sha256=actual)
    if training_dataset is None:
        raise ValueError('Different evaluation data requires --training-dataset for split audit')
    training_dataset = Path(training_dataset)
    if sha256(training_dataset / 'dataset.json') != expected:
        raise ValueError('Training dataset does not match checkpoint provenance')
    manifests = [json.loads((p / 'dataset.json').read_text(encoding='utf8'))
                 for p in (training_dataset, dataset)]
    episode_hashes = [{row['sha256'] for row in m['episodes']} for m in manifests]
    def leaf_sources(root, seen=None):
        seen = set() if seen is None else seen
        root = root.resolve()
        if root in seen:
            raise ValueError('Cyclic or duplicate dataset source provenance')
        seen.add(root)
        manifest = json.loads((root / 'dataset.json').read_text(encoding='utf8'))
        if not manifest.get('sources'):
            return [root]
        leaves = []
        for record in manifest['sources']:
            source = Path(record['directory'])
            if sha256(source / 'dataset.json') != record['sha256']:
                raise ValueError('Merged dataset source manifest changed')
            leaves.extend(leaf_sources(source, seen))
        return leaves

    leaves = [leaf_sources(p) for p in (training_dataset, dataset)]
    qualities = [[row for p in sources for row in json.loads(
        (p.parent / 'sample-quality.json').read_text(encoding='utf8'))] for sources in leaves]
    raw_hashes = [{json.loads((p / 'dataset.json').read_text(encoding='utf8'))['raw_events_sha256']
                   for p in sources} for sources in leaves]
    identities = [{(r['source_sdk_sequence'], r['source_native_packet_id']) for r in rows}
                  for rows in qualities]
    images = [{r['image_sha256'] for r in rows} for rows in qualities]
    if (not all(identities) or not all(images)
            or episode_hashes[0] & episode_hashes[1]
            or identities[0] & identities[1] or images[0] & images[1]
            or raw_hashes[0] & raw_hashes[1]):
        raise ValueError('Evaluation recording overlaps training recording')
    return dict(held_out=True, data_used_for_training=False,
        scope='recorded segment held out; same car/track/session, not unseen circuit',
        training_manifest_sha256=expected, evaluation_manifest_sha256=actual,
        shared_episode_files=0, shared_sensor_identities=0, shared_image_files=0,
        sample_quality_sha256=[[sha256(p.parent / 'sample-quality.json') for p in sources]
                               for sources in leaves])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--training-dataset',
                        help='Original sealed dataset, required for a held-out segment audit')
    parser.add_argument('--organ-detail', action='store_true',
                        help='Also isolate organ values from learned validity-flag drive')
    parser.add_argument('--development-set', action='store_true',
                        help='Declare that these labels have informed design or tuning decisions')
    parser.add_argument('--wheel-detail', action='store_true',
                        help='Ablate explicitly registered native wheel slots without other organs')
    parser.add_argument('--view-detail', action='store_true',
                        help='CSP front-only and each single missing-camera deprivation audit')
    parser.add_argument('--yaw-detail', action='store_true',
                        help='Ablate only the explicitly registered native yaw pair')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pixel-timing-detail', action='store_true',
                        help='Counterfactual pixels versus source age; no live driving')
    args = parser.parse_args()
    training = Path(args.training_run)
    report = json.loads((training / 'result.json').read_text())
    checkpoint = training / 'checkpoint.pt'
    if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified training checkpoint required')
    learned = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = learned['training_configuration']
    if sha256(config['source_checkpoint']) != config['source_checkpoint_sha256']:
        raise ValueError('Initialization checkpoint changed')
    initial = torch.load(config['source_checkpoint'], map_location='cpu', weights_only=True)
    split = split_evidence(learned, args.dataset, args.training_dataset)
    split.update(labels_used_for_design_or_tuning=True if args.development_set else None,
                 evaluation_role='development' if args.development_set else
                     'gradient_disjoint_segment' if split['held_out'] else 'training_replay',
                 independent_final_test=False)
    frozen = {name: equal_tensor(value, learned['model'][name])
              for name, value in initial['model'].items() if name.startswith('circuit.')}
    if not frozen or not all(frozen.values()) or initial['graph'] != learned['graph']:
        raise ValueError('Recurrent structure or fixed parameters changed')
    changed = [name for name, value in initial['model'].items()
               if not equal_tensor(value, learned['model'][name])]
    restricted = config.get('organ_group_subset') is not None
    if restricted and set(changed)-set(learned['trainable_parameters']):
        raise ValueError('Restricted organ training changed undeclared model tensors')
    run = begin_run(args.output, dict(arguments=vars(args), checkpoint_sha256=sha256(checkpoint),
        scope=__doc__, dataset_split=split, actuator_messages_sent=0))
    torch.set_num_threads(1)
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in learned['graph'].items()}
    model = MultiViewCircuitController(graph, **learned['constructor'])
    model.load_state_dict(learned['model'])
    model.eval().requires_grad_(False)
    episodes, evaluation_manifest = load_episodes(args.dataset, model.sensory_calibration)
    if learned.get('teacher_label_contract') is not None and (
            learned['teacher_label_contract'] != evaluation_manifest.get('teacher_label_contract')):
        raise ValueError('Evaluation labels must match the learned teacher task contract')
    if split['held_out']:
        load_episodes(args.training_dataset, model.sensory_calibration)
    modes, normal_predictions = {}, []
    mode_names = ['normal', 'vision_missing', 'body_missing', 'both_missing']
    pixel_timing_modes = ['pixels_training_mean', 'pixels_first_frame', 'valid_source_age_zero']
    if args.pixel_timing_detail:
        mode_names.extend(pixel_timing_modes)
    if args.wheel_detail:
        if not learned.get('native_wheel_registration'):
            raise ValueError('Wheel ablation requires explicitly registered wheel input semantics')
        mode_names.append('wheel_missing')
    if args.organ_detail:
        mode_names.extend(['organ_values_zero', 'organ_validity_drive_zero'])
    if args.view_detail:
        if (not learned.get('csp_input_registration')
                or model.sensory_calibration['view_order'] != ['front', 'left', 'right', 'rear']):
            raise ValueError('Independent CSP camera registration required for per-view audit')
        mode_names.extend(['front_only', 'front_missing', 'left_missing',
                           'right_missing', 'rear_missing'])
    if args.yaw_detail:
        if not learned.get('native_yaw_registration'):
            raise ValueError('Yaw ablation requires explicit native yaw registration')
        mode_names.append('yaw_missing')
    organ_weights = [p.weight.detach().clone() for p in model.adapter.organ_adapters]
    with torch.no_grad():
        for mode in mode_names:
            for layer, original in zip(model.adapter.organ_adapters, organ_weights, strict=True):
                layer.weight.copy_(original)
                if mode == 'organ_validity_drive_zero':
                    # Keep validity gates; suppress only the learned constant mask input.
                    layer.weight[:, layer.in_features // 2:].zero_()
            squared, count, timings, preds, labels = np.zeros(3), np.zeros(3), [], [], []
            validities = []
            for episode in episodes:
                state = model.initial_state()
                for index in range(len(episode['action'])):
                    observation = {k: v[index:index+1].clone()
                                   for k, v in episode['observation'].items()}
                    if mode in pixel_timing_modes:
                        pixel_timing_ablation(observation, model, mode,
                                              episode['observation']['view_rgb'][0:1])
                    if mode in ('vision_missing', 'both_missing'):
                        for key in ('view_rgb', 'view_contrast', 'view_valid', 'contrast_valid'):
                            observation[key].zero_()
                        observation['view_age'].fill_(model.sensory_calibration['max_view_age']*10)
                    if mode == 'front_only':
                        hide_visual_sources(observation, [1, 2, 3],
                                            model.sensory_calibration['max_view_age'])
                    elif args.view_detail and mode in (
                            'front_missing', 'left_missing', 'right_missing', 'rear_missing'):
                        source = model.sensory_calibration['view_order'].index(
                            mode.removesuffix('_missing'))
                        hide_visual_sources(observation, [source],
                                            model.sensory_calibration['max_view_age'])
                    if mode in ('body_missing', 'both_missing'):
                        observation['organs'].zero_()
                        observation['organ_valid'].zero_()
                    if mode == 'wheel_missing':
                        observation['organs'][:, :18].zero_()
                        observation['organ_valid'][:, :18].zero_()
                    if mode == 'yaw_missing':
                        observation['organs'][:, 126:128].zero_()
                        observation['organ_valid'][:, 126:128].zero_()
                    if mode == 'organ_values_zero':
                        observation['organs'].zero_()
                    started = time.perf_counter()
                    drive = model.prepare_drive(observation)
                    for _ in range(config['brain_ticks_per_decision']):
                        prediction, state = model.advance(drive, state)
                    timings.append(time.perf_counter()-started)
                    value = prediction[0].numpy()
                    target = episode['action'][index].numpy()
                    valid = episode['action_valid'][index].numpy()
                    squared += (value-target)**2 * valid
                    count += valid
                    preds.append(value.copy())
                    labels.append(target)
                    validities.append(valid)
            predictions, targets = np.array(preds), np.array(labels)
            if mode == 'normal':
                normal_predictions = predictions
            modes[mode] = dict(rmse=np.sqrt(squared/count).tolist(),
                zero_output_baseline_rmse=np.sqrt(np.mean(targets**2, axis=0)).tolist(),
                prediction_mean=predictions.mean(0).tolist(), target_mean=targets.mean(0).tolist(),
                mean_prediction_delta_from_normal=np.abs(
                    predictions-normal_predictions).mean(0).tolist(),
                model_only_seconds_percentiles=np.percentile(timings, [50, 95, 99]).tolist(),
                braking_label_subset=braking_metrics(
                    predictions, targets, np.array(validities)),
                samples=len(predictions), held_out=split['held_out'])
    write_json(run / 'result.json', dict(status='completed', modes=modes, dataset_split=split,
        teacher_label_contract=evaluation_manifest.get('teacher_label_contract'),
        evaluation_labels_executed=evaluation_manifest.get('labels_are_executed_requests'),
        frozen_circuit_tensors=len(frozen), recurrent_parameters_and_graph_unchanged=True,
        changed_model_tensors=changed, restricted_organ_training=restricted,
        undeclared_model_tensors_unchanged=True if restricted else None,
        graph_nodes=len(graph['ids']), graph_edges=len(graph['pre']),
        all_input_fields_pose_free=True, experimental_pixel_host_pairing=True,
        actual_AC_closed_loop_student_evaluated=False, multiview_ready=False,
        model_only_latency_excludes_capture_preprocessing_transport=True,
        actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
