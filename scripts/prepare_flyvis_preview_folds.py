"""CPU TRAIN-history folds with fold-only channel calibration and preview readouts.

Original author responses are references, not copies. Known DEVELOPMENT arrays,
adapted author weights, and existing all-TRAIN calibrations/readouts are never
opened. Each held-out history retains external TRAIN role and all observations.
This auxiliary setup is not author-network adaptation or a vehicle policy.
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.body_retinal_plane import (
    fuse_scalar_views,
    plane_intersections,
    pool_on_plane,
)
from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.retinotopic_feature_scaling import ChannelMoments
from flybrain_drive.turn_residual import current_turn_prior
from scripts.build_body_registered_preview import centres
from scripts.build_retinotopic_preview_dataset import read
from scripts.diagnose_registered_anticipation import anticipation_groups
from scripts.train_body_registered_preview import RegisteredPreview
from scripts.train_visual_body_preview import body_features, body_moments, direct_body_reference

TRAIN_IDS = ['episode-04-train', 'episode-05-train', 'episode-06-train']
CHANNELS = ['L1', 'L2', 'Mi1', 'Tm3', 'Tm1', 'Tm9']


def fingerprint(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()
                          ).hexdigest()


def selected_histories(episodes, fit_ids):
    if len(fit_ids) != 2 or len(set(fit_ids)) != 2 or not set(fit_ids) < set(TRAIN_IDS):
        raise ValueError('Exactly two distinct predefined TRAIN histories must fit each fold')
    selected = [e for e in episodes if e['id'] in fit_ids]
    if [e['id'] for e in selected] != fit_ids or any(e['role'] != 'train' for e in selected):
        raise ValueError('Ordered whole TRAIN histories required; no role conversion')
    return selected


def fold_statistics(episodes, fit_ids, channel_floor, target_floor):
    fit = selected_histories(episodes, fit_ids)
    channel = ChannelMoments()
    for e in fit:
        channel.add(e['activity'], role=e['role'])
    cm, cs = channel.finish(channel_floor)
    bm, bs, counts = body_moments(fit)
    y, valid = (np.concatenate([e[key] for e in fit]) for key in ('y', 'valid'))
    if not valid.sum(0).all():
        raise ValueError('Every original target head needs fit-history support')
    ym = np.array([y[valid[:, h], h].mean() for h in range(6)])
    ys = np.maximum([y[valid[:, h], h].std() for h in range(6)], target_floor)
    return dict(channel_mean=cm, channel_scale=cs, body_mean=bm, body_scale=bs,
        body_valid_counts=counts, target_mean=ym, target_scale=ys,
        target_valid_counts=valid.sum(0), channel_count=channel.count)


def registered_cpu(e, channel_mean, channel_scale, *, batch_size=8):
    """Same registered transform as baseline, including unaltered support masks."""
    xy, support = plane_intersections(e['rays'], e['origins'], e['body'][:, 0],
        e['body'][:, 1], e['body_valid'][:, 0] & e['body_valid'][:, 1])
    scalar_centres = torch.tensor(centres(8, 8), dtype=torch.float32)
    flow_centres = torch.tensor(centres(4, 4), dtype=torch.float32)
    mean = torch.tensor(channel_mean, dtype=torch.float32)[None, None, :, None]
    scale = torch.tensor(channel_scale, dtype=torch.float32)[None, None, :, None]
    outputs = {key: [] for key in ['intact', 'front_mean', 'left_mean', 'right_mean', 'rear_mean']}
    with torch.no_grad():
        for start in range(0, len(e['clock']), batch_size):
            end = min(start+batch_size, len(e['clock']))
            activity = torch.tanh((torch.tensor(e['activity'][start:end])-mean)/scale)
            coords = torch.tensor(xy[start:end], dtype=torch.float32)
            mask = torch.tensor(support[start:end])
            scalar, scalar_mask = pool_on_plane(activity[:, :, :6], coords, mask,
                scalar_centres, torch.tensor([4., 2.]))
            flow, flow_mask = pool_on_plane(activity[:, :, 6:], coords, mask,
                flow_centres, torch.tensor([8., 4.]))
            fused, fused_mask = fuse_scalar_views(scalar, scalar_mask)
            flags = torch.cat([fused_mask, flow_mask.flatten(1)], 1).float()

            def packed(sc, fl, flags=flags):
                return torch.cat([sc.flatten(1), fl.flatten(1), flags], 1).numpy()

            outputs['intact'].append(packed(fused, flow))
            count = scalar_mask.sum(1).clamp_min(1)[:, None]
            for v, view in enumerate(('front', 'left', 'right', 'rear')):
                changed = fused-scalar[:, v]*scalar_mask[:, v, None]/count
                changed_flow = flow.clone()
                changed_flow[:, v] = 0
                outputs[view+'_mean'].append(packed(changed, changed_flow))
    return {key: np.concatenate(values) for key, values in outputs.items()}


def fit_feature_moments(features, fit_ids):
    x = np.concatenate([features[key]['intact'] for key in fit_ids])
    mean, scale = x.mean(0, dtype=float), x.astype(float).std(0).clip(1e-5)
    mean[896:], scale[896:] = 0., 1.
    return mean, scale


def score(e, predictions, target_scale):
    y, valid = e['y'], e['valid']
    bearing = [0, 2, 4]
    current = np.full(len(y), np.nan)
    available = e['body_valid'][:, 4] & e['body_valid'][:, 16] & (e['body'][:, 16] > 1)
    current[available] = -e['body'][available, 4]/e['body'][available, 16]
    groups = anticipation_groups(current, y[:, 5], available & valid[:, 5], valid[:, 5])
    groups.update(left=valid[:, 5] & (y[:, 5] < -.001),
                  right=valid[:, 5] & (y[:, 5] > .001),
                  near_straight=valid[:, 5] & (abs(y[:, 5]) <= .001), full=valid[:, 5])
    result = {}
    for name, predicted in predictions.items():
        heads = []
        for h in range(6):
            keep = valid[:, h] & np.isfinite(predicted[:, h])
            heads.append(dict(count=int(keep.sum()), rmse=float(np.sqrt(np.mean(
                (y[keep, h]-predicted[keep, h])**2))) if keep.any() else None))
        standardized = [np.mean(((predicted[valid[:, h], h]-y[valid[:, h], h])/
            target_scale[h])**2) for h in range(6)]
        diagnostic = {}
        for group, mask in groups.items():
            keep = mask & valid[:, 5] & np.isfinite(predicted[:, 5])
            diagnostic[group] = dict(rows=int(mask.sum()), scored_rows=int(keep.sum()),
                curvature20m_rmse=float(np.sqrt(np.mean((predicted[keep, 5]-y[keep, 5])**2)))
                if keep.any() else None)
        result[name] = dict(standardized_mse=float(np.mean(standardized)), heads=heads,
            bearing_head_indices=bearing, groups=diagnostic)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    if (plan['train_ids'] != TRAIN_IDS or plan['epochs'] != 200
            or plan['body_route_indices'] != [0, 1, 4, 16]
            or plan['device'] != 'cpu' or plan['conditions'] != ['vision_body', 'body_only']):
        raise ValueError('Fixed CPU three-whole-history, two-condition fold setup required')
    for path, digest in plan['input_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Declared input changed: '+path)
    source = read(Path(plan['source_data'])/'dataset.json')
    references = read(Path(plan['sensor_replay'])/'result.json')
    if references['author_weights_changed'] is not False:
        raise ValueError('Original author response source required')
    entries = [e for e in source['episodes'] if e['role'] == 'train']
    if [e['id'] for e in entries] != TRAIN_IDS:
        raise ValueError('Only exact original TRAIN3 may be opened')
    torch.set_num_threads(1)
    torch.set_default_device('cpu')
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), external_roles_unchanged=True,
        known_development_arrays_opened=0, original_payloads_copied=0,
        author_optimizer_updates=0, device='cpu'))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    episodes, records, updates = [], [], 0
    result = dict(status='failed', known_development_arrays_opened=0,
        original_payloads_copied=0, author_optimizer_updates=0, gpu_used=False,
        actor_deployed=False, independent_final_validation=False, goal_achieved=False)
    try:
        for entry in entries:
            original = next(e for e in references['episodes'] if e['id'] == entry['id'])
            vision = Path(original['output'])
            for file, digest in ((vision/'responses.npz', original['responses_sha256']),
                                 (entry['targets_file'], entry['targets_sha256'])):
                if sha256(file) != digest:
                    raise ValueError('Original whole TRAIN source changed')
            proof = read(vision/'result.json')
            provenance = read(vision/'manifest.json')['config']
            if (proof['weights_changed'] or proof['optimizer_updates'] != 0
                    or provenance['checkpoint_sha256'] != (
                        plan['original_author_checkpoint_sha256'])):
                raise ValueError('Response source is not the fixed original author network')
            with load_array(vision/'responses.npz', allow_pickle=False) as a:
                clock = a['decision_monotonic_seconds'].copy()
                if a['additional_cell_types'].tolist() != CHANNELS:
                    raise ValueError('Explicit original neuronal channel order required')
                activity = np.concatenate([a['additional_activity'], a['motion_activity']], axis=2)
            with load_array(entry['targets_file'], allow_pickle=False) as a:
                np.testing.assert_array_equal(clock, a['observation_time'])
                y, valid = a['preview'].reshape(-1, 6).copy(), np.repeat(a['preview_valid'], 2, 1)
            body, mask, body_path = direct_body_reference(
                entry, clock, sha256(plan['body_contract']))
            with load_array(vision/'retina-registration.npz', allow_pickle=False) as a:
                if a['view_order'].tolist() != ['front', 'left', 'right', 'rear']:
                    raise ValueError('Original four camera identities required')
                rays, origins = a['body_directions'].copy(), a['body_origins_m'].copy()
            if len(clock) != entry['observations'] or len(clock) != 739:
                raise ValueError('Every original observation must be retained')
            episodes.append(dict(entry, activity=activity, clock=clock, y=y, valid=valid,
                body=body, body_valid=mask, rays=rays, origins=origins,
                source_responses=str(vision/'responses.npz'), source_body=str(body_path),
                source_responses_sha256=original['responses_sha256']))
        for heldout in TRAIN_IDS:
            wall = time.perf_counter()
            fit_ids = [key for key in TRAIN_IDS if key != heldout]
            statistics = fold_statistics(episodes, fit_ids, plan['channel_scale_floor'],
                                         plan['target_scale_floor'])
            # Holdout values cannot influence any fitted moment. Perturbing all
            # its numeric/label arrays gives an exact executable leakage check.
            altered = [dict(e) for e in episodes]
            changed = next(e for e in altered if e['id'] == heldout)
            for key in ('activity', 'body', 'y'):
                changed[key] = np.zeros_like(changed[key])
            other = fold_statistics(altered, fit_ids, plan['channel_scale_floor'],
                                     plan['target_scale_floor'])
            for key, value in statistics.items():
                np.testing.assert_array_equal(value, other[key])
            folder = run/heldout
            folder.mkdir()
            features = {e['id']: registered_cpu(e, statistics['channel_mean'],
                statistics['channel_scale'], batch_size=plan['pool_batch_size']) for e in episodes}
            fm, fs = fit_feature_moments(features, fit_ids)
            statistics.update(feature_mean=fm, feature_scale=fs)
            fit = selected_histories(episodes, fit_ids)
            y, valid = (np.concatenate([e[key] for e in fit]) for key in ('y', 'valid'))
            target = torch.tensor((y-statistics['target_mean'])/statistics['target_scale'],
                                  dtype=torch.float32)
            weight = torch.tensor(valid/(6*valid.sum(0)), dtype=torch.float32)
            body_inputs = {e['id']: body_features(e['body'], e['body_valid'],
                statistics['body_mean'], statistics['body_scale'], plan['body_route_indices'])
                for e in episodes}
            tb = torch.tensor(np.concatenate([body_inputs[key] for key in fit_ids]))
            normalized = {key: ((values['intact']-fm)/fs).clip(-5, 5).astype(np.float32)
                          for key, values in features.items()}
            neural_inputs = np.concatenate([normalized[key] for key in fit_ids])
            holdout = next(e for e in episodes if e['id'] == heldout)
            predictions, learning, initial_hashes = {}, {}, []
            for condition in plan['conditions']:
                torch.manual_seed(plan['seed'])
                model = RegisteredPreview().cpu()
                initial_hashes.append(fingerprint(np.concatenate([
                    p.detach().numpy().ravel() for p in model.parameters()])))
                tx = torch.tensor(neural_inputs if condition == 'vision_body'
                                  else np.zeros_like(neural_inputs))
                optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                              weight_decay=plan['weight_decay'])
                losses = []
                for _epoch in range(plan['epochs']):
                    optimizer.zero_grad(set_to_none=True)
                    loss = ((model(tx, tb)-target).square()*weight).sum()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite fold readout objective')
                    loss.backward()
                    if not all(p.grad is not None and torch.isfinite(p.grad).all()
                               for p in model.parameters()):
                        raise ValueError('Nonfinite/missing fold readout gradient')
                    optimizer.step()
                    updates += 1
                    losses.append(float(loss.detach()))
                model.eval()
                holdout_body = torch.tensor(body_inputs[heldout])
                variants = dict(intact=normalized[heldout])
                if condition == 'vision_body':
                    changed = normalized[heldout].copy()
                    changed[:, :896] = 0
                    variants['neural_feature_mean'] = changed
                    for view in ('front', 'left', 'right', 'rear'):
                        variants[view+'_mean'] = ((features[heldout][view+'_mean']-fm)/fs
                                                 ).clip(-5, 5).astype(np.float32)
                    changed = features[heldout]['intact'].copy()
                    changed[:, :896] = 0
                    variants['retinal_neural_mean'] = ((changed-fm)/fs).clip(-5, 5).astype(
                        np.float32)
                with torch.no_grad():
                    for name, values in variants.items():
                        if condition == 'body_only':
                            values = np.zeros_like(values)
                        output = model(torch.tensor(values), holdout_body).numpy()
                        predictions[condition+'__'+name] = (output*statistics['target_scale']+
                                                           statistics['target_mean'])
                learning[condition] = dict(epochs=plan['epochs'], losses=losses,
                    initial_parameter_sha256=initial_hashes[-1])
                torch.save(dict(model=model.state_dict(), plan=plan,
                    fitted_history_ids=fit_ids, heldout_history=heldout, **statistics,
                    body_route_indices=plan['body_route_indices'], condition=condition,
                    actuator_policy=False, original_author_checkpoint_sha256=(
                        plan['original_author_checkpoint_sha256'])), folder/(condition+'.pt'))
            assert initial_hashes[0] == initial_hashes[1]
            predictions['current_turn_prior'], _ = current_turn_prior(
                holdout['body'], holdout['body_valid'])
            predictions['fit_target_mean'] = np.broadcast_to(statistics['target_mean'],
                                                             holdout['y'].shape)
            predictions['zero'] = np.zeros_like(holdout['y'])
            np.savez_compressed(folder/'normalization.npz', **statistics)
            record = dict(heldout_history=heldout, fit_history_ids=fit_ids,
                heldout_external_role='train', fit_observations=sum(len(e['clock']) for e in fit),
                holdout_observations=len(holdout['clock']), all_moments_fit_ids=fit_ids,
                readout_fit_ids=fit_ids, holdout_perturbation_moments_exact=True,
                learning=learning, holdout_metrics=score(holdout, predictions,
                    statistics['target_scale']), wall_seconds=time.perf_counter()-wall,
                source_array_fingerprints={e['id']: {key: fingerprint(e[key]) for key in (
                    'clock', 'valid', 'body', 'body_valid')} for e in episodes},
                models={condition: dict(path=str(folder/(condition+'.pt')),
                    sha256=sha256(folder/(condition+'.pt'))) for condition in plan['conditions']})
            write_json(folder/'result.json', record)
            records.append(record)
            write_json(run/'progress.json', dict(completed_folds=len(records),
                readout_optimizer_updates=updates, author_optimizer_updates=0))
        result.update(status='completed_cpu_train_history_fold_setup', folds=records,
            readout_optimizer_updates=updates,
            all_fold_normalizations_and_readouts_fit_only_two=True,
            source_references=[{k: e[k] for k in ('id', 'role', 'source_responses', 'source_body',
                'source_responses_sha256', 'targets_file', 'targets_sha256')} for e in episodes],
            source_original_observations=2217, whole_history_internal_holdouts=True,
            author_network_temporal_learning_executed=False,
            limitations=['OriginalTRAIN regions are adjacent; not independent final validation.',
                'No physiological, flight-motor, road-driving or lap qualification.',
                'New readout initialization only; original author weights remain unchanged.'])
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), completed_folds=records,
                      readout_optimizer_updates=updates)
        raise
    finally:
        write_json(run/'result.json', result)
    print('completed CPU folds/readout updates', len(records), updates)


if __name__ == '__main__':
    main()
