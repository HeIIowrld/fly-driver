"""Select regularized joint-eye readout training duration using whole TRAIN capture groups."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.train_crossvalidated_joint_ridge import group
from scripts.train_joint_view_expanded_histories import prepare, read, score


def make_features(records):
    x = np.concatenate([r['x'] for r in records])
    y = np.concatenate([r['y'] for r in records])
    valid = np.concatenate([r['valid'] for r in records])
    return (x.mean(0), np.maximum(x.std(0), 1e-5),
            float(y[valid].mean()), max(float(y[valid].std()), 1e-4), x, y, valid)


def normalized(record, mean, scale, device):
    return torch.tensor(np.clip((record['x']-mean)/scale, -5, 5),
                        dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--older-dataset', required=True)
    parser.add_argument('--older-readout', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError('Original source plan changed')
    plan = read(args.plan)
    entries = read(args.older_dataset)['episodes'] + read(plan['dataset'])['episodes']
    source_list = read(args.older_readout)['episodes'] + read(plan['sensor_replay'])['episodes']
    sources = {e['id']: e for e in source_list}
    if (len(sources) != len(entries) or len({e['id'] for e in entries}) != len(entries)
            or len({str(Path(sources[e['id']]['recording']).resolve()) for e in entries})
            != len(entries) or any(e['role'] != sources[e['id']]['role'] for e in entries)):
        raise ValueError('Unique original recordings and roles required')
    train = [e for e in entries if e['role'] == 'train']
    dev = [e for e in entries if e['role'] == 'development']
    if len(train) != 11 or len(dev) != 3 or not torch.cuda.is_available():
        raise ValueError('Expected 11 TRAIN, 3 DEV and available GPU')
    torch.set_num_threads(1)
    device = torch.device('cuda')
    candidates = [10, 50, 150]
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'source_plan_sha256': args.plan_sha256, 'older_dataset_sha256': sha256(args.older_dataset),
        'older_readout_sha256': sha256(args.older_readout), 'candidate_updates': candidates,
        'folds': ['original', 'spatial', 'exterior'], 'optimizer': 'AdamW lr .001 decay .01',
        'development_read_before_checkpoint': False, 'actor_deployed': False,
        'device': torch.cuda.get_device_name(0)})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    records = []
    for e in train:
        records.append(prepare(e, sources[e['id']]))
        print(json.dumps({'loaded_train': e['id']}), flush=True)
    for e in records[1:]:
        np.testing.assert_array_equal(e['uv'], records[0]['uv'])

    def fit(fitting, held, updates, *, capture_candidates):
        mean, scale, target_mean, target_scale, x, y, valid = make_features(fitting)
        xt = torch.tensor(np.clip((x-mean)/scale, -5, 5), dtype=torch.float32, device=device)
        yt = torch.tensor((y-target_mean)/target_scale, dtype=torch.float32, device=device)
        mask = torch.tensor(valid, device=device)
        torch.manual_seed(863692)
        model = nn.Sequential(nn.Linear(896, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        held_x = [normalized(e, mean, scale, device) for e in held]
        reports = []
        for step in range(1, updates+1):
            opt.zero_grad(set_to_none=True)
            loss = (model(xt).squeeze(1)[mask]-yt[mask]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Finite training loss required')
            loss.backward()
            opt.step()
            if capture_candidates and step in candidates:
                with torch.no_grad():
                    measures = [dict(id=e['id'], **score(e,
                        model(v).squeeze(1).cpu().numpy()*target_scale+target_mean, target_mean))
                        for e, v in zip(held, held_x, strict=True)]
                reports.append({'updates': step, 'reports': measures})
        return model, mean, scale, target_mean, target_scale, reports

    folds = []
    for held_group in ['original', 'spatial', 'exterior']:
        fitting = [e for e in records if group(e) != held_group]
        held = [e for e in records if group(e) == held_group]
        _, _, _, _, _, report = fit(fitting, held, max(candidates), capture_candidates=True)
        folds.append({'held_group': held_group, 'fit_ids': [e['id'] for e in fitting],
                      'candidate_reports': report})
        print(json.dumps({'completed_fold': held_group}), flush=True)
    losses = {str(step): float(np.mean([r['rmse_per_m'] for fold in folds
        for c in fold['candidate_reports'] if c['updates'] == step for r in c['reports']]))
        for step in candidates}
    chosen = min(candidates, key=lambda step: (losses[str(step)], step))
    model, mean, scale, target_mean, target_scale, _ = fit(
        records, [], chosen, capture_candidates=False)
    checkpoint = run/'joint-view-readout.pt'
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'feature_mean': mean, 'feature_scale': scale, 'target_mean': target_mean,
        'target_scale': target_scale, 'train_ids': [e['id'] for e in records],
        'updates': chosen, 'actor_qualified': False}, checkpoint)
    print(json.dumps({'checkpoint_fixed': True, 'chosen_updates': chosen, 'fold_rmse': losses}),
          flush=True)
    development = [prepare(e, sources[e['id']], records[0]['uv']) for e in dev]
    reports = []
    for e in development:
        with torch.no_grad():
            p = model(normalized(e, mean, scale, device)).squeeze(1).cpu().numpy()
        reports.append(dict(id=e['id'], role=e['role'],
                            **score(e, p*target_scale+target_mean, target_mean)))
    for e in [*records, *development]:
        if any(sha256(path) != digest for path, digest in e['pins'].items()):
            raise ValueError('Source changed during experiment')
    write_json(run/'result.json', {'status': 'completed_train_group_selected_joint_readout',
        'folds': folds, 'candidate_updates': candidates, 'train_fold_mean_rmse': losses,
        'chosen_updates': chosen, 'development_reports': reports,
        'checkpoint_sha256': sha256(checkpoint), 'author_optimizer_updates': 0,
        'development_read_before_checkpoint': False, 'known_development_not_final_test': True,
        'teacher_target': '20m teacher path chord curvature, not optimal line or corner distance',
        'body_or_map_direct_input': False, 'actor_deployed': False, 'goal_achieved': False})
    print(json.dumps({'chosen_updates': chosen, 'development_reports': reports}), flush=True)


if __name__ == '__main__':
    main()
