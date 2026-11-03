"""TRAIN-only whole-history nonlinear preview validation, never a driving actor."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_circuit import FlightScenePool
from flybrain_drive.graph import sha256


class PreviewProbe(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        if variant == 'coarse_dense64':
            self.readout = nn.Sequential(nn.Linear(576, 64), nn.Tanh(), nn.Linear(64, 6))
        elif variant == 'front8_shared8_dense64':
            self.patch = nn.Sequential(nn.Linear(9, 8), nn.Tanh())
            self.readout = nn.Sequential(nn.Linear(512, 64), nn.Tanh(), nn.Linear(64, 6))
        else:
            raise ValueError('Unknown predeclared architecture')

    def forward(self, x):
        if self.variant == 'front8_shared8_dense64':
            x = self.patch(x.reshape(-1, 9, 64).transpose(1, 2)).flatten(1)
        return self.readout(x)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def metrics(pred, y, valid, baseline, clock):
    result = {}
    for name, late in [('full', np.ones(len(y), bool)), ('after3s', clock >= clock[0]+3)]:
        rows = []
        for h in range(3):
            keep = valid[:, h] & late
            p, target = pred[keep, h], y[keep, h]
            rows.append(dict(count=int(keep.sum()),
                rmse=np.sqrt(((p-target)**2).mean(0)).tolist(),
                fit_mean_rmse=np.sqrt(((baseline[h]-target)**2).mean(0)).tolist(),
                zero_rmse=np.sqrt((target**2).mean(0)).tolist()))
        result[name] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    data, preview = Path(plan['data']), Path(plan['preview'])
    dataset, label_manifest = read(data/'dataset.json'), read(preview/'result.json')
    labels = {e['file']: e for e in label_manifest['episodes']}
    entries = [e for e in dataset['episodes'] if e['role'] == 'train']
    assert len(entries) == 6 and len({e['file'] for e in entries}) == 6
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        script_sha256=sha256(__file__), parent_dataset_sha256=sha256(data/'dataset.json'),
        preview_result_sha256=sha256(preview/'result.json'),
        device=torch.cuda.get_device_name(0),
        previous_goal_turn_classification='progress: completed six spatial comparison probes',
        development_arrays_opened=False, evaluation_arrays_opened=False,
        ui_actions_sent=0, actuator_messages_sent=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    (run/'plan.json').write_bytes(Path(args.plan).read_bytes())
    episodes, proofs = [], []
    reference_uv = None
    for entry in entries:
        assert labels[entry['file']]['role'] == 'train'
        assert sha256(data/entry['file']) == entry['sha256']
        assert sha256(preview/entry['file']) == labels[entry['file']]['auxiliary_sha256']
        with load_array(data/entry['file'], allow_pickle=False) as a:
            coarse, clock = a['scene_features'].copy(), a['observation_time'].copy()
        with load_array(preview/entry['file'], allow_pickle=False) as a:
            y, valid = a['preview'].copy(), a['preview_valid'].copy()
            np.testing.assert_array_equal(clock, a['observation_time'])
        visual = Path(entry['original']['scene_replay'])
        assert sha256(visual/'responses.npz') == entry['source_visual_sha256']
        with load_array(visual/'retina-registration.npz', allow_pickle=False) as a:
            uv = a['image_uv'].copy()
        if reference_uv is None:
            reference_uv = uv
            pool = FlightScenePool(uv, plan['pool8']).to('cuda')
        else:
            np.testing.assert_array_equal(reference_uv, uv)
        with load_array(visual/'responses.npz', allow_pickle=False) as a:
            np.testing.assert_array_equal(clock, a['decision_monotonic_seconds'])
            raw = np.concatenate([a['retina_intensity'][:, :, None], a['motion_activity']], 2)
        with torch.no_grad():
            tensor = torch.tensor(raw, device='cuda', dtype=torch.float32)
            fine = pool(tensor).reshape(-1, 4, 9, 64)[:, 0].flatten(1).cpu().numpy()
        episodes.append(dict(file=entry['file'], y=y, valid=valid, clock=clock,
            coarse_dense64=coarse, front8_shared8_dense64=fine))
        proofs.append(dict(file=entry['file'], episode_sha256=entry['sha256'],
            labels_sha256=labels[entry['file']]['auxiliary_sha256'],
            neural_responses_sha256=entry['source_visual_sha256']))
    records = []
    for fold, held in enumerate(episodes):
        fit = [e for e in episodes if e['file'] != held['file']]
        assert len(fit) == 5 and held['file'] not in [e['file'] for e in fit]
        y = np.concatenate([e['y'] for e in fit]).reshape(-1, 6)
        valid = np.concatenate([e['valid'] for e in fit]).repeat(2, axis=1)
        target_mean = np.array([y[valid[:, c], c].mean() for c in range(6)])
        target_scale = np.array([y[valid[:, c], c].std() for c in range(6)])
        target_scale = np.maximum(target_scale, np.tile(plan['target_scale_floor'], 3))
        target = torch.tensor((y-target_mean)/target_scale, dtype=torch.float32, device='cuda')
        mask = torch.tensor(valid, dtype=torch.float32, device='cuda')
        assert bool((mask.sum(0) > 0).all())
        for variant in plan['variants']:
            torch.manual_seed(plan['seed']+fold)
            model = PreviewProbe(variant).to('cuda')
            x = np.concatenate([e[variant] for e in fit]).astype(float)
            mean, scale = x.mean(0), x.std(0).clip(plan['normalization_floor'])
            def normalize(value, mean=mean, scale=scale):
                return torch.tensor(((value-mean)/scale).clip(
                    -plan['feature_clip'], plan['feature_clip']),
                    device='cuda', dtype=torch.float32)
            batch = normalize(x)
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
                                           weight_decay=plan['weight_decay'])
            losses = []
            for _epoch in range(plan['epochs']):
                optimizer.zero_grad(set_to_none=True)
                loss = (((model(batch)-target)**2*mask).sum(0)/mask.sum(0)).mean()
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                held_pred = model(normalize(held[variant])).cpu().numpy()*target_scale+target_mean
                fit_pred = model(batch).cpu().numpy()*target_scale+target_mean
            prefix = f'fold-{fold:02d}-{variant}'
            np.savez_compressed(run/(prefix+'.npz'), prediction=held_pred.reshape(-1, 3, 2),
                target=held['y'], target_valid=held['valid'], observation_time=held['clock'])
            torch.save(dict(state=model.state_dict(), feature_mean=mean, feature_scale=scale,
                target_mean=target_mean, target_scale=target_scale, plan=plan,
                fit_files=[e['file'] for e in fit], held_file=held['file']), run/(prefix+'.pt'))
            fit_scores = []
            offset = 0
            for e in fit:
                n = len(e['y'])
                score = metrics(fit_pred[offset:offset+n].reshape(-1, 3, 2),
                    e['y'], e['valid'], target_mean.reshape(3, 2), e['clock'])
                fit_scores.append(dict(file=e['file'], metrics=score))
                offset += n
            records.append(dict(fold=fold, variant=variant, held_file=held['file'],
                fit_files=[e['file'] for e in fit], epochs=len(losses),
                first_loss=losses[0], last_pre_update_loss=losses[-1], losses=losses,
                fit=fit_scores, held=metrics(held_pred.reshape(-1, 3, 2), held['y'],
                    held['valid'], target_mean.reshape(3, 2), held['clock'])))
            write_json(run/'progress.json', dict(completed=len(records), total=12,
                held_file=held['file'], variant=variant, first_loss=losses[0],
                last_pre_update_loss=losses[-1]))
    write_json(run/'result.json', dict(
        status='completed_train_only_whole_history_preview_validation',
        source_proof=proofs, records=records, development_arrays_opened=False,
        evaluation_arrays_opened=False, actor_deployed=False, physiological_claim=False,
        optimizer_updates=sum(r['epochs'] for r in records), goal_achieved=False))


if __name__ == '__main__':
    main()
