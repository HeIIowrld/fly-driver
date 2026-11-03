"""Reproduce saved whole-episode predictions, then replace body values without changing validity."""

import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.build_retinotopic_preview_dataset import read
from scripts.train_column_preview_comparison import metrics
from scripts.train_visual_body_preview import FusedPreview, body_features, direct_body_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--training', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan, root = read(a.plan), Path(a.training)
    source = read(root/'manifest.json')['config']
    if (source['plan_sha256'] != sha256(a.plan)
            or read(root/'result.json')['status'] != 'completed_visual_body_preview_comparison'):
        raise ValueError('Original fixed-plan completed training required')
    data = read(Path(plan['visual_data'])/'dataset.json')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Original GPU inference required')
    hashes = {c: sha256(root/(c+'.pt')) for c in plan['conditions']}
    run = begin_run(a.output, dict(arguments=vars(a), program_sha256=sha256(__file__),
        training_result_sha256=sha256(root/'result.json'), checkpoint_sha256=hashes,
        dataset_sha256=sha256(Path(plan['visual_data'])/'dataset.json'), optimizer_updates=0,
        validity_unchanged=True,
        correction='values-only control differs from explicit missing-sensor control'))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    reports, errors, summaries = [], [], []
    for condition in plan['conditions']:
        saved = torch.load(root/(condition+'.pt'), map_location='cuda', weights_only=False)
        if saved['plan'] != plan or saved['condition'] != condition:
            raise ValueError('Checkpoint plan/condition differs')
        model = FusedPreview().to('cuda')
        model.load_state_dict(saved['model'])
        model.eval()
        with load_array(root/(condition+'-predictions.npz'), allow_pickle=False) as previous:
            for entry in data['episodes']:
                if entry['role'] not in ('train', 'development'):
                    raise ValueError('No evaluation access')
                if sha256(entry['features_file']) != entry['features_sha256']:
                    raise ValueError('Original whole episode changed')
                with load_array(entry['features_file'], allow_pickle=False) as arrays:
                    clock, y = arrays['observation_time'], arrays['preview'].reshape(-1, 6)
                    valid = np.repeat(arrays['preview_valid'], 2, axis=1)
                    vx = ((arrays['processed_actual']-saved['vision_mean'])/saved[
                        'vision_scale']).clip(-5, 5).astype(np.float32)
                body, mask, _ = direct_body_reference(entry, clock, sha256(plan['body_contract']))
                bx = body_features(body, mask, saved['body_mean'], saved['body_scale'],
                                   saved['plan'].get('body_route_indices'))
                vx, bx = torch.tensor(vx, device='cuda'), torch.tensor(bx, device='cuda')
                if condition == 'body_only':
                    vx.zero_()
                if condition == 'vision_only':
                    bx.zero_()
                variants = [('intact', bx)]
                for name, indices in plan['body_channel_interventions'].items():
                    changed = bx.clone()
                    changed[:, indices] = 0  # Keep the entire original validity half.
                    variants.append((name.replace('_absent', '_train_mean'), changed))
                for intervention, values in variants:
                    with torch.no_grad():
                        prediction = model(vx, values).cpu().numpy()*saved['target_scale']+saved[
                            'target_mean']
                    if intervention == 'intact':
                        error = float(np.max(np.abs(prediction-previous[entry['id']+'__intact'])))
                        errors.append(dict(condition=condition, episode=entry['id'], error=error))
                        if error > 2e-6:
                            raise ValueError('Original whole-episode predictions do not reproduce')
                    measured = metrics(y, valid, clock, prediction, saved['target_mean'])
                    reports.append(dict(condition=condition, episode=entry['id'],
                        role=entry['role'],
                        intervention=intervention, **measured))
                    if entry['role'] == 'development':
                        summaries.append(dict(condition=condition, intervention=intervention,
                            standardized_mse=float(np.mean([(h['rmse']/s)**2 for h, s in zip(
                                measured['full'], saved['target_scale'], strict=True)]))))
    write_json(run/'result.json', dict(status='completed_body_values_only_controls',
        reproduction=errors, reports=reports, development_summary=summaries,
        validity_unchanged=True, optimizer_updates=0, checkpoint_parameters_changed=False,
        physiological_function_verified=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
