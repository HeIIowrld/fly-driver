"""Fixed-checkpoint development information removals at every scene feature entry."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.train_flight_initial import substeps
from scripts.train_flight_visual_boundary import evaluate_visual


def retained_scene_mask(intervention):
    mask = np.ones((4, 9, 16), dtype=bool)
    if intervention == 'all_motion_information':
        mask[:, 1:] = False
    elif intervention == 'all_intensity_information':
        mask[:, :1] = False
    elif intervention.endswith('_information'):
        view = intervention.removesuffix('_information')
        mask[('front', 'left', 'right', 'rear').index(view)] = False
    else:
        raise ValueError('Explicit scene information intervention required')
    return mask.reshape(-1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root = Path(plan['training'])
    receipt, manifest = read(root/'result.json'), read(root/'manifest.json')['config']
    assert receipt['status'] == 'completed_guarded_optimizer_experiment'
    assert sha256(receipt['checkpoint']) == receipt['checkpoint_sha256']
    data_root = Path(manifest['plan']['data'])
    assert sha256(data_root/'dataset.json') == manifest['dataset_manifest_sha256']
    assert sha256(manifest['plan']['spec']) == manifest['spec_sha256']
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(1)
    run = begin_run(args.output, dict(scope=__doc__, arguments=vars(args),
        plan_sha256=sha256(args.plan), script_sha256=sha256(__file__),
        checkpoint_sha256=receipt['checkpoint_sha256'], optimizer_updates=0,
        original_body_goals_labels_clocks_preserved=True))
    payload = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
    model = VisualFlightCircuit(manifest['plan']['spec'],
        glutamate_sign=payload['glutamate_sign']).to('cuda')
    model.load_state_dict(payload['model'], strict=True)
    model.requires_grad_(False)
    episodes = []
    for entry in read(data_root/'dataset.json')['episodes']:
        if entry['role'] != 'development':
            continue
        assert sha256(data_root/entry['file']) == entry['sha256']
        with load_array(data_root/entry['file'], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        data = {k: torch.tensor(v, device='cuda') for k, v in arrays.items()
                if v.dtype.kind in 'fbiu' and k != 'observation_time'}
        schedule = [[]]+[substeps(float(dt)) for dt in np.diff(arrays['observation_time'])]
        episodes.append((entry, data, schedule, arrays))
    assert len(episodes) == 2
    reports = {}
    for intervention in plan['interventions']:
        modified, lesion = episodes, None
        if intervention in ('scene_command', 'vision'):
            lesion = intervention
        elif intervention != 'intact':
            mask = torch.tensor(retained_scene_mask(intervention), device='cuda')
            modified = []
            for entry, data, schedule, arrays in episodes:
                new_data = dict(data)
                new_data['scene_features'] = torch.where(
                    mask, data['scene_features'], model.scene_command.mean)
                modified.append((entry, new_data, schedule, arrays))
        reports[intervention] = evaluate_visual(model, modified, lesion=lesion)[0]
        write_json(run/'progress.json', dict(completed=list(reports)))
    checks = [dict(file=entry['file'],
        full=reports['intact'][i]['rmse'][0] < reports['all_motion_information'][i]['rmse'][0],
        after3s=reports['intact'][i]['late_turn_rmse']
        < reports['all_motion_information'][i]['late_turn_rmse'])
        for i, (entry, _, _, _) in enumerate(episodes)]
    write_json(run/'result.json', dict(status='completed_development_information_evaluation',
        metrics=reports, motion_checks=checks,
        motion_information_gate_passed=all(c['full'] and c['after3s'] for c in checks),
        whole_input_information_removal_not_cell_lesion=True, original_history_resets_added=0,
        evaluation_role_arrays_opened=False, optimizer_updates=0, checkpoint_modified=False,
        actor_qualified=False, physiological_fusion_verified=False, goal_achieved=False))


if __name__ == '__main__':
    main()
