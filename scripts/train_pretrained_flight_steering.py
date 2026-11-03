"""Calibrate descending turn currents through the actual frozen fly motor graph."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.evaluate_embodied_flight_navigation import metrics
from scripts.pretrain_reflected_steering import VisualCommandPretrain
from scripts.train_embodied_flight_navigation import objective_pass
from scripts.train_flight_initial import substeps
from scripts.train_flight_visual_guarded import guarded_step


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    root = base/'20261029-t1434-body21-flight-cache'
    pretrained = base/'20261029-t1472-reflected-steering-pretraining'
    source = read(base/'20261029-t1361-v2-visual-boundary-training/result.json')
    assert read(pretrained/'result.json')['status'] == 'completed_fixed_visual_steering_pretraining'
    # Chosen in advance as the augmentation experiment, never by development scores.
    selected = pretrained/'original_plus_reflected.pt'
    assert sha256(source['checkpoint']) == source['checkpoint_sha256']
    torch.set_num_threads(1)
    torch.manual_seed(863473)
    payload = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
    model = EmbodiedVisualFlightCircuit('configs/flight-circuit-visual-v6-20261029.json',
                                       payload).to('cuda')
    # This is our own locally produced checkpoint; it contains NumPy normalization arrays.
    pre = torch.load(selected, map_location='cpu', weights_only=False)
    assert pre['schema'] == 'flybrain.visual.command_pretrain.v1' and not pre['actuator_policy']
    latent = VisualCommandPretrain().to('cuda')
    latent.load_state_dict(pre['model'], strict=True)
    latent.requires_grad_(False)
    with torch.no_grad():
        model.boundary.mean.copy_(torch.as_tensor(pre['feature_mean'], device='cuda'))
        model.boundary.scale.copy_(torch.as_tensor(pre['feature_scale'], device='cuda'))
        model.boundary.patch.load_state_dict(latent.patch.state_dict())
        model.boundary.turn_hidden.weight.zero_()
        model.boundary.turn_hidden.weight[:, :512].copy_(latent.hidden.weight)
        model.boundary.turn_hidden.bias.copy_(latent.hidden.bias)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.boundary.turn_output.requires_grad_(True)
    parameters = list(model.boundary.turn_output.parameters())
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    run = begin_run(args.output, dict(program_sha256=sha256(__file__),
        data=str(root), dataset_sha256=sha256(root/'dataset.json'),
        source_checkpoint_sha256=source['checkpoint_sha256'],
        visual_pretrain_sha256=sha256(selected), condition='original_plus_reflected',
        epochs=24, learning_rate=.01, seed=863473, trainable='turn_output only',
        preserved='Author visual weights, motor graph/body/readout, power head frozen',
        scope='Original low-speed steering calibration, not racing-speed or live driving',
        runtime_actor_deployed=False, ui_actions_sent=0, actuator_messages_sent=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    episodes, count = [], np.zeros(3, int)
    for entry in read(root/'dataset.json')['episodes']:
        if entry['role'] != 'train':
            continue
        assert sha256(root/entry['file']) == entry['sha256']
        with load_array(root/entry['file'], allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        valid = arrays['target_valid'][:, 0]
        groups = np.where(arrays['target'][:, 0] > .01, 0,
                          np.where(arrays['target'][:, 0] < -.01, 1, 2))
        count += np.bincount(groups[valid], minlength=3)
        data = {k: torch.tensor(arrays[k], device='cuda') for k in
            ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features', 'target')}
        with torch.no_grad():
            baseline = model.backbone._drive_tensor(data['visual'], data['visual_valid'],
                data['body'][:, :16], data['body_valid'][:, :16], data['goals'],
                data['scene_features'])
            drive = model._modulate_cached_drive(baseline, data['body'], data['body_valid'],
                                                 data['goals'], data['scene_features'])
            assert torch.equal(baseline, drive)
        episodes.append(dict(entry=entry, arrays=arrays, data=data, baseline=baseline,
            groups=groups, valid_any=valid,
            schedule=[[]]+[substeps(float(dt)) for dt in np.diff(arrays['observation_time'])]))
    for e in episodes:
        weights = np.zeros_like(e['arrays']['target'], dtype=float)
        weights[:, 0] = e['valid_any']/(3*count[e['groups']])
        e['weights'] = torch.tensor(weights, device='cuda')
    optimizer = torch.optim.Adam(parameters, lr=.01)
    records = []
    for epoch in range(24):
        optimizer.zero_grad(set_to_none=True)
        loss, disconnected = objective_pass(model, episodes, 64, backward=True)
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
        @torch.no_grad()
        def objective():
            return objective_pass(model, episodes, 64)[0]
        step = guarded_step(optimizer, parameters, objective, loss,
                            trials=4, contraction=.5, armijo=.0001)
        assert all(torch.equal(p, frozen[n]) for n, p in model.named_parameters()
                   if n in frozen)
        model.backbone.assert_body_boundary_protected()
        records.append(dict(epoch=epoch+1, disconnected=disconnected, **step))
        torch.save(dict(schema=model.schema, model=model.state_dict(), spec=model.spec,
            goal_contract=payload['goal_contract'], source_checkpoint=source['checkpoint'],
            source_checkpoint_sha256=source['checkpoint_sha256'],
            glutamate_sign=payload['glutamate_sign'], visual_pretrain_sha256=sha256(selected),
            actuator_qualified=False), run/'checkpoint.partial.pt')
        os.replace(run/'checkpoint.partial.pt', run/'checkpoint.pt')
        write_json(run/'progress.json', dict(completed_epochs=epoch+1, epochs=24,
                                            latest=records[-1]))
    write_json(run/'training-result.json', dict(status='completed_fixed_turn_current_training',
        records=records, accepted_updates=sum(r['accepted'] for r in records),
        checkpoint_sha256=sha256(run/'checkpoint.pt'), original_frozen_parameters_unchanged=True))
    # Terminal development report only; no parameter/epoch choices follow these scores.
    all_train = np.concatenate([e['arrays']['target'] for e in episodes])
    all_masks = np.concatenate([e['arrays']['target_valid'] for e in episodes])
    mean = np.array([all_train[all_masks[:, h], h].mean() for h in range(2)])
    reports = []
    for entry in read(root/'dataset.json')['episodes']:
        if entry['role'] != 'development':
            continue
        assert sha256(root/entry['file']) == entry['sha256']
        with load_array(root/entry['file'], allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        data = {k: torch.tensor(arrays[k], device='cuda') for k in
            ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features')}
        late = arrays['target_valid'] & (
            arrays['observation_time'] >= arrays['observation_time'][0]+3)[:, None]
        record = dict(file=entry['file'], conditions={},
            train_mean=metrics(np.tile(mean, (len(late), 1)),
                               arrays['target'], arrays['target_valid']),
            zero=metrics(np.zeros_like(arrays['target']), arrays['target'], arrays['target_valid']))
        for condition in ('intact', 'new_command_removed', 'all_visual_train_mean'):
            scene = data['scene_features'].clone()
            if condition == 'all_visual_train_mean':
                scene[:] = model.boundary.mean
            with torch.no_grad():
                drive = model._drive_tensor(data['visual'], data['visual_valid'], data['body'],
                    data['body_valid'], data['goals'], scene,
                    lesion='new_command' if condition == 'new_command_removed' else None)
                state, held, predictions = model.initial_state(1), None, []
                for index in range(len(scene)):
                    if index:
                        dt = arrays['observation_time'][index]-arrays['observation_time'][index-1]
                        for step in substeps(float(dt)):
                            state = model._advance_elapsed_tensor(held, state, step)
                    predictions.append(model._read_state_tensor(state)[0])
                    held = drive[index:index+1]
                prediction = torch.cat(predictions).cpu().numpy()
            np.savez_compressed(run/(entry['file']+'-'+condition+'.npz'), prediction=prediction)
            record['conditions'][condition] = dict(
                full=metrics(prediction, arrays['target'], arrays['target_valid']),
                after3s=metrics(prediction, arrays['target'], late))
        reports.append(record)
    write_json(run/'result.json', dict(status='completed_pretrained_actual_flight_steering',
        training=records, development=reports, frozen_parameters_unchanged=True,
        only_turn_descending_current_head_trained=True,
        calibrated_racing_speed=False, physiological_claim=False,
        actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()
