"""Train mean-anchored motion sensitivity into HS/VS on the original fixed graph."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_centered_motion_boundary import CenteredMotionBoundary
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from scripts.qualify_flight_pedal_graph_v2 import tensor_digest
from scripts.train_flight_initial import substeps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--mode', choices=['qualify', 'train'], required=True)
    parser.add_argument('--qualification')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError('Original GPU execution required')
    training = Path(args.training)
    receipt, manifest = read(training/'result.json'), read(training/'manifest.json')
    plan = manifest['config']['plan']
    if (receipt['status'] != 'completed_fixed_joint_command_training'
            or sha256(receipt['checkpoint']) != receipt['checkpoint_sha256']):
        raise ValueError('Original fixed v7 checkpoint required')
    if args.mode == 'train':
        if not args.qualification:
            raise ValueError('Successful exact-source gradient qualification required')
        proof = read(Path(args.qualification)/'result.json')
        if (proof['status'] != 'qualified_trainable_hsvs_input_gradient'
                or proof['source_checkpoint_sha256'] != receipt['checkpoint_sha256']
                or proof['program_sha256'] != sha256(__file__)):
            raise ValueError('Matching source and positive qualification required')
    source = read(Path(plan['source'])/'result.json')
    if sha256(source['checkpoint']) != source['checkpoint_sha256']:
        raise ValueError('Original v6 source checkpoint changed')
    torch.set_num_threads(1)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'],
        torch.load(source['checkpoint'], map_location='cpu', weights_only=True)).to('cuda')
    checkpoint = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint['schema'] != model.schema or checkpoint['spec'] != model.spec:
        raise ValueError('Original fixed graph and model contracts required')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    model.backbone.vision = CenteredMotionBoundary(
        model.backbone.vision, model.backbone.scene_command.mean).to("cuda")
    direct_ids = torch.cat([model.backbone.steering_ports, model.backbone.goal_ports])
    if torch.isin(model.backbone.vision_ports, direct_ids).any():
        raise ValueError('Vision and direct descending input ports must not overlap')
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != {'backbone.vision.delta'}:
        raise ValueError('Only original visual-to-HS/VS boundary can train')
    before = {name: tensor_digest(tensor) for name, tensor in model.state_dict().items()}
    root = Path(plan['data'])
    dataset = read(root/'dataset.json')
    entries = [e for e in dataset['episodes'] if e['role'] == 'train']
    if len(entries) != 6:
        raise ValueError('Original six training histories required')
    if args.mode == 'qualify':
        entries = entries[:1]
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'mode': args.mode,
        'source_checkpoint_sha256': receipt['checkpoint_sha256'],
        'dataset_sha256': sha256(root/'dataset.json'), 'original_train_ids':
        [e['file'] for e in entries], 'loss': 'original two-head normalized-logit BCE',
        'tbptt_observations': 64, 'learning_rate': .0001, 'epochs': 1,
        'direct_motion_path': 'steering/goal ports receive motion-mean counterfactual drive',
        'boundary': 'fixed original plus trainable T4/T5 deviations from TRAIN mean',
        'actor_deployed': False, 'development_opened': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    optimizer = torch.optim.AdamW(trainable.values(), lr=.0001, weight_decay=.002)
    start_time = time.perf_counter()
    updates, disconnected_windows, counts, gradient_norms, losses = 0, 0, [], [], []
    for entry in entries:
        path = root/entry['file']
        if sha256(path) != entry['sha256']:
            raise ValueError('Original TRAIN episode changed')
        with load_array(path, allow_pickle=False) as archive:
            a = {k: archive[k].copy() for k in archive.files}
        if (len(a['observation_time']) != entry['observations']
                or a['target_valid'].dtype != bool or a['body_valid'].dtype != bool
                or a['visual_valid'].any() or not np.isfinite(a['target']).all()
                or np.any(np.abs(a['target']) > 1)):
            raise ValueError('Original TRAIN history, masks and targets required')
        names = ('visual', 'visual_valid', 'body', 'body_valid', 'goals', 'scene_features',
                 'target', 'target_valid')
        data = {key: torch.tensor(a[key], device='cuda') for key in names}
        schedule = [[]] + [substeps(float(dt)) for dt in np.diff(a['observation_time'])]
        stop = min(64, len(schedule)) if args.mode == 'qualify' else len(schedule)
        state, held, episode_updates = model.initial_state(1), None, 0
        for start in range(0, stop, 64):
            end = min(start+64, stop)
            optimizer.zero_grad(set_to_none=True)
            drive = model._drive_tensor(*(data[k][start:end] for k in names[:6]))
            mean_scene = data['scene_features'][start:end].clone()
            motion = torch.zeros(576, dtype=torch.bool, device='cuda').reshape(4, 9, 16)
            motion[:, 1:] = True
            motion = motion.flatten()
            mean_scene[:, motion] = model.backbone.scene_command.mean[motion]
            with torch.no_grad():
                mean_drive = model._drive_tensor(*(data[k][start:end] for k in names[:5]),
                                                 mean_scene)
            drive = drive.clone()
            drive[:, direct_ids] = mean_drive[:, direct_ids]
            predictions, indices = [], []
            for i in range(start, end):
                if i:
                    for dt in schedule[i]:
                        state = model._advance_elapsed_tensor(held, state, dt)
                if data['target_valid'][i].any():
                    predictions.append(model._read_state_preactivation_tensor(state)[0][0])
                    indices.append(i)
                held = drive[i-start:i-start+1]
            if predictions:
                pred = torch.stack(predictions)
                target = (data['target'][indices] + 1) / 2
                valid = data['target_valid'][indices]
                loss = (F.binary_cross_entropy_with_logits(2*pred, target, reduction='none')
                        * valid).sum()/valid.sum()
                if not torch.isfinite(loss):
                    raise ValueError('Finite original-label loss required')
                if not loss.requires_grad:
                    disconnected_windows += 1
                else:
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all()
                           for p in trainable.values()):
                        raise ValueError('Finite visual-boundary gradient required')
                    gradient_norm = float(torch.sqrt(sum(p.grad.square().sum()
                                                         for p in trainable.values())))
                    gradient_norms.append(gradient_norm)
                    losses.append(float(loss.detach()))
                    if args.mode == 'train' and gradient_norm > 0:
                        optimizer.step()
                        updates += 1
                        episode_updates += 1
            state = state.detach()
            held = held.detach()
        if sha256(path) != entry['sha256']:
            raise ValueError('Original TRAIN history changed during run')
        counts.append({'file': entry['file'], 'original_rows': len(schedule),
                       'replayed_rows': stop, 'optimizer_updates': episode_updates})
        print(json.dumps(counts[-1]), flush=True)
    after = {name: tensor_digest(tensor) for name, tensor in model.state_dict().items()}
    changed = [name for name in before if before[name] != after[name]]
    allowed = {'backbone.vision.delta'}
    if not set(changed) <= allowed:
        raise ValueError('Fixed graph, body, direct command or readout changed')
    if not gradient_norms or not any(x > 0 for x in gradient_norms):
        raise ValueError('No nonzero gradient to visual/HS-VS boundary')
    result = {'status': ('qualified_trainable_hsvs_input_gradient' if args.mode == 'qualify'
                         else 'completed_one_pass_hsvs_boundary_train'),
        'mode': args.mode, 'program_sha256': sha256(__file__),
        'source_checkpoint_sha256': receipt['checkpoint_sha256'],
        'optimizer_updates': updates, 'disconnected_windows': disconnected_windows,
        'history_counts': counts,
        'nonzero_gradient_windows': sum(x > 0 for x in gradient_norms),
        'gradient_norm_range': [min(gradient_norms), max(gradient_norms)],
        'loss_first_last': [losses[0], losses[-1]], 'changed_tensors': changed,
        'direct_dn_parameters_and_graph_unchanged': True,
        'direct_dn_motion_features_held_at_train_mean': True, 'development_opened': False,
        'original_flyvis_weights_changed': False, 'braking_positive_targets': 0,
        'flight_physiology_or_driving_verified': False, 'actor_deployed': False,
        'wall_seconds': time.perf_counter()-start_time, 'goal_achieved': False}
    if args.mode == 'train':
        if updates == 0 or not changed:
            raise ValueError('Training needs actual parameter updates')
        path = run/'visual-boundary.pt'
        weight, bias = model.backbone.vision.materialize()
        torch.save({'schema': 'flybrain.hsvs_visual_boundary_delta.v1',
            'weight': weight.detach().cpu(),
            'bias': bias.detach().cpu(),
            'source_checkpoint_sha256': receipt['checkpoint_sha256'],
            'actor_qualified': False}, path)
        result['checkpoint_sha256'] = sha256(path)
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
