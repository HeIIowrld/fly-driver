"""Train-only R1 gradient-credit audit at identical numeric flight states.

No optimizer. Replay the complete original prefix, then compare 8/32/64-event
gradient detach schedules on the next 128 events, with identical teacher loss.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import (assemble_student, detach_recurrence,
                                               MuscleVehicleReadout, read)
from scripts.train_qualified_sac_unified import context, load_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(780307)
    root = Path('runs/teacher-synchronized-20261102')
    checkpoint = root / 'student-fit-01/unified-sac-student.pt'
    if sha256(checkpoint) != '24a8f4e03614636fe6eede71137faae161c8e82dcc1f212dc4bd4a75fb5203d4':
        raise ValueError('Original student checkpoint changed')
    entry = read(root / 'student-split-01.json')['episodes'][0]
    if entry['role'] != 'train':
        raise ValueError('Training-only episode required')
    data, contract, evidence = load_episode(entry['data'])
    if evidence['arrays_sha256'] != entry['arrays_sha256']:
        raise ValueError('Original training episode changed')
    first = int(np.flatnonzero(data['fit_mask'])[0])
    model, _ = assemble_student()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    model.motor.backbone.circuit.sparse_backend = 'cached_csr_cpu'
    _, muscles = model.motor.backbone._read_state_tensor(model.motor.initial_state(1))
    readout = MuscleVehicleReadout(muscles.shape[-1])
    readout.load_state_dict(state['readout'], strict=True)
    model.eval()
    readout.eval()
    run = begin_run(a.output, dict(seed=780307, program_sha256=sha256(__file__),
        source_checkpoint_sha256=sha256(checkpoint), episode=evidence,
        first_event=first, gradient_events=128, windows=[8, 32, 64],
        prefix='Complete original prefix once; exact cloned recurrent state used for independent diagnostic replays',
        optimizer_updates=0, consumed_dev_opened=False, actor_deployed=False))
    result = dict(status='failed', optimizer_updates=0, vehicle_control_sent=False)
    output, reports = {}, {}
    scene = torch.tensor(data['scene'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.float32)
    goals, valid = torch.tensor(racing_goal(300, 1)), torch.ones((1, 4), dtype=torch.bool)
    parameters = [(n, param) for n, param in model.named_parameters() if param.requires_grad]

    def decision(i, learn=False):
        fn = model.sensory_training_decision if learn else model.sensory_decision
        return fn(decision_time=float(data['decision_time'][i]), racing_goals=goals,
            scene_features=scene[i:i+1], scene_valid=valid,
            body_context=context(data, contract, i))[1]

    try:
        for i in range(first):
            decision(i)
            if i % 500 == 0:
                write_json(run / 'progress.json', dict(phase='original_prefix', event=i, total=first))
        keys = ['state', 'traces', 'previous_drive', 'previous_scene', 'previous_raw_scene',
                'previous_time', 'previous_body_receipt', 'previous_render_lowers']
        prefix = {key: copy.deepcopy(getattr(model, key)) for key in keys}
        count = int(data['fit_mask'][first:first+128].sum())
        if count < 1:
            raise ValueError('Original labelled train interval required')
        for window in [8, 32, 64]:
            model.reset_episode()
            for key, value in prefix.items():
                setattr(model, key, copy.deepcopy(value))
            model.zero_grad(set_to_none=True)
            readout.zero_grad(set_to_none=True)
            losses, predictions = [], []
            loss_total = 0.
            for offset in range(128):
                i = first + offset
                prediction = readout(decision(i, True))
                predictions.append(prediction[0].detach().numpy().copy())
                if data['fit_mask'][i]:
                    loss = ((prediction[0] - labels[i]).square() * prediction.new_tensor([4., 1., 1.])).mean() / count
                    losses.append(loss)
                    loss_total += float(loss.detach())
                if (offset + 1) % window == 0:
                    if losses:
                        torch.stack(losses).sum().backward()
                        losses.clear()
                    detach_recurrence(model)
            gradient = np.concatenate([param.grad.detach().numpy().ravel() if param.grad is not None
                                       else np.zeros(param.numel(), dtype=np.float32) for _, param in parameters])
            if not np.isfinite(gradient).all():
                raise ValueError('Nonfinite gradient')
            output[f'gradient_{window}'] = gradient
            output[f'prediction_{window}'] = np.asarray(predictions)
            reports[str(window)] = dict(loss=loss_total, gradient_norm=float(np.linalg.norm(gradient)),
                parameter_gradient_norm={n: float(param.grad.norm()) if param.grad is not None else 0.
                                         for n, param in parameters})
            write_json(run / 'progress.json', dict(phase='gradient_audit', completed_window=window))
        for window in [32, 64]:
            delta = abs(output[f'prediction_{window}'] - output['prediction_8']).max()
            if delta > 1e-6:
                raise ValueError('Detach changed numeric forward state')
            first_grad, current = output['gradient_8'], output[f'gradient_{window}']
            denom = np.linalg.norm(first_grad) * np.linalg.norm(current)
            reports[str(window)].update(forward_max_error=float(delta),
                cosine_with_8=float(first_grad @ current / denom) if denom else None)
        np.savez_compressed(run / 'credit.npz', **output)
        result.update(status='completed_train_only_credit_audit', reports=reports,
            arrays_sha256=sha256(run / 'credit.npz'), prefix_events=first, labelled_events=count,
            source_checkpoint_unchanged=sha256(checkpoint)==read(run / 'manifest.json')['config']['source_checkpoint_sha256'],
            scope='Local train interval gradient sensitivity, not whole-episode performance or physiological latency',
            termination_reason='Three fixed gradient windows compared; no optimizer')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit diagnostic failure')
    write_json(run / 'result.json', result)
    print(result)
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
