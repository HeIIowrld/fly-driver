"""Whole-episode channel-specific body intervention, diagnostic only.

All visual input stays original. Neutralized values are NOT real sensor repairs
or physiological lesions, and never become training data or vehicle requests.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import assemble_student, MuscleVehicleReadout, read
from scripts.improve_sac_anticipation import command_metrics
from scripts.trace_sac_premuscle import CHECKPOINT, CHECKPOINT_SHA
from scripts.train_qualified_sac_unified import context, load_episode


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['yaw_zero', 'temperature_train_mean', 'wheel_train_mean', 'tilt_load_train_mean'], required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Original best offline candidate differs')
    entries = read('runs/teacher-synchronized-20261102/student-split-01.json')['episodes']
    items = [load_episode(e['data']) for e in entries]
    d, contract, evidence = items[2]
    changed = {k: v.copy() for k, v in d.items()}
    if a.case == 'yaw_zero':
        indices, replacement = [4], np.array([0.])
    else:
        indices = (list(range(12, 16)) if a.case == 'temperature_train_mean' else
                   list(range(17, 21)) if a.case == 'wheel_train_mean' else [0, 1, 8, 9, 10, 11])
        train = np.concatenate([x['body'][x['fit_mask']][:, indices] for x, _, _ in items[:2]])
        replacement = train.mean(0)
    changed['body'][:, indices] = replacement
    changed['view_body'][:, :, indices] = replacement
    model, lineage = assemble_student()
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    model.motor.backbone.circuit.sparse_backend = 'cached_csr_cpu'
    _, muscle = model.motor.backbone._read_state_tensor(model.motor.initial_state(1))
    readout = MuscleVehicleReadout(muscle.shape[-1])
    readout.load_state_dict(payload['readout'], strict=True)
    readout.eval().requires_grad_(False)
    run = begin_run(a.output, dict(arguments=vars(a), seed=780406, program_sha256=sha256(__file__),
        checkpoint_sha256=CHECKPOINT_SHA, original_episode=evidence, lineage=lineage,
        changed_channels=indices, replacement=replacement.tolist(), camera='All original four-view observations unchanged',
        scope='Consumed development teacher-trajectory diagnostic, not real missing sensing or physiological lesion',
        optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False))
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False)
    start = time.monotonic()
    try:
        scene = torch.tensor(d['scene'], dtype=torch.float32)
        goals, view = torch.tensor(racing_goal(300, 1)), torch.ones(1, 4, dtype=torch.bool)
        model.reset_episode()
        predictions = []
        for i in range(len(scene)):
            _, muscle = model.sensory_decision(decision_time=float(d['decision_time'][i]),
                racing_goals=goals, scene_features=scene[i:i+1], scene_valid=view,
                body_context=context(changed, contract, i))
            predictions.append(readout(muscle)[0].numpy().copy())
            if i % 500 == 0:
                write_json(run/'progress.json', dict(event=i, total=len(scene), elapsed_seconds=time.monotonic()-start))
        prediction = np.asarray(predictions)
        np.savez_compressed(run/'predictions.npz', prediction=prediction)
        normal = read('runs/visual-anticipation-20261103/continuation8-02/result.json')['arms']['continuation']['dev']
        metrics = command_metrics(prediction, d['labels'], d['fit_mask'])
        result.update(status='completed_body_channel_information_intervention', metrics=metrics,
            normal=normal, weighted_mse_ratio=metrics['weighted_mse']/normal['weighted_mse'],
            predictions_sha256=sha256(run/'predictions.npz'),
            interpretation='Causal response to this channel manipulation on a teacher trajectory; may be OOD and is not autonomous driving or physiological proof',
            termination_reason='Full original episode continuously replayed with one declared channel intervention')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit intervention failure')
    result['elapsed_seconds'] = time.monotonic()-start
    write_json(run/'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
