"""Trace the exact best offline flight candidate before its six-muscle summary.

No fitting or simulator. All native/teacher events retain original recurrence;
full neural states are stored only at original supervised label events.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import assemble_student, MuscleVehicleReadout, read
from scripts.train_qualified_sac_unified import context, load_episode

CHECKPOINT = Path('runs/visual-anticipation-20261103/continuation8-02/continuation.pt')
CHECKPOINT_SHA = 'da4b3342f2e09e5fc3ce7653268922bf7036f96c149cf88aec33717a6aa345de'


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(780401)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Exact best offline development candidate required')
    entries = read('runs/teacher-synchronized-20261102/student-split-01.json')['episodes']
    items = [load_episode(e['data']) for e in entries]
    if any(c['episode_id'] != e['episode_id'] or proof['arrays_sha256'] != e['arrays_sha256']
           for e, (_, c, proof) in zip(entries, items)):
        raise ValueError('Original chronological independent source split changed')
    model, lineage = assemble_student()
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if payload['lineage'] != lineage or payload['actor_qualified'] is not False:
        raise ValueError('Candidate circuit lineage/qualification differs')
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    model.motor.backbone.circuit.sparse_backend = 'cached_csr_cpu'
    backbone = model.motor.backbone
    groups = {key: getattr(backbone, key + '_ports').cpu().tolist()
              for key in ['vision', 'body', 'steering', 'goal']}
    groups['motor'] = backbone.circuit.output_ports.cpu().tolist()
    matrix = backbone.circuit.matrix.to_sparse_coo().coalesce()
    row, col = matrix.indices().cpu().numpy()
    active = matrix.values().cpu().numpy() != 0
    groups['premotor'] = np.unique(col[active & np.isin(row, groups['motor'])]).tolist()
    _, activity = backbone._read_state_tensor(model.motor.initial_state(1))
    readout = MuscleVehicleReadout(activity.shape[-1])
    readout.load_state_dict(payload['readout'], strict=True)
    readout.eval().requires_grad_(False)
    run = begin_run(a.output, dict(arguments=vars(a), seed=780401, program_sha256=sha256(__file__),
        checkpoint_sha256=CHECKPOINT_SHA, episodes=entries, lineage=lineage,
        groups=groups, neuronal_states='Original measured-ID rate surrogate state; no new physiology claim',
        scope='Consumed Monza development, full continuous teacher-trajectory replay; not autonomous driving',
        optimizer_updates=0, vehicle_control_sent=False, actor_deployed=False))
    write_json(run / 'groups.json', groups)
    result = dict(status='failed', optimizer_updates=0, vehicle_control_sent=False, actor_deployed=False)
    goals, views = torch.tensor(racing_goal(300, 1)), torch.ones(1, 4, dtype=torch.bool)
    begin = time.monotonic()
    captured = {}
    hook = model.visual.register_forward_hook(lambda module, inputs, output:
        captured.update(visual_base=output.sigmoid().numpy().copy()))
    try:
        files, reports = {}, []
        for episode, (data, contract, _) in enumerate(items):
            supervised = np.flatnonzero(data['fit_mask'])
            states = np.lib.format.open_memmap(run / f'state-{episode}.npy', mode='w+',
                dtype=np.float32, shape=(len(supervised), model.motor.neurons))
            scene = torch.tensor(data['scene'], dtype=torch.float32)
            saved = {key: [] for key in ['event_indices', 'muscles', 'visual_base', 'visual_drive', 'prediction']}
            model.reset_episode()
            k = 0
            for i in range(len(scene)):
                _, muscle = model.sensory_decision(decision_time=float(data['decision_time'][i]),
                    racing_goals=goals, scene_features=scene[i:i+1], scene_valid=views,
                    body_context=context(data, contract, i))
                if data['fit_mask'][i]:
                    states[k] = model.state[0].numpy()
                    saved['event_indices'].append(i)
                    saved['muscles'].append(muscle[0].numpy().copy())
                    saved['visual_base'].append(captured['visual_base'][0])
                    saved['visual_drive'].append(model.previous_drive[0, backbone.vision_ports].numpy().copy())
                    saved['prediction'].append(readout(muscle)[0].numpy().copy())
                    k += 1
                if i % 500 == 0:
                    write_json(run / 'progress.json', dict(episode=episode, event=i, total=len(scene),
                        supervised_states=k, elapsed_seconds=time.monotonic()-begin))
            states.flush()
            del states
            if k != len(supervised):
                raise ValueError('Every original labelled state required')
            np.savez_compressed(run / f'observed-{episode}.npz', **{key: np.asarray(v) for key, v in saved.items()})
            for name in [f'state-{episode}.npy', f'observed-{episode}.npz']:
                files[name] = sha256(run / name)
            error = np.asarray(saved['prediction']) - data['labels'][supervised]
            reports.append(dict(events=len(scene), labels=k,
                weighted_mse=float(np.mean(error**2 * [4, 1, 1]))))
        with np.load('runs/visual-anticipation-20261103/continuation8-02/predictions.npz', allow_pickle=False) as old:
            with np.load(run / 'observed-2.npz', allow_pickle=False) as new:
                expected = old['continuation_dev_prediction'][items[2][0]['fit_mask']]
                delta = float(abs(expected - new['prediction']).max())
        if delta > 2e-6:
            raise ValueError('Original candidate replay changed')
        model.motor.backbone.assert_body_boundary_protected()
        result.update(status='completed_premuscle_state_trace', files_sha256=files, episodes=reports,
            best_candidate_replay_max_error=delta, groups_sha256=sha256(run / 'groups.json'),
            group_sizes={key: len(value) for key, value in groups.items()},
            termination_reason='All three original episodes continuously replayed; source and numeric states preserved')
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit trace failure')
    finally:
        hook.remove()
    result['elapsed_seconds'] = time.monotonic()-begin
    write_json(run / 'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
