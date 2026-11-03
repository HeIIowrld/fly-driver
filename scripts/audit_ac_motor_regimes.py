"""Offline continuous-state motor diagnostics; no game or actuator API."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import episode_chunk, load_episodes, replay_chunk


def summarize(prediction, labels, valid):
    count = valid.sum(0)
    return dict(samples=len(labels), supervision_counts=count.astype(int).tolist(),
        rmse=[float(np.sqrt(e / n)) if n else None for e, n in
              zip(((prediction-labels)**2 * valid).sum(0), count, strict=True)],
        prediction_mean=prediction.mean(0).tolist() if len(labels) else None,
        target_mean=labels.mean(0).tolist() if len(labels) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training = Path(args.training_run)
    report = json.loads((training / 'result.json').read_text())
    checkpoint = training / 'checkpoint.pt'
    if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified checkpoint required')
    torch.set_num_threads(1)
    learned = torch.load(checkpoint, map_location='cpu', weights_only=True)
    graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                    'weights': np.float32}.get(k))
             for k, v in learned['graph'].items()}
    model = MultiViewCircuitController(graph, **learned['constructor'])
    model.load_state_dict(learned['model'])
    model.eval().requires_grad_(False)
    episodes, manifest = load_episodes(args.dataset, model.sensory_calibration)
    if learned.get('teacher_label_contract') != manifest.get('teacher_label_contract'):
        raise ValueError('Matching task labels required')
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        checkpoint_sha256=sha256(checkpoint),
        dataset_manifest_sha256=sha256(Path(args.dataset) / 'dataset.json'),
        seed=learned['training_configuration']['seed'], actuator_messages_sent=0,
        evaluation_role='training/development diagnosis, not independent final test'))
    rows, arrays = [], {}
    ticks = learned['training_configuration']['brain_ticks_per_decision']
    with torch.no_grad():
        for number, episode in enumerate(episodes):
            state, predictions = model.initial_state(), []
            for offset in range(0, len(episode['action']), 16):
                obs, _, _, active = episode_chunk([episode], offset, 16)
                value, state = replay_chunk(model, obs, active, state, ticks)
                predictions.append(value[:, 0][active[:, 0].bool()].numpy())
            pred = np.concatenate(predictions)
            labels, valid = episode['action'].numpy(), episode['action_valid'].numpy()
            # Teacher targets define diagnostics only; never actor features or runtime commands.
            both_pedals = valid[:, 1] & valid[:, 2]
            acceleration = both_pedals & (labels[:, 1] > .05) & (labels[:, 2] <= .05)
            braking = both_pedals & (labels[:, 2] > .05)
            coasting = both_pedals & ~acceleration & ~braking
            regions = dict(all=np.ones(len(pred), bool), accelerate=acceleration,
                           brake=braking, coast=coasting)
            segments = [dict(start=i, stop=min(i+25, len(pred)),
                **summarize(pred[i:i+25], labels[i:i+25], valid[i:i+25]))
                for i in range(0, len(pred), 25)]
            rows.append(dict(episode_id=episode['source']['episode_id'],
                regimes={k: summarize(pred[m], labels[m], valid[m]) for k, m in regions.items()},
                consecutive_25_decision_segments=segments))
            arrays.update({f'episode_{number}_prediction': pred,
                           f'episode_{number}_target': labels, f'episode_{number}_valid': valid})
    np.savez_compressed(run / 'motor-replay.npz', **arrays)
    write_json(run / 'result.json', dict(status='completed', episodes=rows,
        replay_sha256=sha256(run / 'motor-replay.npz'),
        recurrent_state='reset per original episode only; no reset between chunks',
        dataset_expert_labels_available=manifest.get('expert_labels_available'),
        actuator_messages_sent=0, independent_final_test=False, goal_achieved=False))


if __name__ == '__main__':
    main()
