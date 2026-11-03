"""Compare original-episode fit on declared training data; never a held-out driving test."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import load_episodes, replay_metrics


def brake_fit_metrics(predictions, labels, valid, threshold=.05):
    """Separate missed braking from unwanted braking; labels are not safety truth."""
    selected = valid[:, 2].astype(bool)
    positive = selected & (labels[:, 2] > threshold)
    negative = selected & ~positive
    requested = predictions[:, 2] > threshold
    result = dict(label_threshold=threshold)
    for name, mask in [('positive', positive), ('negative', negative)]:
        count = int(mask.sum())
        result[name] = dict(samples=count,
            rmse=float(np.sqrt(np.mean((predictions[mask, 2]-labels[mask, 2])**2)))
                if count else None,
            mean_request=float(predictions[mask, 2].mean()) if count else None,
            requests_above_threshold=int((requested & mask).sum()))
    return result


def throttle_fit_metrics(predictions, labels, valid):
    """Actual valid label classes; reported requests are not closed-loop speed evidence."""
    selected = valid[:, 1].astype(bool)
    result = {}
    for name, mask in [('coast', selected & (labels[:, 1] <= .005)),
                       ('positive_drive', selected & (labels[:, 1] > .05))]:
        count = int(mask.sum())
        result[name] = dict(samples=count,
            rmse=float(np.sqrt(np.mean((predictions[mask, 1]-labels[mask, 1])**2)))
                if count else None,
            mean_request=float(predictions[mask, 1].mean()) if count else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', action='append', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    run = begin_run(args.output, dict(arguments=vars(args), seed=862953, scope=__doc__,
        entry_script_sha256=sha256(__file__),
        dataset_sha256=sha256(Path(args.dataset)/'dataset.json'),
        actuator_messages_sent=0, evaluation_role='training fit / successful-prefix regression'))
    results = []
    for directory in args.training_run:
        root = Path(directory)
        report = json.loads((root/'result.json').read_text(encoding='utf8'))
        if (report['status'] != 'completed'
                or sha256(root/'checkpoint.pt') != report['checkpoint_sha256']):
            raise ValueError('Completed verified model required')
        source = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=True)
        graph = {k: np.asarray(v, dtype={'pre': np.int64, 'post': np.int64,
                                        'weights': np.float32}.get(k))
                 for k, v in source['graph'].items()}
        constructor = copy.deepcopy(source['constructor'])
        constructor['sparse_backend'] = 'cached_csr_gpu'
        model = MultiViewCircuitController(graph, **constructor).cuda().eval().requires_grad_(False)
        model.load_state_dict(source['model'])
        model.chunk_validated_training = True
        episodes, _ = load_episodes(args.dataset, model.sensory_calibration)
        rows = []
        for episode in episodes:
            episode['observation'] = {k: v.cuda() for k, v in episode['observation'].items()}
            episode['action'] = episode['action'].cuda()
            episode['action_valid'] = episode['action_valid'].cuda()
            predictions = []

            def collect(prediction, batch, offset, predictions=predictions):
                predictions.append(prediction[:, 0].detach().cpu().numpy())
                return 0.

            metrics = replay_metrics(model, [episode], 64, 4, chunk_metric=collect)
            metrics.pop('extra_chunk_metric')
            replayed = np.concatenate(predictions)[:len(episode['action'])]
            metrics['last_prediction'] = replayed[-1].tolist()
            metrics['last50_prediction_mean'] = replayed[-50:].mean(0).tolist()
            metrics['brake_subsets'] = brake_fit_metrics(
                np.concatenate(predictions)[:len(episode['action'])],
                episode['action'].cpu().numpy(), episode['action_valid'].cpu().numpy())
            metrics['throttle_subsets'] = throttle_fit_metrics(
                np.concatenate(predictions)[:len(episode['action'])],
                episode['action'].cpu().numpy(), episode['action_valid'].cpu().numpy())
            rows.append(dict(source=episode['source'], metrics=metrics))
        results.append(dict(training_run=directory, checkpoint_sha256=sha256(root/'checkpoint.pt'),
            exact_dataset_manifest_matches_training=(source['training_teacher_manifest_sha256']
                == sha256(Path(args.dataset)/'dataset.json')), episodes=rows))
        write_json(run/'progress.json', results)
        del model, episodes
    write_json(run/'result.json', dict(status='completed', models=results,
        scope=__doc__, held_out=False, closed_loop_verified=False,
        actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
