"""Offline halt activity and truncated gradients on sealed braking-label states; no AC I/O."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.multiview_circuit import MultiViewCircuitController
from flybrain_drive.multiview_learning import configure_trainable_groups, load_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training = Path(args.training_run)
    report = json.loads((training/'result.json').read_text(encoding='utf8'))
    checkpoint = training/'checkpoint.pt'
    if report['status'] != 'completed' or sha256(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('Completed hash-verified checkpoint required')
    learned = torch.load(checkpoint, map_location='cpu', weights_only=True)
    torch.set_num_threads(1)
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
        dataset_manifest_sha256=sha256(Path(args.dataset)/'dataset.json'),
        seed=learned['training_configuration']['seed'], actuator_messages_sent=0,
        evaluation_role='training/development diagnosis; no optimizer or independent final test'))
    ticks = learned['training_configuration']['brain_ticks_per_decision']
    chunk = learned['training_configuration']['chunk_length']
    halt = model.circuit.output_ports[model.readout.halt_ports]
    matrix = model.circuit.matrix.coalesce()
    edge_mask = torch.isin(matrix.indices()[0], halt)
    post, pre = matrix.indices()[:, edge_mask]
    weights = matrix.values()[edge_mask]
    activity_rows, gradient_rows = [], []
    before = {k: v.detach().clone() for k, v in model.named_parameters()}

    def observation(episode, index):
        return {k: v[index:index+1] for k, v in episode['observation'].items()}

    for number, episode in enumerate(episodes):
        selected = episode['action_valid'][:, 2].bool() & (episode['action'][:, 2] > .05)
        state, starts, predictions = model.initial_state(), {}, []
        last = {}

        def capture(module, inputs, last=last):
            drive, previous = inputs
            contribution = weights*previous[0, pre]
            positive = [float(contribution[(post == port) & (weights > 0)].sum())
                        for port in halt]
            negative = [float(contribution[(post == port) & (weights < 0)].sum())
                        for port in halt]
            last.update(positive=positive, negative=negative,
                        previous=previous[0, halt].tolist(), drive=drive[0, halt].tolist())

        handle = model.circuit.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                for index in range(len(episode['action'])):
                    if index % chunk == 0 and selected[index:index+chunk].any():
                        starts[index] = state.clone()
                    drive = model.prepare_drive(observation(episode, index))
                    for _ in range(ticks):
                        prediction, state = model.advance(drive, state)
                    predictions.append(prediction.clone())
                    if selected[index]:
                        activity_rows.append(dict(episode=number, index=index,
                            target=episode['action'][index].tolist(),
                            prediction=prediction[0].tolist(),
                            halt_activity=state[0, halt].tolist(),
                            halt_positive_message=last['positive'],
                            halt_negative_message=last['negative'],
                            halt_direct_drive=last['drive'],
                            halt_previous_activity=last['previous']))
        finally:
            handle.remove()
        configure_trainable_groups(model, ['visual', 'organs'])
        for offset, initial in starts.items():
            model.zero_grad(set_to_none=True)
            state, predicted = initial, []
            stop = min(offset+chunk, len(episode['action']))
            for index in range(offset, stop):
                drive = model.prepare_drive(observation(episode, index))
                for _ in range(ticks):
                    prediction, state = model.advance(drive, state)
                predicted.append(prediction[0])
            values = torch.stack(predicted)
            mask = selected[offset:stop]
            loss = (values[mask, 2]-episode['action'][offset:stop][mask, 2]).square().mean()
            loss.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    gradient = parameter.grad
                    if gradient is not None and not torch.isfinite(gradient).all():
                        raise ValueError('Nonfinite diagnostic gradient')
                    gradients[name] = dict(
                        norm=float(gradient.double().norm()) if gradient is not None else None,
                        max_abs=float(gradient.abs().max()) if gradient is not None else None,
                        nonzero=int(gradient.count_nonzero()) if gradient is not None else 0)
            reference = torch.cat(predictions[offset:stop])
            gradient_rows.append(dict(episode=number, offset=offset, stop=stop,
                braking_samples=int(mask.sum()), brake_only_loss=float(loss.detach()),
                replay_max_error=float((values.detach()-reference).abs().max()),
                parameter_gradients=gradients))
        model.requires_grad_(False)
    if any(not torch.equal(value, dict(model.named_parameters())[name])
           for name, value in before.items()):
        raise ValueError('Diagnostic changed model parameters')
    write_json(run/'braking-activity.json', activity_rows)
    write_json(run/'braking-gradients.json', gradient_rows)
    write_json(run/'result.json', dict(status='completed', braking_samples=len(activity_rows),
        chunks_with_braking=len(gradient_rows),
        halt_body_ids=[str(graph['ids'][i]) for i in halt.tolist()],
        halt_readout_weights=model.readout.halt_weights.softmax(0).detach().tolist(),
        activity_sha256=sha256(run/'braking-activity.json'),
        gradients_sha256=sha256(run/'braking-gradients.json'),
        gradient_kind=model.circuit.activation_gradient,
        gradient_scope='brake-only MSE, original persistent episode state; truncated at source '
                       'chunk boundaries; all sensory adapters enabled for diagnosis only',
        model_parameters_unchanged=True, optimizer_updates=0, actuator_messages_sent=0,
        independent_final_test=False, goal_achieved=False))


if __name__ == '__main__':
    main()
