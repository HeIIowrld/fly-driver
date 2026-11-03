"""Train a stop-task DNg modulation through the frozen original flight graph."""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_stop_task_adapter import ExperimentalStopTaskCircuit
from flybrain_drive.graph import sha256
from flybrain_drive.pedal_plan_b2 import declared_loader_pedal_review
from flybrain_drive.pedal_tbptt_engine_v1 import whole_history, window_loss
from scripts.train_pedal_b2 import OriginalGraphAdapter


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-plan', required=True)
    parser.add_argument('--fit-result', required=True)
    parser.add_argument('--pedal-checkpoint', required=True)
    parser.add_argument('--qualification', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan, original_result, pedal = read(args.fit_plan), read(args.fit_result), read(
        args.pedal_checkpoint)
    qualification = read(Path(args.qualification)/'result.json')
    if (qualification['status'] != 'qualified_neutral_body_adapter_actual_graph'
            or qualification['original_body_observations'] != 986
            or original_result['status'] != 'completed_one_pass_pedal_component_fit'
            or original_result['optimizer_updates'] != 123
            or pedal['bindings'] != original_result['bindings']):
        raise ValueError('Qualified new boundary and original completed DNg fit required')
    if sha256(plan['loader_binding']['source_path']) != plan['loader_binding']['source_sha256']:
        raise ValueError('Original complete-history loader changed')
    loader = importlib.import_module(plan['loader_binding']['module'])
    loaded = getattr(loader, plan['loader_binding']['function'])(plan['capture_root'],
        expected_capture_manifest_sha256=plan['capture_manifest_sha256'],
        pedal_review_bytes=declared_loader_pedal_review(plan))
    if (loaded.get('fit_allowed') is not False or loaded.get('all_states_retained') is not True
            or loaded.get('source_clock_schedule_valid') is not True
            or loaded.get('physical_body_interface_valid') is not True):
        raise ValueError('Original complete-history loader evidence required')
    history = whole_history(loaded['arrays'], expected_fingerprints=plan['history']['fingerprints'])
    if history.observations != 986 or history.nonempty_windows != 123:
        raise ValueError('Original unretimed complete TRAIN history required')
    pins = {path: sha256(path) for path in (args.fit_plan, args.fit_result, args.pedal_checkpoint,
        str(Path(args.qualification)/'result.json'), plan['loader_binding']['source_path'])}
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'source_hashes': pins,
        'adapter_source_sha256': sha256('src/flybrain_drive/flight_stop_task_adapter.py'),
        'history_fingerprints': history.fingerprints, 'training_role': 'original_1909_TRAIN',
        'optimizer': 'Adam lr=0.001, no weight decay', 'tbptt_observations': 8,
        'one_pass': True, 'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    base = OriginalGraphAdapter(plan, lambda: None)
    pedal_params = base.parameters()
    with torch.no_grad():
        for name, parameter in pedal_params.items():
            parameter.copy_(torch.tensor(pedal['values'][name], device=parameter.device))
        quiet = base.initialize()
    frozen_pedal = {name: parameter.detach().clone() for name, parameter in
                    pedal_params.items()}
    expected_fixed = original_result['fit']['final_fixed_rollout']['fixed_fingerprint']
    if base.fixed_fingerprint() != expected_fixed:
        raise ValueError('Fixed original flight graph and readout changed')
    base.model.requires_grad_(False)
    circuit = ExperimentalStopTaskCircuit(base.model).to('cuda').eval()
    optimizer = torch.optim.Adam(circuit.adapter.parameters(), lr=.001, weight_decay=0,
                                 foreach=False, fused=False)
    arrays = history.arrays
    values = {k: torch.tensor(arrays[k], device='cuda') for k in
              ('physical_command', 'body', 'body_valid', 'racing_goals')}

    def rollout(*, mode):
        if mode not in ('baseline', 'qualify', 'train', 'final'):
            raise ValueError('Explicit evaluation, gradient qualification or train mode required')
        state = quiet.detach().clone()
        counts, sums, full_count, full_sum = np.zeros(2, np.int64), np.zeros(2), 0, 0.
        updates, gradients = 0, []
        for window, (start, end) in enumerate(history.windows):
            if mode == 'train':
                optimizer.zero_grad(set_to_none=True)
            predictions = []
            with torch.set_grad_enabled(mode in ('train', 'qualify')):
                for index in range(start, end):
                    if index:
                        held = index-1
                        state = circuit.advance(*(values[k][held:held+1] for k in
                            ('physical_command', 'body', 'body_valid', 'racing_goals')),
                            state, float(arrays['original_elapsed_seconds'][index]),
                            **base.kwargs)[2]
                    predictions.append(base.predict(state))
                loss, metric = window_loss(predictions, history, start, end)
                if mode in ('train', 'qualify') and metric['counts'].sum() and loss.requires_grad:
                    loss.backward()
                    grad = circuit.adapter.stop_delta.grad
                    if grad is None or not torch.isfinite(grad).all():
                        raise ValueError('Finite body-boundary gradient required')
                    norm = float(grad.norm())
                    gradients.append(norm)
                    if mode == 'train' and norm > 0:
                        optimizer.step()
                        updates += 1
                    if mode == 'qualify':
                        circuit.adapter.zero_grad(set_to_none=True)
                state = state.detach()
            counts += metric['counts']
            sums += metric['sums']
            full_count += metric['fullbrake_one_count']
            full_sum += metric['fullbrake_one_squared_error_sum']
            if mode in ('train', 'qualify') and (window+1) % 25 == 0:
                print(json.dumps({'stage': mode, 'window': window+1,
                                  'optimizer_updates': updates}), flush=True)
        return {'observations': history.observations, 'optimizer_updates': updates,
            'valid_counts': counts.tolist(),
            'pedal_mse': [float(sums[i]/counts[i]) if counts[i] else None for i in range(2)],
            'full_brake_one_count': full_count,
            'full_brake_one_mse': float(full_sum/full_count) if full_count else None,
            'nonzero_gradient_windows': sum(x > 0 for x in gradients)}

    baseline = rollout(mode='baseline')
    expected = original_result['fit']['final_fixed_rollout']['metrics']['per_axis_mean']
    if any(abs(actual-wanted) > 1e-6 for actual, wanted in zip(
            baseline['pedal_mse'], expected, strict=True)):
        raise ValueError('Neutral baseline fails original fixed-rollout reproduction')
    print(json.dumps({'stage': 'baseline', 'metrics': baseline}), flush=True)
    qualification = rollout(mode='qualify')
    if not qualification['nonzero_gradient_windows']:
        raise ValueError('Nonzero full-history gradient qualification required')
    print(json.dumps({'stage': 'qualified', 'metrics': qualification}), flush=True)
    trained = rollout(mode='train')
    if not trained['optimizer_updates']:
        raise ValueError('Actual nonzero training updates required')
    final = rollout(mode='final')
    if base.fixed_fingerprint() != expected_fixed:
        raise ValueError('Frozen original graph/readout fingerprint changed')
    if any(not torch.equal(parameter, frozen_pedal[name]) for name, parameter in
           pedal_params.items()):
        raise ValueError('Original two DNg pedal coefficients changed')
    if any(sha256(path) != expected for path, expected in pins.items()):
        raise ValueError('Original evidence changed during run')
    checkpoint = run/'stop-task-adapter.pt'
    torch.save({'schema': 'flybrain.experimental_stop_task_dn.v1',
        'model': circuit.adapter.state_dict(), 'source_pedal_checkpoint_sha256':
        sha256(args.pedal_checkpoint), 'actor_qualified': False}, checkpoint)
    result = {'status': 'completed_one_pass_stop_task_dn_fit', 'baseline': baseline,
        'gradient_qualification': qualification, 'training': trained, 'final': final,
        'learned_stop_delta': float(circuit.adapter.stop_delta.detach()),
        'checkpoint_sha256': sha256(checkpoint),
        'original_graph_readout_and_pedal_coefficients_frozen': True,
        'training_data_only_not_independent': True, 'no_new_author_vision': True,
        'task_projection_is_engineering_not_measured_fly': True, 'actor_deployed': False,
        'wall_seconds': time.perf_counter()-started, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
