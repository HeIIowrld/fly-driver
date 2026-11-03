"""Diagnose gas/brake gradient conflict on the original fixed body boundary."""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_body_feedback_adapter import ExperimentalBodyFeedbackCircuit
from flybrain_drive.flight_pedal_learning_v2 import native_pedal_supervision
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
        'adapter_source_sha256': sha256('src/flybrain_drive/flight_body_feedback_adapter.py'),
        'history_fingerprints': history.fingerprints, 'training_role': 'original_1909_TRAIN',
        'optimizer': 'none', 'tbptt_observations': 8,
        'loss': 'separate original gas and brake interval MSE gradients',
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
    circuit = ExperimentalBodyFeedbackCircuit(base.model).to('cuda').eval()
    arrays = history.arrays
    original_rows = original_result['fit']['final_fixed_rollout']['rows']
    if len(original_rows) != history.observations:
        raise ValueError('Complete original reference rollout required')
    for i, row in enumerate(original_rows):
        if row['index'] != i or row['decision_clock'] != arrays['decision_clock'][i]:
            raise ValueError('Original reference index and clock mismatch')
    steering_ref = torch.tensor([r['prediction']['canonical'][0][0] for r in original_rows],
                                dtype=torch.float32, device='cuda')
    values = {k: torch.tensor(arrays[k], device='cuda') for k in
              ('physical_command', 'body', 'body_valid', 'racing_goals')}

    def rollout(*, train):
        state = quiet.detach().clone()
        counts, sums, full_count, full_sum = np.zeros(2, np.int64), np.zeros(2), 0, 0.
        updates, gradients, steering_sum, steering_max = 0, [], 0., 0.
        for window, (start, end) in enumerate(history.windows):
            predictions = []
            with torch.set_grad_enabled(train):
                for index in range(start, end):
                    if index:
                        held = index-1
                        state = circuit.advance(*(values[k][held:held+1] for k in
                            ('physical_command', 'body', 'body_valid', 'racing_goals')),
                            state, float(arrays['original_elapsed_seconds'][index]),
                            **base.kwargs)[2]
                    predictions.append(base.predict(state))
                _, metric = window_loss(predictions, history, start, end)
                canonical = torch.cat([p['canonical'] for p in predictions], dim=0)
                deviation = canonical[:, 0]-steering_ref[start:end]
                steering_sum += float(deviation.detach().abs().sum())
                steering_max = max(steering_max, float(deviation.detach().abs().max()))
                native = torch.tensor(arrays['native_pedals'][start:end],
                                      dtype=canonical.dtype, device='cuda')
                mask = torch.tensor(arrays['pedal_loss_valid'][start:end],
                                    dtype=torch.bool, device='cuda')
                supervision = native_pedal_supervision(native, mask,
                                                       projection_policy='reject_unreachable')
                valid = supervision['supervision_valid']
                if not np.array_equal(valid.cpu().numpy(), history.effective_loss_valid[start:end]):
                    raise ValueError('Original pedal supervision mask mismatch')
                pedals = canonical[:, 1:]
                error = ((supervision['canonical_lower']-pedals).relu()
                         + (pedals-supervision['canonical_upper']).relu())
                per_axis = (error.square()*valid).sum(0)/valid.sum(0).clamp_min(1)
                loss = per_axis[0] + 4*per_axis[1] + deviation.square().mean()
                if train and all(metric['counts']) and loss.requires_grad:
                    parameter = circuit.adapter.projection.weight
                    gas = torch.autograd.grad(per_axis[0], parameter, retain_graph=True,
                                              allow_unused=True)[0]
                    brake = torch.autograd.grad(per_axis[1], parameter,
                                                allow_unused=True)[0]
                    if gas is None or brake is None:
                        raise ValueError('Both original axis gradients must connect')
                    if not torch.isfinite(gas).all() or not torch.isfinite(brake).all():
                        raise ValueError('Finite original axis gradients required')
                    gas_norm, brake_norm = float(gas.norm()), float(brake.norm())
                    cosine = (float((gas*brake).sum())/(gas_norm*brake_norm)
                              if gas_norm > 1e-12 and brake_norm > 1e-12 else None)
                    gradients.append({'window': window, 'gas_norm': gas_norm,
                        'brake_norm': brake_norm, 'cosine': cosine,
                        'full_brake_one_count': metric['fullbrake_one_count']})
                state = state.detach()
            counts += metric['counts']
            sums += metric['sums']
            full_count += metric['fullbrake_one_count']
            full_sum += metric['fullbrake_one_squared_error_sum']
            if train and (window+1) % 25 == 0:
                print(json.dumps({'stage': 'diagnose', 'window': window+1}), flush=True)
        return {'observations': history.observations, 'optimizer_updates': updates,
            'valid_counts': counts.tolist(),
            'pedal_mse': [float(sums[i]/counts[i]) if counts[i] else None for i in range(2)],
            'full_brake_one_count': full_count,
            'full_brake_one_mse': float(full_sum/full_count) if full_count else None,
            'mean_abs_steering_delta_from_original': steering_sum/history.observations,
            'max_abs_steering_delta_from_original': steering_max,
            'axis_gradient_windows': gradients}

    baseline = rollout(train=False)
    expected = original_result['fit']['final_fixed_rollout']['metrics']['per_axis_mean']
    if any(abs(actual-wanted) > 1e-6 for actual, wanted in zip(
            baseline['pedal_mse'], expected, strict=True)):
        raise ValueError('Neutral baseline fails original fixed-rollout reproduction')
    print(json.dumps({'stage': 'baseline', 'metrics': baseline}), flush=True)
    diagnostic = rollout(train=True)
    if base.fixed_fingerprint() != expected_fixed:
        raise ValueError('Frozen original graph/readout fingerprint changed')
    if any(not torch.equal(parameter, frozen_pedal[name]) for name, parameter in
           pedal_params.items()):
        raise ValueError('Original two DNg pedal coefficients changed')
    if any(sha256(path) != expected for path, expected in pins.items()):
        raise ValueError('Original evidence changed during run')
    result = {'status': 'completed_body_pedal_gradient_conflict_audit', 'baseline': baseline,
        'diagnostic': diagnostic,
        'original_graph_readout_and_pedal_coefficients_frozen': True,
        'training_data_only_not_independent': True, 'no_new_author_vision': True,
        'sensor_projection_is_engineering_not_measured_fly': True, 'actor_deployed': False,
        'wall_seconds': time.perf_counter()-started, 'goal_achieved': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
