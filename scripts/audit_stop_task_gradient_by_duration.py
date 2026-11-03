"""Audit local stop-task gradients by recorded full-brake duration."""

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
from flybrain_drive.pedal_tbptt_engine_v1 import whole_history
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
        'optimizer': None, 'tbptt_observations': 8,
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
    arrays = history.arrays
    values = {k: torch.tensor(arrays[k], device='cuda') for k in
              ('physical_command', 'body', 'body_valid', 'racing_goals')}
    full = arrays['native_pedals'][:, 1] == 1
    eligible = full & history.effective_loss_valid[:, 1]
    age = np.zeros(history.observations)
    run_start = None
    for i in range(history.observations):
        if full[i]:
            if run_start is None:
                run_start = arrays['decision_clock'][i]
            age[i] = arrays['decision_clock'][i] - run_start
        else:
            run_start = None
    bins = [(0., .5), (.5, 3.), (3., float('inf'))]
    reports = [dict(age_seconds=[lo, hi if np.isfinite(hi) else None], eligible_rows=0,
                    zero_brake_rows=0, loss_sum=0., windows=0, zero_gradient_windows=0,
                    gradient_abs_sum=0., gradient_signed_sum=0.) for lo, hi in bins]
    state = quiet.detach().clone()
    predicted = []
    for window, (start, end) in enumerate(history.windows):
        predictions = []
        for index in range(start, end):
            if index:
                held = index - 1
                state = circuit.advance(*(values[k][held:held+1] for k in
                    ('physical_command', 'body', 'body_valid', 'racing_goals')),
                    state, float(arrays['original_elapsed_seconds'][index]),
                    **base.kwargs)[2]
            predictions.append(base.predict(state)['canonical'])
        canonical = torch.cat(predictions, dim=0)
        predicted.append(canonical.detach().cpu().numpy())
        for report, (lo, hi) in zip(reports, bins, strict=True):
            mask = eligible[start:end] & (age[start:end] >= lo) & (age[start:end] < hi)
            if not mask.any():
                continue
            brake = canonical[torch.as_tensor(mask, device='cuda'), 2]
            loss = ((1-brake)**2).sum()
            grad = torch.autograd.grad(loss, circuit.adapter.stop_delta, retain_graph=True,
                                       allow_unused=True)[0] if loss.requires_grad else None
            g = float(grad.detach()) if grad is not None else 0.
            if not np.isfinite(g):
                raise ValueError('Nonfinite local gradient')
            report['eligible_rows'] += int(mask.sum())
            report['zero_brake_rows'] += int((brake.detach() == 0).sum())
            report['loss_sum'] += float(loss.detach())
            report['windows'] += 1
            report['zero_gradient_windows'] += int(g == 0.)
            report['gradient_abs_sum'] += abs(g)
            report['gradient_signed_sum'] += g
        state = state.detach()
        if (window + 1) % 25 == 0:
            print(json.dumps({'window': window+1}), flush=True)
    predicted = np.concatenate(predicted)
    reference = np.array([r['prediction']['canonical'][0] for r in
                          original_result['fit']['final_fixed_rollout']['rows']])
    parity = float(np.max(np.abs(predicted-reference)))
    if parity > 1e-6 or float(circuit.adapter.stop_delta.detach()) != 0:
        raise ValueError('Neutral original rollout mismatch or adapter changed')
    if base.fixed_fingerprint() != expected_fixed or any(
        not torch.equal(parameter, frozen_pedal[name]) for name, parameter in pedal_params.items()
    ) or any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError('Frozen original model or evidence changed')
    for report in reports:
        report['mean_squared_error'] = (report['loss_sum']/report['eligible_rows']
                                        if report['eligible_rows'] else None)
    result = {'status': 'completed_local_stop_gradient_duration_audit', 'reports': reports,
        'eligible_fullbrake_rows': int(eligible.sum()), 'baseline_max_abs_difference': parity,
        'interpretation_limit': ('TBPTT-8 local parameter gradients; '
                                 'prior-window state sensitivity is detached'),
        'no_optimizer_updates': True, 'actor_deployed': False,
        'wall_seconds': time.perf_counter()-started}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
