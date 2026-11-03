"""Offline body-channel interventions through the fixed physical flight graph."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_body_feedback_adapter import VehicleBodyFeedbackAdapter
from flybrain_drive.flight_embodied_navigation import modulate_current
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
    parser.add_argument('--body-fit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan, fitted, pedal = read(args.fit_plan), read(args.fit_result), read(args.pedal_checkpoint)
    body_fit = Path(args.body_fit)
    result = read(body_fit/'result.json')
    if (fitted['status'] != 'completed_one_pass_pedal_component_fit'
            or result['status'] != 'completed_one_pass_body_feedback_fit'
            or sha256(body_fit/'body-feedback-adapter.pt') != result['checkpoint_sha256']
            or pedal['bindings'] != fitted['bindings']
            or sha256(plan['loader_binding']['source_path'])
            != plan['loader_binding']['source_sha256']):
        raise ValueError('Original fixed fit and completed body adapter required')
    loader = importlib.import_module(plan['loader_binding']['module'])
    loaded = getattr(loader, plan['loader_binding']['function'])(plan['capture_root'],
        expected_capture_manifest_sha256=plan['capture_manifest_sha256'],
        pedal_review_bytes=declared_loader_pedal_review(plan))
    if (loaded.get('all_states_retained') is not True
            or loaded.get('source_clock_schedule_valid') is not True):
        raise ValueError('Unmodified complete TRAIN history required')
    history = whole_history(loaded['arrays'], expected_fingerprints=plan['history']['fingerprints'])
    if history.observations != 986:
        raise ValueError('All original 986 observations required')
    paths = [args.fit_plan, args.fit_result, args.pedal_checkpoint,
             str(body_fit/'result.json'), str(body_fit/'body-feedback-adapter.pt')]
    pins = {p: sha256(p) for p in paths}
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'source_hashes': pins,
        'role': 'train_only_body_feature_interventions', 'optimizer_updates': 0,
        'actor_deployed': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    base = OriginalGraphAdapter(plan, lambda: None)
    with torch.no_grad():
        for name, parameter in base.parameters().items():
            parameter.copy_(torch.tensor(pedal['values'][name], device=parameter.device))
        quiet = base.initialize()
    if base.fixed_fingerprint() != fitted['fit']['final_fixed_rollout']['fixed_fingerprint']:
        raise ValueError('Original fixed graph/readout changed')
    base.model.requires_grad_(False)
    adapter = VehicleBodyFeedbackAdapter(len(base.model.backbone.body_ports)).to('cuda')
    payload = torch.load(body_fit/'body-feedback-adapter.pt', map_location='cpu', weights_only=True)
    if (payload['schema'] != 'flybrain.experimental_body_feedback_adapter.v1'
            or payload['source_pedal_checkpoint_sha256'] != sha256(args.pedal_checkpoint)):
        raise ValueError('Exact body candidate and pedal checkpoint binding required')
    adapter.load_state_dict(payload['model'], strict=True)
    adapter.eval().requires_grad_(False)
    arrays = history.arrays
    body = torch.tensor(arrays['body'], device='cuda')
    valid = torch.tensor(arrays['body_valid'], device='cuda')
    command = torch.tensor(arrays['physical_command'], device='cuda')
    goals = torch.tensor(arrays['racing_goals'], device='cuda')
    groups = {'tilt_mean': [0, 1], 'load_mean': list(range(8, 12)),
              'temperature_mean': list(range(12, 16)), 'wheel_mean': list(range(17, 21))}
    conditions = ['intact', *groups, 'neutral_adapter']
    data = []
    for name in conditions:
        b = body.clone()
        if name in groups:
            for index in groups[name]:
                available = valid[:, index]
                if not available.any():
                    raise ValueError('Original valid TRAIN samples required for mean intervention')
                b[available, index] = b[available, index].mean()
        data.append(b)
    with torch.no_grad():
        original_drive = base.model.drive(command, body, valid, goals, **base.kwargs)
        port = base.model.backbone.body_ports
        drives = []
        for name, b in zip(conditions, data, strict=True):
            drive = original_drive.clone()
            if name != 'neutral_adapter':
                drive[:, port] = modulate_current(original_drive[:, port], adapter(b, valid))
            drives.append(drive)
        state = quiet.detach().expand(len(conditions), -1).clone()
        predictions = [[] for _ in conditions]
        for index in range(history.observations):
            if index:
                held = index-1
                step = torch.cat([x[held:held+1] for x in drives], 0)
                nominal = base.model.backbone.spec['dynamics']['dt']
                elapsed = float(arrays['original_elapsed_seconds'][index])
                count = int(np.floor(elapsed/nominal))
                for _ in range(count):
                    state = base.model.backbone._advance_elapsed_tensor(step, state, nominal)
                tail = elapsed-count*nominal
                if tail > 1e-12:
                    state = base.model.backbone._advance_elapsed_tensor(step, state, tail)
            decoded = base.predict(state)
            for i in range(len(conditions)):
                predictions[i].append({'canonical': decoded['canonical'][i:i+1]})
        reports = []
        canonical = [torch.cat([p['canonical'] for p in rows]).cpu().numpy()
                     for rows in predictions]
        for name, rows, prediction in zip(conditions, predictions, canonical, strict=True):
            counts, sums, full_count, full_sum = np.zeros(2, np.int64), np.zeros(2), 0, 0.
            for start, end in history.windows:
                _, metric = window_loss(rows[start:end], history, start, end)
                counts += metric['counts']
                sums += metric['sums']
                full_count += metric['fullbrake_one_count']
                full_sum += metric['fullbrake_one_squared_error_sum']
            reports.append({'condition': name, 'valid_counts': counts.tolist(),
                'pedal_mse': (sums/counts).tolist(),
                'full_brake_one_mse': float(full_sum/full_count),
                'mean_absolute_output_difference': np.abs(
                    prediction-canonical[0]).mean(0).tolist()})
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError('Original input changed during intervention')
    write_json(run/'result.json', {'status': 'completed_train_body_interventions',
        'reports': reports, 'observations': history.observations,
        'original_body_to_motor_unchanged': True, 'interventions_only_adapter_features': True,
        'original_masks_and_clocks_preserved': True, 'train_only_not_independent': True,
        'no_physiological_or_real_driving_claim': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
