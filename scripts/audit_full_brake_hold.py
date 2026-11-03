"""Compare original and fixed body candidates over original TRAIN full-brake durations."""

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
from flybrain_drive.pedal_tbptt_engine_v1 import whole_history
from scripts.train_pedal_b2 import OriginalGraphAdapter


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-plan', required=True)
    parser.add_argument('--fit-result', required=True)
    parser.add_argument('--pedal-checkpoint', required=True)
    parser.add_argument('--body-fit', required=True)
    parser.add_argument('--anchored-fit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan, fitted, pedal = read(args.fit_plan), read(args.fit_result), read(args.pedal_checkpoint)
    dirs = [Path(args.body_fit), Path(args.anchored_fit)]
    body_results = [read(path/'result.json') for path in dirs]
    if (fitted['status'] != 'completed_one_pass_pedal_component_fit'
            or pedal['bindings'] != fitted['bindings']
            or sha256(plan['loader_binding']['source_path'])
            != plan['loader_binding']['source_sha256']):
        raise ValueError('Original completed fixed model and loader required')
    for directory, result in zip(dirs, body_results, strict=True):
        if sha256(directory/'body-feedback-adapter.pt') != result['checkpoint_sha256']:
            raise ValueError('Original fixed body candidate changed')
    loader = importlib.import_module(plan['loader_binding']['module'])
    loaded = getattr(loader, plan['loader_binding']['function'])(plan['capture_root'],
        expected_capture_manifest_sha256=plan['capture_manifest_sha256'],
        pedal_review_bytes=declared_loader_pedal_review(plan))
    history = whole_history(loaded['arrays'], expected_fingerprints=plan['history']['fingerprints'])
    if history.observations != 986 or loaded.get('all_states_retained') is not True:
        raise ValueError('Original complete 986-state TRAIN history required')
    paths = [args.fit_plan, args.fit_result, args.pedal_checkpoint,
             *[str(d/'result.json') for d in dirs],
             *[str(d/'body-feedback-adapter.pt') for d in dirs]]
    pins = {p: sha256(p) for p in paths}
    run = begin_run(args.output, {'program_sha256': sha256(__file__), 'source_hashes': pins,
        'role': 'original_train_posthoc_brake_duration_audit', 'optimizer_updates': 0,
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
    adapters = []
    for directory in dirs:
        payload = torch.load(directory/'body-feedback-adapter.pt', map_location='cpu',
                             weights_only=True)
        if payload['source_pedal_checkpoint_sha256'] != sha256(args.pedal_checkpoint):
            raise ValueError('Candidate pedal source binding mismatch')
        adapter = VehicleBodyFeedbackAdapter(len(base.model.backbone.body_ports)).to('cuda')
        adapter.load_state_dict(payload['model'], strict=True)
        adapters.append(adapter.eval().requires_grad_(False))
    a = history.arrays
    body = torch.tensor(a['body'], device='cuda')
    valid = torch.tensor(a['body_valid'], device='cuda')
    command = torch.tensor(a['physical_command'], device='cuda')
    goals = torch.tensor(a['racing_goals'], device='cuda')
    with torch.no_grad():
        original = base.model.drive(command, body, valid, goals, **base.kwargs)
        drives = [original]
        ports = base.model.backbone.body_ports
        for adapter in adapters:
            drive = original.clone()
            drive[:, ports] = modulate_current(original[:, ports], adapter(body, valid))
            drives.append(drive)
        state = quiet.detach().expand(3, -1).clone()
        predictions = []
        nominal = base.model.backbone.spec['dynamics']['dt']
        for i in range(history.observations):
            if i:
                elapsed = float(a['original_elapsed_seconds'][i])
                held = torch.cat([d[i-1:i] for d in drives], dim=0)
                count = int(np.floor(elapsed/nominal))
                for _ in range(count):
                    state = base.model.backbone._advance_elapsed_tensor(held, state, nominal)
                tail = elapsed-count*nominal
                if tail > 1e-12:
                    state = base.model.backbone._advance_elapsed_tensor(held, state, tail)
            predictions.append(base.predict(state)['canonical'].cpu().numpy())
    predictions = np.stack(predictions)
    reference = np.array([r['prediction']['canonical'][0] for r in
                          fitted['fit']['final_fixed_rollout']['rows']])
    parity = float(np.max(np.abs(predictions[:, 0]-reference)))
    if parity > 1e-6:
        raise ValueError('Original full-history output does not reproduce')
    full = a['native_pedals'][:, 1] == 1
    age = np.zeros(history.observations)
    start = None
    for i in range(history.observations):
        if full[i]:
            if start is None:
                start = a['decision_clock'][i]
            age[i] = a['decision_clock'][i]-start
        else:
            start = None
    eligible = full & history.effective_loss_valid[:, 1]
    bins = [(0., .1), (.1, .5), (.5, 1.), (1., 3.), (3., 100.)]
    reports = []
    for lo, hi in bins:
        mask = eligible & (age >= lo) & (age < hi)
        row = {'age_seconds': [lo, hi], 'eligible_rows': int(mask.sum()), 'conditions': {}}
        for j, name in enumerate(['original', 'pedal_only', 'brake_steer_anchored']):
            brake = predictions[mask, j, 2]
            row['conditions'][name] = {'brake_below_0_1': int((brake < .1).sum()),
                'brake_rmse_to_one': float(np.sqrt(np.mean((1-brake)**2))) if len(brake) else None,
                'mean_predicted_brake': float(brake.mean()) if len(brake) else None}
        reports.append(row)
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError('Original evidence changed during audit')
    write_json(run/'result.json', {'status': 'completed_train_brake_duration_audit',
        'reports': reports, 'eligible_fullbrake_rows': int(eligible.sum()),
        'raw_fullbrake_runs': int((full & np.r_[True, ~full[:-1]]).sum()),
        'max_raw_fullbrake_duration_seconds': float(age.max()),
        'original_replay_max_abs_difference': parity,
        'bins_are_posthoc_not_qualification_gates': True, 'train_only_not_independent': True,
        'pedal_targets_not_input_to_actor': True, 'optimizer_updates': 0,
        'actor_deployed': False, 'goal_achieved': False})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
