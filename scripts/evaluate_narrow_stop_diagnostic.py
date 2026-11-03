"""Replay a fixed brake-only stop-vector candidate on old diagnostic history."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_pedal_learning_v2 import native_pedal_supervision
from flybrain_drive.flight_stop_narrow_adapter import ExperimentalNarrowStopCircuit
from flybrain_drive.graph import sha256
from scripts.train_pedal_b2 import OriginalGraphAdapter


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-plan', required=True)
    parser.add_argument('--fit-result', required=True)
    parser.add_argument('--pedal-checkpoint', required=True)
    parser.add_argument('--stop-fit', required=True)
    parser.add_argument('--diagnostic', required=True)
    parser.add_argument('--body-replay', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan, fitted, pedal = read(args.fit_plan), read(args.fit_result), read(args.pedal_checkpoint)
    diagnostic, replay = Path(args.diagnostic), Path(args.body_replay)
    diag_result, body_result = read(diagnostic/'result.json'), read(replay/'result.json')
    stop_fit = Path(args.stop_fit)
    stop_result = read(stop_fit/'result.json')
    if (diag_result['role'] != 'diagnostic' or diag_result['fit_allowed'] is not False
            or diag_result['actor_trace']['decisions'] != 986
            or body_result['status'] != 'completed'
            or sha256(replay/'virtual-body.npz') != body_result['virtual_body_sha256']
            or fitted['status'] != 'completed_one_pass_pedal_component_fit'
            or pedal['bindings'] != fitted['bindings']):
        raise ValueError('Original unpromoted diagnostic and original physical fit required')
    if (stop_result['status'] != 'completed_fixed_stop_strength_selection'
            or sha256(stop_fit/'stop-task-adapter.pt') != stop_result['checkpoint_sha256']):
        raise ValueError('Original fixed stop-task candidate required')
    recording = diagnostic/'recording'
    event_path = recording/'events.jsonl'
    with event_path.open(encoding='utf8') as stream:
        events = [json.loads(line) for line in stream]
    with np.load(replay/'virtual-body.npz', allow_pickle=False) as a:
        body, valid = a['values'].copy(), a['valid'].copy()
        clocks, packets = a['decision_monotonic_seconds'].copy(), a['native_packet_ids'].copy()
    if (len(events) != 986 or body.shape != (986, 21) or valid.dtype != bool
            or not valid[:, [4, 16]].all()
            or np.any((np.diff(clocks) <= 0) | (np.diff(clocks) > .08))):
        raise ValueError('Complete original diagnostic history and intervals required')
    commands = []
    for i, event in enumerate(events):
        task = event['external_task_at_receipt']
        native = event['native']
        if (event['decision_monotonic_seconds'] != clocks[i] or native['packet_id'] != packets[i]
                or task['schema'] != 'flybrain.external_speed_task.v1'
                or task['held_external_instruction'] is not True or task['actor_input'] is not True
                or not 0 <= task['observed_at_host_seconds'] <= native['received_monotonic_seconds']
                <= clocks[i] or not 0 <= task['desired_speed_m_s'] <= 100/3.6+1e-5):
            raise ValueError('Original sensor clocks and held external task required')
        commands.append([0., task['desired_speed_m_s']])
    paths = [args.fit_plan, args.fit_result, args.pedal_checkpoint, str(diagnostic/'result.json'),
        str(event_path), str(replay/'result.json'), str(replay/'virtual-body.npz')]
    paths.extend([str(stop_fit/'result.json'), str(stop_fit/'stop-task-adapter.pt')])
    pins = {path: sha256(path) for path in paths}
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'narrow_gate_source_sha256': sha256('src/flybrain_drive/flight_stop_narrow_adapter.py'),
        'source_hashes': pins,
        'role': 'original_1903_diagnostic_only', 'fit_allowed': False, 'optimizer_updates': 0,
        'student_deployed': False, 'teacher_pedal_bytes_decoded_after_prediction': True})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    base = OriginalGraphAdapter(plan, lambda: None)
    with torch.no_grad():
        for name, parameter in base.parameters().items():
            parameter.copy_(torch.tensor(pedal['values'][name], device=parameter.device))
        quiet = base.initialize()
    if base.fixed_fingerprint() != fitted['fit']['final_fixed_rollout']['fixed_fingerprint']:
        raise ValueError('Original fixed flight graph/readout changed')
    base.model.requires_grad_(False)
    saved = torch.load(stop_fit/'stop-task-adapter.pt', map_location='cpu', weights_only=True)
    if saved['source_pedal_checkpoint_sha256'] != sha256(args.pedal_checkpoint):
        raise ValueError('Original pedal model binding changed')
    wrapped = ExperimentalNarrowStopCircuit(base.model).to('cuda').eval()
    wrapped.adapter.load_state_dict(saved['model'], strict=True)
    wrapped.requires_grad_(False)
    body_t = torch.tensor(body, dtype=torch.float32, device='cuda')
    valid_t = torch.tensor(valid, dtype=torch.bool, device='cuda')
    commands_t = torch.tensor(commands, dtype=torch.float32, device='cuda')
    goals = torch.tensor([[0., 1.]], dtype=torch.float32, device='cuda').repeat(986, 1)
    with torch.no_grad():
        original_drive = base.model.drive(commands_t, body_t, valid_t, goals, **base.kwargs)
        drives = [original_drive, wrapped.drive(commands_t, body_t, valid_t, goals, **base.kwargs)]
        state = quiet.detach().expand(2, -1).clone()
        output, power_output = [], []
        nominal = base.model.backbone.spec['dynamics']['dt']
        for i in range(986):
            if i:
                elapsed = float(clocks[i]-clocks[i-1])
                step = torch.cat([d[i-1:i] for d in drives], 0)
                count = int(np.floor(elapsed/nominal))
                for _ in range(count):
                    state = base.model.backbone._advance_elapsed_tensor(step, state, nominal)
                tail = elapsed-count*nominal
                if tail > 1e-12:
                    state = base.model.backbone._advance_elapsed_tensor(step, state, tail)
            prediction = base.predict(state)
            output.append(prediction['canonical'].cpu().numpy())
            power_output.append(prediction['common_power'].cpu().numpy())
    predictions = np.stack(output)
    power = np.stack(power_output)
    if not np.isfinite(predictions).all():
        raise ValueError('Finite diagnostic predictions required')
    # Teacher pedal bytes are decoded only after all actor predictions are fixed.
    targets, eligible = [], []
    for i, event in enumerate(events):
        native, auth = event['native'], event['native_teacher_authority']
        raw = base64.b64decode(native['prefix_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != native['prefix_sha256']:
            raise ValueError('Original native body prefix hash differs')
        sample = decode_physics_prefix(raw, native['received_monotonic_seconds'])
        if sample.packet_id != packets[i] or auth['body_prefix_sha256'] != native['prefix_sha256']:
            raise ValueError('Original same-copy native identity mismatch')
        targets.append([sample.fields['gas'], sample.fields['brake']])
        eligible.append(auth['teacher_authority_eligible'] is True
            and auth['same_native_copy'] is True and auth['builtin_ai_controlled'] is True
            and auth['actor_input'] is False and auth['packet_id'] == native['packet_id']
            and 0 <= clocks[i]-native['received_monotonic_seconds'] <= .05)
    target = torch.tensor(targets, dtype=torch.float32, device='cuda')
    mask = torch.tensor(np.repeat(np.asarray(eligible)[:, None], 2, axis=1), device='cuda')
    supervision = native_pedal_supervision(target, mask, projection_policy='reject_unreachable')
    keep = supervision['supervision_valid']
    target_np = target.cpu().numpy()
    full_age = np.zeros(len(events), dtype=np.float64)
    start_time = None
    for i, brake in enumerate(target_np[:, 1]):
        if brake == 1:
            if start_time is None:
                start_time = clocks[i]
            full_age[i] = clocks[i] - start_time
        else:
            start_time = None
    reports = []
    for index, label in enumerate(['original', 'narrow_gate_512_stop_candidate']):
        canonical = torch.tensor(predictions[:, index], device='cuda')
        error = ((supervision['canonical_lower']-canonical[:, 1:]).relu()
                 + (canonical[:, 1:]-supervision['canonical_upper']).relu())
        counts = keep.sum(0)
        sums = (error.square()*keep).sum(0)
        full = (target[:, 1] == 1) & keep[:, 1]
        sustained = full & torch.tensor(full_age >= 3., device='cuda')
        p0 = base.binding.reference.quiet_power
        span = base.binding.reference.span
        unclipped = (p0 - torch.tensor(power[:, index], device='cuda'))/span
        reports.append({'condition': label, 'eligible_counts': counts.cpu().tolist(),
            'pedal_mse': (sums/counts.clamp_min(1)).cpu().tolist(),
            'full_brake_count': int(full.sum()),
            'full_brake_mse': float(error[full, 1].square().mean()) if full.any() else None,
            'sustained_full_brake_over_3s_count': int(sustained.sum()),
            'sustained_full_brake_over_3s_mean_prediction': (
                float(canonical[sustained, 2].mean()) if sustained.any() else None),
            'sustained_full_brake_over_3s_unclipped_mean': (
                float(unclipped[sustained].mean()) if sustained.any() else None),
            'sustained_full_brake_over_3s_below_brake_boundary': (
                int((unclipped[sustained] <= 0).sum())),
            'mean_abs_steering_change_from_original': float(np.abs(
                predictions[:, index, 0]-predictions[:, 0, 0]).mean())})
    if any(sha256(path) != digest for path, digest in pins.items()):
        raise ValueError('Original source changed during diagnostic')
    np.savez_compressed(run/'predictions.npz', canonical=predictions, full_brake_age=full_age,
                        common_power=power)
    write_json(run/'result.json', {'status': 'completed_original_1903_diagnostic_only',
        'reports': reports, 'observations': 986, 'teacher_authority_rows': int(sum(eligible)),
        'predictions_sha256': sha256(run/'predictions.npz'),
        'teacher_targets_decoded_only_after_predictions': True, 'fit_allowed': False,
        'diagnostic_not_geographically_independent': True,
        'teacher_camera_and_simulation_time_alignment_not_verified': True,
        'no_optimizer_or_actor_deployment': True, 'goal_achieved': False})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
