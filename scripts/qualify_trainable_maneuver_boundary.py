"""CPU command-boundary gradients through the original measured flight graph; no driving labels."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.flight_maneuver_learning import (
    LEARNING_BOUNDARY_CONTRACT,
    TrainablePhysicalManeuverCircuit,
    applied_axis_supervision,
    canonical_actuator_tensors,
)
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT, racing_goal


def tensors_bitwise_equal(a, b):
    """Compare original sparse storage explicitly; torch.equal lacks SparseCPU kernels."""
    if a.shape != b.shape or a.dtype != b.dtype or a.layout != b.layout:
        return False
    if a.layout == torch.strided:
        return torch.equal(a, b)
    if a.layout == torch.sparse_coo:
        return (a.is_coalesced() == b.is_coalesced()
                and torch.equal(a._indices(), b._indices())
                and torch.equal(a._values(), b._values()))
    if a.layout == torch.sparse_csr:
        return (torch.equal(a.crow_indices(), b.crow_indices())
                and torch.equal(a.col_indices(), b.col_indices())
                and torch.equal(a.values(), b.values()))
    raise TypeError('Uninspected original circuit tensor storage: '+str(a.layout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    if (plan['schema'] != 'flybrain.flight.motor_boundary_cpu_qualification.v1'
            or plan['optimizer_steps'] != 1 or plan['diagnostic_learning_rate'] != .001
            or plan['diagnostic_objective'] != 'fixed weighted squared original motor activity'
            or plan['new_boundary_contract'] != LEARNING_BOUNDARY_CONTRACT):
        raise ValueError('Exact diagnostic-only single-step CPU qualification required')
    for name, digest in plan['input_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Declared qualification source changed: '+name)
    run = begin_run(args.output, dict(plan=plan, plan_sha256=sha256(args.plan),
        program_sha256=sha256(__file__), scope=__doc__, gpu_used=False,
        runtime_mappings_opened=0, control_messages_sent=0, driving_training_labels_used=0))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', gpu_used=False, runtime_mappings_opened=0,
        control_messages_sent=0, optimizer_updates=0, driving_training_updates=0,
        optimal_pedal_labels_fabricated=0, source_payloads_copied=0, actor_deployed=False,
        native_target_data_used_for_fitting=False, goal_achieved=False)
    try:
        wall = time.perf_counter()
        torch.set_num_threads(1)
        torch.set_default_device('cpu')
        torch.manual_seed(plan['seed'])
        payload = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False)
        old = PhysicalManeuverCircuit(plan['spec'], payload, sparse_backend='cached_csr_cpu')
        new = TrainablePhysicalManeuverCircuit(plan['spec'], payload)
        old.eval()
        new.eval()
        if (type(old).drive is not type(new).drive
                or type(old).advance is not type(new).advance):
            raise ValueError('Original physical drive/advance must remain the implementation')
        frozen = {k: v.detach().clone() for k, v in new.backbone.state_dict().items()}
        if any(p.requires_grad for p in new.backbone.parameters()):
            raise ValueError('Original backbone must be frozen')
        kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        goals = torch.tensor(racing_goal(100, 1))
        quiet = torch.zeros(1, 21)
        valid = torch.zeros_like(quiet, dtype=torch.bool)
        valid[:, [0, 1, 4, 16]] = True
        zero_command = torch.zeros(1, 2)
        events = [json.loads(line) for line in Path(plan['original_events']).read_text(
            encoding='utf8').splitlines()]
        rows = [e for e in events if e['kind'] == 'neural_component']
        if len(rows) != 68:
            raise ValueError('Declared original68-observation component history required')
        original_body = np.asarray([r['body'] for r in rows])
        original_masks = np.asarray([r['body_valid'] for r in rows])
        commands = torch.tensor([r['physical_command'] for r in rows], dtype=torch.float32)
        body = torch.tensor(original_body, dtype=torch.float32)
        masks = torch.tensor(original_masks, dtype=torch.bool)
        with torch.no_grad():
            drive_old = old.drive(commands, body, masks, torch.tensor(racing_goal(100, len(rows))),
                                  **kwargs)
            drive_new = new.drive(commands, body, masks, torch.tensor(racing_goal(100, len(rows))),
                                  **kwargs)
            if not torch.equal(drive_old, drive_new):
                raise ValueError('Zero-init actual-input drive must match old bitwise')
            old_state, new_state = old.initial_state(1), new.initial_state(1)
            for _ in range(100):
                _, _, old_state = old.advance(zero_command, quiet, valid, goals,
                                               old_state, .01, **kwargs)
                _, _, new_state = new.advance(zero_command, quiet, valid, goals,
                                               new_state, .01, **kwargs)
            for i, row in enumerate(rows):
                if i:
                    elapsed = row['decision_clock']-rows[i-1]['decision_clock']
                    if elapsed != row['original_elapsed_seconds'] or not 0 < elapsed <= .08:
                        raise ValueError('Original previous-held interval required')
                    _, _, old_state = old.advance(commands[i-1:i], body[i-1:i], masks[i-1:i],
                                                  goals, old_state, elapsed, **kwargs)
                    _, _, new_state = new.advance(commands[i-1:i], body[i-1:i], masks[i-1:i],
                                                  goals, new_state, elapsed, **kwargs)
                if not torch.equal(old_state, new_state):
                    raise ValueError('Zero-init full actual-clock recurrent replay differs')
            # These are engineering high-speed/turn injections, not new native
            # observations or known valid driving outputs.
            diagnostic_body = torch.zeros(4, 21)
            diagnostic_body[:, 16] = 60/3.6
            diagnostic_valid = torch.zeros_like(diagnostic_body, dtype=torch.bool)
            diagnostic_valid[:, [0, 1, 4, 16]] = True
            diagnostic_commands = torch.tensor([[.3, 100/3.6], [-.3, 100/3.6],
                                                 [.3, 20/3.6], [-.3, 20/3.6]])
            diagnostic_goals = torch.tensor(racing_goal(100, 4))
            state = new.initial_state(4)
            for _ in range(100):
                _, _, state = new.advance(torch.zeros(4, 2), torch.zeros(4, 21),
                    diagnostic_valid, diagnostic_goals, state, .01, **kwargs)
            initial_state = state.detach()
            old_current = old.drive(diagnostic_commands, diagnostic_body, diagnostic_valid,
                                     diagnostic_goals, **kwargs)
            if not torch.equal(old_current, new.drive(diagnostic_commands, diagnostic_body,
                diagnostic_valid, diagnostic_goals, **kwargs)):
                raise ValueError('Synthetic60/100 old-drive initialization parity failed')
        parameters = list(new.command.parameters())
        if sum(p.numel() for p in parameters) != 4:
            raise ValueError('Exactly four learnable command-boundary scalars required')
        optimizer = torch.optim.Adam(parameters, lr=plan['diagnostic_learning_rate'])
        optimizer.zero_grad(set_to_none=True)
        intent, muscles, _ = new.advance(diagnostic_commands, diagnostic_body,
            diagnostic_valid, diagnostic_goals, initial_state, .05, **kwargs)
        before_intent = intent.detach().clone()
        jacobian = []
        for value in intent.flatten():
            gradients = torch.autograd.grad(value, parameters, retain_graph=True)
            jacobian.append(torch.cat([g.flatten() for g in gradients]).detach())
        jacobian = torch.stack(jacobian).numpy()
        muscle_weight = torch.tensor([.3, .5, .7, .9, 1.1, 1.3])
        diagnostic = (muscles.square()*muscle_weight).mean()
        codec = canonical_actuator_tensors(intent)
        objective = diagnostic+.01*codec['feasibility_loss']
        objective.backward()
        gradients = {k: p.grad.detach().clone() for k, p in new.command.named_parameters()}
        if not all(torch.isfinite(g).all() and torch.count_nonzero(g) == g.numel()
                   for g in gradients.values()):
            raise ValueError('All four coefficients require finite nonzero graph gradients')
        if any(p.grad is not None for p in new.backbone.parameters()):
            raise ValueError('Frozen graph/body/readout received a gradient')
        before_parameters = {k: p.detach().clone() for k, p in new.command.named_parameters()}
        optimizer.step()
        result['optimizer_updates'] = 1
        changed = {k: int(torch.count_nonzero(p-before_parameters[k]))
                   for k, p in new.command.named_parameters()}
        with torch.no_grad():
            after_intent, _, _ = new.advance(diagnostic_commands, diagnostic_body,
                diagnostic_valid, diagnostic_goals, initial_state, .05, **kwargs)
            frozen_exact = all(tensors_bitwise_equal(value, new.backbone.state_dict()[key])
                               for key, value in frozen.items())
            if not frozen_exact or any(count == 0 for count in changed.values()):
                raise ValueError('Diagnostic update failed or changed original circuit values')
            if torch.equal(before_intent, after_intent):
                raise ValueError('Learnable DN boundary did not change original graph response')
            neutral = new.drive(zero_command, quiet, valid, goals, **kwargs)
            if not torch.equal(neutral, old.drive(zero_command, quiet, valid, goals, **kwargs)):
                raise ValueError('Learned coefficients introduced new quiet-body drive')
        native_targets = torch.tensor([[0., 0., .016], [0., .005, 0.], [0., .03, .04]])
        label_mask = torch.ones_like(native_targets, dtype=torch.bool)
        rejected = applied_axis_supervision(native_targets, label_mask,
                                            steering_gain=1.406249850988388)
        projected = applied_axis_supervision(native_targets, label_mask,
            steering_gain=1.406249850988388, projection_policy='nearest_reachable')
        # Save only four derived calibration coefficients and buffers; reuse the
        # original graph/checkpoint by sealed reference, never duplicate it.
        torch.save(dict(schema='flybrain.flight.boundary_diagnostic_not_driving_candidate.v1',
            command=new.command.state_dict(), source_checkpoint_sha256=sha256(plan['checkpoint']),
            source_spec_sha256=sha256(plan['spec']), contract=LEARNING_BOUNDARY_CONTRACT,
            optimizer_updates=1, driving_training_updates=0, actor_policy=False),
            run/'diagnostic-boundary-only.pt')
        result.update(status='completed_cpu_trainable_motor_boundary_qualification',
            old_current_initialization_bitwise_exact=True,
            original_actual_input_observations=68, original_actual_intervals=67,
            original_elapsed_seconds=rows[-1]['decision_clock']-rows[0]['decision_clock'],
            original_body_and_masks_unchanged=True, original_clock_replay_state_exact=True,
            synthetic_commands=diagnostic_commands.tolist(),
            synthetic_body_valid=diagnostic_valid.tolist(),
            synthetic_current_initialization_bitwise_exact=True,
            original_graph_body_readout_bitwise_unchanged=frozen_exact,
            boundary_parameter_elements=4, changed_boundary_elements=changed,
            actual_graph_gradients={k: v.tolist() for k, v in gradients.items()},
            synthetic_intent_jacobian=jacobian.tolist(),
            synthetic_intent_jacobian_rank=int(np.linalg.matrix_rank(jacobian)),
            synthetic_intent_jacobian_singular_values=np.linalg.svd(jacobian,
                compute_uv=False).tolist(), diagnostic_scalar=float(diagnostic.detach()),
            codec_feasibility_scalar=float(codec['feasibility_loss'].detach()),
            intent_before=before_intent.tolist(), intent_after=after_intent.tolist(),
            learned_quiet_drive_matches_old=True,
            final_coefficients={k: v.detach().tolist()
                                for k, v in new.command.coefficients().items()},
            source_minmax_actual_speed_m_s=[float(original_body[:, 16].min()),
                                           float(original_body[:, 16].max())],
            teacher_target_policy_example=dict(native_target=native_targets.tolist(),
                reject_supervision_mask=rejected['supervision_valid'].tolist(),
                declared_projected_target=projected['target_for_supervision'].tolist(),
                declared_projected_mask=projected['supervision_valid'].tolist()),
            wall_seconds=time.perf_counter()-wall,
            limitations=['Diagnostic objective is not a driving task or optimal control label.',
                'Synthetic neuronal identifiability is not vehicle plant identification.',
                't1689 student1m/s component data is numerical evidence, not teacher fitting data.',
                'No load/tyre/wheelrate causal route has been added.',
                'New diverse sourceTRAIN teacher/native motion/control data is still required.'])
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        raise
    finally:
        write_json(run/'result.json', result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('synthetic_intent_jacobian', 'input_sha256')}))


if __name__ == '__main__':
    main()
