"""Bounded CPU-only original-graph pedal gradients; no optimizer or driving data."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.flight_embodiment import decode_embodiment
from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from flybrain_drive.flight_maneuver_learning import TrainablePhysicalManeuverCircuit
from flybrain_drive.flight_pedal_learning_v2 import (
    bind_registered_reference,
    install_registered_private_offsets,
    native_pedal_supervision,
    pedal_only_parameters,
    v2_pedal_imitation_loss,
    v2_pedal_tensors,
)
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT

CONDITIONS = [[0, 100], [60, 100], [100, 60], [60, 20]]
BASIS = [[1., 0.], [0., 1.]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {k: safe(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(v) for v in item]
        return str(item) if isinstance(item, float) and not math.isfinite(item) else item
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False)+'\n', encoding='utf8')


def tensor_digest(value):
    """Same dtype/shape/byte fingerprint as executed1734, extended for CSR storage."""
    value = value.detach().cpu()
    if value.layout == torch.sparse_coo:
        return tensor_digest(value._indices())+tensor_digest(value._values())
    if value.layout == torch.sparse_csr:
        return (tensor_digest(value.crow_indices())+tensor_digest(value.col_indices())
                +tensor_digest(value.values()))
    value = np.ascontiguousarray(value.numpy())
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()
                          +value.tobytes()).hexdigest()


def model_digest(model):
    return {k: tensor_digest(v) for k, v in model.backbone.state_dict().items()}


def rank_report(rows):
    """SVD in float64; threshold respects original float32 gradient precision."""
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or len(matrix) == 0:
        raise ValueError('Nonempty fixed two-column diagnostic matrix required')
    if not np.isfinite(matrix).all():
        return dict(shape=list(matrix.shape), matrix=matrix.tolist(), finite=False,
            singular_values=None, threshold=None, rank=None, both_columns_nonzero=None)
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = max(1e-12, max(matrix.shape)*np.finfo(np.float32).eps*singular[0])
    return dict(shape=list(matrix.shape), matrix=matrix.tolist(),
        singular_values=singular.tolist(), threshold=float(threshold),
        rank=int(np.count_nonzero(singular > threshold)),
        both_columns_nonzero=[bool(np.any(matrix[:, i] != 0)) for i in range(2)],
        finite=True)


def load_plan(path):
    plan = json.loads(Path(path).read_bytes())
    if (plan['schema'] != 'flybrain.flight.original_graph_pedal_gradient_cpu.v1'
            or plan['conditions_goal_body_kmh'] != CONDITIONS
            or plan['diagnostic_pedal_target_basis'] != BASIS
            or plan['quiet_steps'] != 100 or plan['quiet_dt_seconds'] != .01
            or plan['step_endpoint_seconds'] != .05
            or plan['body_valid_indices'] != [0, 1, 4, 16]
            or plan['device'] != 'cpu' or plan['cpu_threads'] != 1
            or plan['optimizer_updates'] != 0 or plan['fit_allowed'] is not False
            or plan['wall_budget_seconds'] != 120
            or plan['rank_absolute_floor'] != 1e-12
            or plan['rank_relative_rule'] != 'max(matrix.shape)*float32_epsilon*smax'):
        raise ValueError('Exact preregistered bounded CPU diagnostic required')
    for name, digest in plan['input_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Registered source differs: '+name)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if not args.execute:
        print(json.dumps(dict(status='source_registration_verified_NO_model_loaded',
            input_count=len(plan['input_sha256']), plan_sha256=sha(args.plan))))
        return
    if not args.output:
        raise ValueError('New explicit output required')
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=False)
    write(run/'registration-before-model.json', dict(plan=plan,
        plan_sha256=sha(args.plan), executed_program_sha256=sha(__file__)))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    result = dict(status='failed', fit_allowed=False, optimizer_updates=0,
        teacher_labels_used=False, diagnostic_basis_is_not_teacher_or_optimal_control=True,
        source_payloads_copied=0, gpu_initialized=False, live_actions=0,
        cases=[], original_graph_pedal_gradient_qualified=False)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_device('cpu')
        torch.manual_seed(plan['seed'])
        binding = bind_registered_reference(Path(plan['cpu_reference']).read_bytes(),
                                             Path(plan['cpu_readout']).read_bytes())
        payload = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False)
        model = TrainablePhysicalManeuverCircuit(plan['spec'], payload,
                                                 sparse_backend='cached_csr_cpu').eval()
        if (type(model).advance is not PhysicalManeuverCircuit.advance
                or type(model).drive is not PhysicalManeuverCircuit.drive):
            raise ValueError('Original physical drive/advance implementations required')
        del payload
        model.command.turn_delta.requires_grad_(False)
        parameters = pedal_only_parameters(model)
        initial_parameters = {k: tensor_digest(v) for k, v in model.named_parameters()}
        original_backbone = model_digest(model)
        kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        goals = torch.tensor([[0., 1.]])
        valid = torch.zeros(1, 21, dtype=torch.bool)
        valid[:, plan['body_valid_indices']] = True
        zero_command, quiet_body = torch.zeros(1, 2), torch.zeros(1, 21)
        frozen, loss_rows, output_rows = None, [], []

        def check_budget():
            if time.perf_counter()-started > plan['wall_budget_seconds']:
                raise TimeoutError('Original fixed CPU diagnostic wall budget exhausted')

        for index, (goal_kmh, body_kmh) in enumerate(CONDITIONS):
            check_budget()
            # Independent case boundary; no reset inside quiet->step trajectory.
            with torch.no_grad():
                state = model.initial_state(1)
                for _ in range(plan['quiet_steps']):
                    check_budget()
                    _, _, state = model.advance(zero_command, quiet_body, valid, goals,
                                                state, .01, **kwargs)
                quiet_hash = tensor_digest(state)
                if quiet_hash != binding.reference.provenance.quiet_end_state_sha256:
                    raise ValueError('Actual CPU quiet state differs; never reanchor')
                offset_receipt = install_registered_private_offsets(model.backbone.readout,
                    binding, quiet_state_sha256=quiet_hash)
                quiet_raw, quiet_mn = model.backbone._read_state_tensor(state)
                quiet_decoded = decode_embodiment(quiet_raw, quiet_mn,
                                                  reference=binding.reference)
                if (not torch.equal(quiet_raw, torch.zeros_like(quiet_raw))
                        or quiet_decoded.canonical_v2[0, 2].item() != 0):
                    raise ValueError('Actual private quiet forward differs')
                private = model_digest(model)
                changed = {k for k in original_backbone if original_backbone[k] != private[k]}
                if not changed <= {'readout.power_reference', 'readout.lateral_reference',
                                   'readout.reference_calibrated'}:
                    raise ValueError('Only exact private neutral buffers may change')
                if frozen is None:
                    frozen = private
                if private != frozen:
                    raise ValueError('Repeated private backbone differs')
                state = state.detach()
            body = quiet_body.clone()
            body[:, 16] = body_kmh/3.6
            command = torch.tensor([[0., goal_kmh/3.6]])
            # Same original advance implementation and original cached CSR adjoint.
            intent, mn, endpoint = model.advance(command, body, valid, goals, state, .05, **kwargs)
            prediction = v2_pedal_tensors(intent, mn, binding=binding)
            case = dict(index=index, desired_speed_kmh=goal_kmh, body_speed_kmh=body_kmh,
                desired_yaw_rad_s=0., body_valid_indices=plan['body_valid_indices'],
                quiet_state_sha256=quiet_hash, endpoint_state_sha256=tensor_digest(endpoint),
                intermediate_state_resets=0, endpoint_seconds=.05,
                raw_intent=intent.detach().tolist(), mn6=mn.detach().tolist(),
                common_power=prediction['common_power'].detach().tolist(),
                canonical_v2=prediction['canonical'].detach().tolist(),
                encoded=prediction['encoded'].detach().tolist(),
                fixed_anchor=binding.reference.brake_anchor_power,
                offset_installation=offset_receipt, diagnostic_losses=[],
                pedal_jacobian=[], original_advance_implementation=True)
            for target in BASIS:
                supervision = native_pedal_supervision(torch.tensor([target]),
                                                       torch.ones(1, 2, dtype=torch.bool))
                loss = v2_pedal_imitation_loss(prediction, supervision)['loss']
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
                row = [g.item() for g in gradients]
                loss_rows.append(row)
                case['diagnostic_losses'].append(dict(target_basis=target, loss=loss.item(),
                    gradients=row, target_is_teacher_or_optimal=False))
            for axis in (1, 2):
                gradients = torch.autograd.grad(prediction['canonical'][0, axis], parameters,
                                                retain_graph=True)
                row = [g.item() for g in gradients]
                output_rows.append(row)
                case['pedal_jacobian'].append(row)
            if (model_digest(model) != frozen
                    or any(p.grad is not None for p in model.parameters())):
                raise ValueError('Frozen state changed or parameter grad buffer populated')
            result['cases'].append(case)
            write(run/'partial-result.json', result)
            del endpoint, prediction, intent, mn, state, loss, gradients
        check_budget()
        loss_rank, output_rank = rank_report(loss_rows), rank_report(output_rows)
        final_parameters = {k: tensor_digest(v) for k, v in model.named_parameters()}
        if initial_parameters != final_parameters or torch.cuda.is_initialized():
            raise ValueError('Parameter mutation or unexpected GPU initialization')
        for path, digest in plan['input_sha256'].items():
            if sha(path) != digest:
                raise ValueError('Source drift during diagnostic: '+path)
        result.update(status='completed_original_graph_CPU_pedal_gradient_diagnostic',
            loss_gradient_rank=loss_rank, pedal_output_jacobian_rank=output_rank,
            original_graph_pedal_gradient_qualified=(loss_rank['rank'] == 2
                and output_rank['rank'] == 2 and all(loss_rank['both_columns_nonzero'])),
            parameter_fingerprints_before=initial_parameters,
            parameter_fingerprints_after=final_parameters,
            protected_backbone_fingerprints=frozen,
            source_drift_count=0, quiet_cases=4, step_cases=4,
            scientific_scope='Synthetic fixed-input local neural capability, not dataset fit '
                'admission, identified vehicle/hold policy or learned driving')
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        result['wall_seconds'] = time.perf_counter()-started
        write(run/'result.json', result)
        print(json.dumps(dict(status=result['status'], result_sha256=sha(run/'result.json'),
            retained_cases=len(result['cases']), wall_seconds=result['wall_seconds'])))


if __name__ == '__main__':
    main()
