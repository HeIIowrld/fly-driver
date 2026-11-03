"""Matched original-graph CPU/HIP pedal gradients; model/device work requires --execute."""

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

CONDITIONS = [[0, 100], [60, 100], [100, 60], [60, 20]]
BASIS = [[1., 0.], [0., 1.]]
GATES = dict(state_absolute=1e-6, output_absolute=1e-4,
             gradient_absolute=1e-5, gradient_relative=1e-3)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {k: safe(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(v) for v in item]
        return str(item) if isinstance(item, float) and not math.isfinite(item) else item
    Path(path).write_text(json.dumps(safe(value), indent=2, allow_nan=False)+'\n', encoding='utf8')


def parity(cpu, gpu, *, absolute, relative=0.):
    cpu, gpu = np.asarray(cpu, np.float64), np.asarray(gpu, np.float64)
    finite = cpu.shape == gpu.shape and cpu.size > 0 and np.isfinite(cpu).all()
    finite = bool(finite and np.isfinite(gpu).all())
    if not finite:
        return dict(passed=False, finite=False, shape_cpu=list(cpu.shape),
                    shape_gpu=list(gpu.shape), maximum_absolute_error=None)
    error = abs(gpu-cpu)
    limits = absolute+relative*abs(cpu)
    return dict(passed=bool(np.all(error <= limits)), finite=True, shape=list(cpu.shape),
        maximum_absolute_error=float(error.max()),
        failing_cells=int(np.count_nonzero(error > limits)),
        absolute_tolerance=absolute, relative_tolerance=relative,
        reference='CPU', CPU_zero_cells=int(np.count_nonzero(cpu == 0)),
        GPU_nonzero_at_CPU_zero=int(np.count_nonzero((cpu == 0) & (gpu != 0))))


def graph_functions(value, *, maximum_nodes=10000):
    """Read autograd node identities only, without altering a tensor or gradient."""
    pending, seen, names = [value.grad_fn], set(), set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if len(seen) > maximum_nodes:
            raise ValueError('Autograd graph inspection cap exhausted')
        names.add(type(node).__name__)
        pending.extend(n for n, _ in node.next_functions)
    return dict(node_count=len(seen), types=sorted(names),
        original_CSR_adjoint_present='_FixedCSRMessageBackward' in names)


def load_plan(path):
    plan = json.loads(Path(path).read_bytes())
    if (plan.get('schema') != 'flybrain.flight.matched_original_graph_pedal_gradient.v1'
            or plan.get('conditions_goal_body_kmh') != CONDITIONS
            or plan.get('diagnostic_pedal_target_basis') != BASIS or plan.get('gates') != GATES
            or plan.get('quiet_steps') != 100 or plan.get('quiet_dt_seconds') != .01
            or plan.get('endpoint_seconds') != .05 or plan.get('cpu_threads') != 1
            or plan.get('body_valid_indices') != [0, 1, 4, 16]
            or plan.get('wall_budget_seconds') != 180
            or plan.get('optimizer_updates') != 0 or plan.get('fit_allowed') is not False
            or plan.get('expected_torch') != '2.9.1+rocm7.2.1'
            or plan.get('expected_device') != 'AMD Radeon RX 9070 XT'
            or plan.get('expected_hip') != '7.2.53211-158bd99533'):
        raise ValueError('Exact preregistered CPU/HIP diagnostic, no tolerance changes')
    for file, expected in plan['input_sha256'].items():
        if sha(file) != expected:
            raise ValueError('Registered input differs: '+file)
    return plan


def run(plan, output, plan_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'registration-before-model.json', dict(plan=plan, plan_sha256=sha(plan_path),
        program_sha256=sha(__file__), process_terminal_not_yet_observed=True))
    (output/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = dict(status='failed', fit_allowed=False, optimizer_updates=0,
        teacher_labels_used=False,
        source_payloads_copied=0, live_actions=0, models_loaded=False,
        device_queried=False, CPU_models_constructed=0, cases=[], all_parity_gates_passed=False,
        process_exit_verified_by_this_process=False)
    models, inputs, states = [], [], []
    model = payload = cpu_model = gpu_model = cpu_state = gpu_state = None
    started, cuda_started = time.perf_counter(), False
    try:
        # Lazy and failure-recorded: staging/tests import no torch/model/device API.
        import torch

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
        from scripts.qualify_flight_pedal_graph_v2 import model_digest, rank_report, tensor_digest

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(plan['seed'])
        result.update(torch_version=torch.__version__, hip_version=torch.version.hip,
                      cuda_version=torch.version.cuda)
        if torch.__version__ != plan['expected_torch'] or torch.version.hip != plan['expected_hip']:
            raise ValueError('Registered actual torch/HIP runtime differs')
        result['device_queried'] = True
        result['device_available'] = torch.cuda.is_available()
        if not result['device_available']:
            raise RuntimeError('No actual HIP device; no CPU fallback')
        cuda_started = True
        device = torch.cuda.get_device_name(0)
        result.update(device_queried=True, device=device, torch_version=torch.__version__,
            hip_version=torch.version.hip, cuda_version=torch.version.cuda,
            gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            gpu_allocated_before_models_bytes=torch.cuda.memory_allocated())
        if device != plan['expected_device']:
            raise ValueError('Registered GPU differs')
        torch.cuda.reset_peak_memory_stats()
        binding = bind_registered_reference(Path(plan['cpu_reference']).read_bytes(),
                                             Path(plan['cpu_readout']).read_bytes())
        original = json.loads(Path(plan['original_cpu_result']).read_bytes())
        kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        payload = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False)

        def budget():
            if time.perf_counter()-started > plan['wall_budget_seconds']:
                raise TimeoutError('Fixed diagnostic wall budget exhausted; no extension')

        def tensors(device, goal=0., speed=0.):
            body = torch.zeros(1, 21, device=device)
            body[:, 16] = speed/3.6
            valid = torch.zeros_like(body, dtype=torch.bool)
            valid[:, [0, 1, 4, 16]] = True
            return (torch.tensor([[0., goal/3.6]], device=device), body, valid,
                    torch.tensor([[0., 1.]], device=device))

        def advance_state(model, values, state, dt):
            command, body, valid, goals = values
            return model.advance(command, body, valid, goals, state, dt, **kwargs)[2]

        def cpu_staging_model():
            model = TrainablePhysicalManeuverCircuit(plan['spec'], payload,
                                                     sparse_backend='cached_csr_cpu').eval()
            result['CPU_models_constructed'] += 1
            result['models_loaded'] = True
            if (type(model).advance is not PhysicalManeuverCircuit.advance
                    or type(model).drive is not PhysicalManeuverCircuit.drive):
                raise ValueError('Original physical graph route required')
            model.command.turn_delta.requires_grad_(False)
            pedal_only_parameters(model)
            before = model_digest(model)
            with torch.no_grad():
                state = model.initial_state(1)
                quiet_inputs = tensors('cpu')
                for _ in range(100):
                    budget()
                    state = advance_state(model, quiet_inputs, state, .01)
                actual_CPU_hash = tensor_digest(state)
                receipt = install_registered_private_offsets(model.backbone.readout, binding,
                    quiet_state_sha256=actual_CPU_hash)
            after = model_digest(model)
            changed = {k for k in before if before[k] != after[k]}
            if not changed <= {'readout.power_reference', 'readout.lateral_reference',
                               'readout.reference_calibrated'}:
                raise ValueError('Only fixed CPU private neutral buffers may change')
            return model, dict(actual_CPU_quiet_sha256=actual_CPU_hash,
                CPU_only_offset_installation=receipt, changed_backbone_keys=sorted(changed))

        cpu_model, cpu_staging = cpu_staging_model()
        gpu_model, gpu_staging = cpu_staging_model()
        if model_digest(cpu_model) != model_digest(gpu_model):
            raise ValueError('Independent CPU-staged graph/private buffers differ')
        # Explicit allowed execution route only; no weight/state/readout recalibration.
        gpu_model.backbone.circuit.sparse_backend = 'cached_csr_gpu'
        result['GPU_move_started'] = True
        gpu_model = gpu_model.to('cuda')
        result['GPU_move_completed'] = True
        models = [cpu_model, gpu_model]
        result.update(models_loaded=True, CPU_staging=[cpu_staging, gpu_staging],
            GPU_route='same original chemical graph _FixedCSRMessage GPU CSR+transpose adjoint',
            CPU_hash_never_used_as_actual_GPU_quiet_fingerprint=True)
        payload = None
        frozen = [model_digest(m) for m in models]
        if frozen[0] != frozen[1]:
            raise ValueError('Device move changed original protected graph/readout values')
        parameter_before = [{k: tensor_digest(v) for k, v in m.named_parameters()} for m in models]
        all_loss, all_output = [[], []], [[], []]
        checks = []

        def descriptor(model, state):
            raw, mn = model.backbone._read_state_tensor(state)
            prediction = v2_pedal_tensors(raw, mn, binding=binding)
            return dict(state_sha256=tensor_digest(state), tensor_device=str(state.device),
                tensor_dtype=str(state.dtype), raw=raw.detach().cpu().tolist(),
                mn6=mn.detach().cpu().tolist(),
                canonical=prediction['canonical'].detach().cpu().tolist(),
                encoded=prediction['encoded'].detach().cpu().tolist(),
                common_power=prediction['common_power'].detach().cpu().tolist())

        def endpoint(model, values, state):
            command, body, valid, goals = values
            raw, mn, state = model.advance(command, body, valid, goals, state, .05, **kwargs)
            prediction = v2_pedal_tensors(raw, mn, binding=binding)
            parameters = pedal_only_parameters(model)
            row = descriptor(model, state)
            row.update(loss_basis=[], loss_gradients=[], pedal_jacobian=[],
                autograd=graph_functions(prediction['common_power']))
            if not row['autograd']['original_CSR_adjoint_present']:
                raise ValueError('Actual original chemical CSR autograd route absent')
            circuit = model.backbone.circuit
            row['cached_execution_route'] = dict(backend=circuit.sparse_backend,
                matrix_device=str(circuit._cached_matrix.device),
                transpose_device=str(circuit._cached_transpose.device),
                matrix_nnz=circuit._cached_matrix._nnz())
            for target in BASIS:
                supervision = native_pedal_supervision(torch.tensor([target], device=raw.device),
                    torch.ones(1, 2, dtype=torch.bool, device=raw.device))
                loss = v2_pedal_imitation_loss(prediction, supervision)['loss']
                gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
                row['loss_basis'].append(loss.item())
                row['loss_gradients'].append([g.item() for g in gradients])
            for axis in (1, 2):
                gradients = torch.autograd.grad(prediction['canonical'][0, axis], parameters,
                                                retain_graph=True)
                row['pedal_jacobian'].append([g.item() for g in gradients])
            row['individual_case_pedal_jacobian_rank'] = rank_report(row['pedal_jacobian'])
            return row, state.detach().cpu()

        for index, (goal, speed) in enumerate(CONDITIONS):
            budget()
            inputs = [tensors(d) for d in ('cpu', 'cuda')]
            with torch.no_grad():
                states = [m.initial_state(1) for m in models]
                quiet_error, quiet_nonfinite = 0., False
                for _ in range(100):
                    for i, model in enumerate(models):
                        budget()
                        states[i] = advance_state(model, inputs[i], states[i], .01)
                    check = parity(states[0].cpu().numpy(), states[1].cpu().numpy(), absolute=1e-6)
                    checks.append(check['passed'])
                    quiet_nonfinite |= not check['finite']
                    quiet_error = max(quiet_error, check['maximum_absolute_error'] or 0.)
                quiet = [descriptor(m, s) for m, s in zip(models, states, strict=True)]
            if quiet[0]['state_sha256'] != binding.reference.provenance.quiet_end_state_sha256:
                raise ValueError('Recomputed actual CPU quiet differs; no reanchor')
            for q in quiet:
                checks.append(q['encoded'][0][1:] == [0., 0.])
            inputs = [tensors(d, goal, speed) for d in ('cpu', 'cuda')]
            cpu_row, cpu_state = endpoint(models[0], inputs[0], states[0].detach())
            gpu_row, gpu_state = endpoint(models[1], inputs[1], states[1].detach())
            cpu_original = original['cases'][index]
            cpu_exact = (cpu_row['state_sha256'] == cpu_original['endpoint_state_sha256']
                and cpu_row['raw'] == cpu_original['raw_intent']
                and cpu_row['mn6'] == cpu_original['mn6']
                and cpu_row['canonical'] == cpu_original['canonical_v2']
                and cpu_row['encoded'] == cpu_original['encoded']
                and cpu_row['loss_basis'] == [d['loss'] for d in cpu_original['diagnostic_losses']]
                and cpu_row['loss_gradients'] == [d['gradients']
                    for d in cpu_original['diagnostic_losses']]
                and cpu_row['pedal_jacobian'] == cpu_original['pedal_jacobian'])
            comparisons = dict(endpoint_state=parity(cpu_state.numpy(), gpu_state.numpy(),
                absolute=1e-6), CPU_recomputed_matches_original1820_exactly=cpu_exact)
            checks.append(cpu_exact)
            for key in ('raw', 'mn6', 'canonical', 'encoded', 'common_power', 'loss_basis'):
                comparisons[key] = parity(cpu_row[key], gpu_row[key], absolute=1e-4)
                comparisons['quiet_'+key] = (parity(quiet[0][key], quiet[1][key], absolute=1e-4)
                    if key in quiet[0] else None)
            for key in ('loss_gradients', 'pedal_jacobian'):
                comparisons[key] = parity(cpu_row[key], gpu_row[key],
                                           absolute=1e-5, relative=1e-3)
            checks.extend(v['passed'] for v in comparisons.values() if isinstance(v, dict))
            for i, row in enumerate((cpu_row, gpu_row)):
                all_loss[i].extend(row['loss_gradients'])
                all_output[i].extend(row['pedal_jacobian'])
            result['cases'].append(dict(index=index, desired_body_kmh=[goal, speed],
                quiet_actual=quiet, maximum_state_error_over_all100quiet_steps=(
                    None if quiet_nonfinite else quiet_error),
                quiet_nonfinite_observed=quiet_nonfinite,
                CPU=cpu_row, GPU=gpu_row, comparisons=comparisons))
            if any(model_digest(m) != f for m, f in zip(models, frozen, strict=True)):
                raise ValueError('Frozen graph/body/readout changed')
            write(output/'partial-result.json', result)
        budget()
        ranks = dict(loss=[rank_report(v) for v in all_loss],
                     pedal_output=[rank_report(v) for v in all_output])
        rank2 = all(r['rank'] == 2 for group in ranks.values() for r in group)
        parameter_after = [{k: tensor_digest(v) for k, v in m.named_parameters()} for m in models]
        if parameter_before != parameter_after or any(p.grad is not None
                for m in models for p in m.parameters()):
            raise ValueError('Parameter mutation or unexpected parameter grad buffer')
        for file, expected in plan['input_sha256'].items():
            if sha(file) != expected:
                raise ValueError('Source drift: '+file)
        torch.cuda.synchronize()
        result.update(status='completed_matched_original_graph_pedal_gradient_diagnostic',
            all_parity_gates_passed=bool(all(checks)), both_devices_global_rank2=rank2, ranks=ranks,
            scoped_qualification_passed=bool(all(checks) and rank2), source_drift_count=0,
            parameter_fingerprints_before=parameter_before,
            parameter_fingerprints_after=parameter_after,
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            full_neural_states_serialized=False, intermediate_state_resets=0,
            actual_GPU_quiet_hashes=[c['quiet_actual'][1]['state_sha256'] for c in result['cases']])
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, reason=str(error))
        raise
    finally:
        inputs.clear()
        states.clear()
        models.clear()
        model = payload = cpu_model = gpu_model = cpu_state = gpu_state = None
        gc.collect()
        try:
            if cuda_started:
                torch.cuda.synchronize()
                result['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                result['GPU_allocated_before_empty_cache_bytes'] = torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                result['GPU_allocated_after_cleanup_bytes'] = torch.cuda.memory_allocated()
                result['GPU_reserved_after_cleanup_bytes'] = torch.cuda.memory_reserved()
                result['owned_allocator_cleanup_passed'] = (
                    result['GPU_allocated_after_cleanup_bytes'] == 0)
        except BaseException as cleanup_error:
            result['cleanup_error'] = dict(type=type(cleanup_error).__name__,
                                           reason=str(cleanup_error))
            result['owned_allocator_cleanup_passed'] = False
        result['wall_seconds'] = time.perf_counter()-started
        result['process_terminal_receipt_required_after_return'] = True
        write(output/'result.json', result)
        print(json.dumps(dict(status=result['status'], result_sha256=sha(output/'result.json'),
                              process_exit_not_yet_observed=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if not args.execute:
        print(json.dumps(dict(status='registration_only', model_loaded=False, device_queries=0)))
        return
    if not args.output:
        raise ValueError('New explicit output required')
    run(plan, args.output, args.plan)


if __name__ == '__main__':
    main()
