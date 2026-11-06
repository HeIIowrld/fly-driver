"""Whole-episode information intervention of an audited offline flight candidate.

No fitting, live interface or missing-sensor fallback. Zero features with valid
masks are an explicit diagnostic manipulation, not an admissible camera failure.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.anticipation_targets import future_command_targets
from flybrain_drive.flight_anticipation_readout import FlightAnticipationReadout
from flybrain_drive.graph import sha256
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import assemble_student, MuscleVehicleReadout, read
from scripts.improve_sac_anticipation import command_metrics
from scripts.train_qualified_sac_unified import context, load_episode


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit', required=True)
    p.add_argument('--arm', required=True, choices=['continuation', 'credit32', 'auxiliary'])
    p.add_argument('--intervention', required=True, choices=['vision', 'body_feedback'])
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    fit = Path(a.fit)
    fit_result, fit_manifest = read(fit / 'result.json'), read(fit / 'manifest.json')
    checkpoint = fit / f'{a.arm}.pt'
    if (fit_result['status'] != 'completed_fixed_anticipation_learning'
            or sha256(checkpoint) != fit_result['arms'][a.arm]['checkpoint_sha256']):
        raise ValueError('Completed fixed learning/checkpoint required')
    entry = fit_manifest['config']['episodes'][2]
    data, contract, evidence = load_episode(entry['data'])
    if evidence['arrays_sha256'] != entry['arrays_sha256']:
        raise ValueError('Original consumed development data changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['actor_qualified'] is not False or payload['manifest_sha256'] != sha256(fit / 'manifest.json'):
        raise ValueError('Candidate qualification/provenance differs')
    model, lineage = assemble_student()
    if payload['lineage'] != lineage:
        raise ValueError('Original circuit lineage differs')
    model.load_state_dict(payload['model'], strict=True)
    model.motor.backbone.assert_body_boundary_protected()
    model.motor.backbone.circuit.sparse_backend = 'cached_csr_cpu'
    _, activity = model.motor.backbone._read_state_tensor(model.motor.initial_state(1))
    readout = MuscleVehicleReadout(activity.shape[-1])
    readout.load_state_dict(payload['readout'], strict=True)
    auxiliary = None
    if payload['auxiliary'] is not None:
        auxiliary = FlightAnticipationReadout(payload['auxiliary']['reference'], payload['auxiliary']['scale'])
        auxiliary.load_state_dict(payload['auxiliary'], strict=True)
        auxiliary.eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    readout.eval().requires_grad_(False)
    run = begin_run(a.output, dict(arguments=vars(a), seed=780309, program_sha256=sha256(__file__),
        fit_result_sha256=sha256(fit / 'result.json'), checkpoint_sha256=sha256(checkpoint),
        episode=evidence, scope='Consumed Monza development, explicit information clamp only',
        optimizer_updates=0, actor_deployed=False, physiological_lesion=False))
    result = dict(status='failed', optimizer_updates=0, actor_deployed=False, vehicle_control_sent=False)
    start = time.monotonic()
    body_hook = None
    try:
        scene = torch.tensor(data['scene'], dtype=torch.float32)
        if a.intervention == 'vision':
            scene.zero_()
        else:
            body_hook = model.body_feedback.register_forward_hook(
                lambda module, inputs, value: torch.zeros_like(value))
        goals, view = torch.tensor(racing_goal(300, 1)), torch.ones((1, 4), dtype=torch.bool)
        predictions, future_plans = [], []
        model.reset_episode()
        for i in range(len(scene)):
            _, activity = model.sensory_decision(decision_time=float(data['decision_time'][i]),
                racing_goals=goals, scene_features=scene[i:i+1], scene_valid=view,
                body_context=context(data, contract, i, a.intervention))
            predictions.append(readout(activity)[0].numpy().copy())
            if auxiliary is not None:
                future_plans.append(auxiliary(activity)[0].numpy().copy())
            if i % 500 == 0:
                write_json(run / 'progress.json', dict(event=i, total=len(scene),
                    intervention=a.intervention, elapsed_seconds=time.monotonic()-start))
        prediction = np.asarray(predictions)
        arrays = dict(prediction=prediction)
        if future_plans:
            arrays['future'] = np.asarray(future_plans)
        np.savez_compressed(run / 'predictions.npz', **arrays)
        metrics = command_metrics(prediction, data['labels'], data['fit_mask'])
        result.update(status='completed_continuous_information_intervention', metrics=metrics,
            normal_metrics=fit_result['arms'][a.arm]['dev'],
            weighted_mse_ratio_to_normal=metrics['weighted_mse']/fit_result['arms'][a.arm]['dev']['weighted_mse'],
            arrays_sha256=sha256(run / 'predictions.npz'), model_events=len(scene),
            recurrent_resets='One explicit whole-episode restart; no internal numeric reset',
            body_clamp_scope='Learned body feedback and yaw/tilt/load/tyre/wheel signals clamped; current speed retained solely for neutral command cancellation',
            interpretation='Information dependence on consumed development only; zero clamp may be OOD, no correct anticipatory response/physiology/new-track claim',
            termination_reason='Whole original episode continuously replayed')
        if future_plans:
            target, target_mask, _ = future_command_targets(data['decision_time'], data['labels'],
                                                           data['label_valid'], data['fit_mask'])
            result['future'] = [command_metrics(arrays['future'][:, h], target[:, h], target_mask[:, h])
                                for h in range(4)]
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error), termination_reason='Explicit intervention failure')
    finally:
        if body_hook is not None:
            body_hook.remove()
    result['elapsed_seconds'] = time.monotonic()-start
    write_json(run / 'result.json', result)
    print(json.dumps(result))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
