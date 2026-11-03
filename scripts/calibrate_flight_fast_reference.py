"""Versioned motor-only reference preserving original visual/command policy bias."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_fast_haltere import FastHaltereDynamics
from flybrain_drive.flight_fast_reference import relative_motor_reference
from flybrain_drive.flight_frequency import fundamental
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.probe)
    before = read(source/'result.json')
    source_config = read(source/'manifest.json')['config']
    plan = source_config['plan']
    assert before['status'] == 'completed_fast_haltere_hypothesis_probe'
    assert sha256(source/'prototype-frequency-responses.npz') == before['archive_sha256']
    config = read(plan['stream_config'])
    training = Path(config['training'])
    parent = read(training/'result.json')
    spec = read(training/'manifest.json')['config']['plan']['spec']
    assert sha256(parent['checkpoint']) == config['checkpoint_sha256']
    payload = torch.load(parent['checkpoint'], map_location='cpu', weights_only=True)
    hypotheses = plan['hypotheses']
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        source_result_sha256=sha256(source/'result.json'),
        original_checkpoint_sha256=config['checkpoint_sha256'], hypotheses=hypotheses,
        reference='Same recorded body context with yaw0 and external vision/steering DN drive0',
        calibration='Add each changed motor lateral reference minus original-model motor reference',
        preserved='Readout gain/mix, power reference, neural graph, body encoder, scene policy',
        model_time_only=True, settling_seconds=4., additional_settling_audit_seconds=.5))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0,
        optimizer_updates=0, ui_actions_sent=0, actor_qualified=False, physiology_verified=False)
    try:
        torch.set_num_threads(1)
        assert torch.cuda.is_available()
        model = VisualFlightCircuit(spec, glutamate_sign=payload['glutamate_sign'])
        model.load_state_dict(payload['model'], strict=True)
        model.to('cuda').eval().requires_grad_(False)
        dynamics = FastHaltereDynamics(model, plan['inventory'])
        count = len(hypotheses)
        body = torch.tensor(source_config['reference_body'], device='cuda').repeat(count, 1)
        valid = torch.tensor(source_config['reference_valid'], device='cuda').repeat(count, 1)
        body[:, 4] = 0.
        goals = torch.tensor([[0., .2]], device='cuda').repeat(count, 1)
        features = model.scene_command.mean[None].expand(count, -1)
        visual = torch.zeros(count, 6, device='cuda')
        visual_valid = torch.zeros_like(visual, dtype=torch.bool)
        couplings = torch.tensor([row['coupling'] for row in hypotheses], device='cuda')
        taus = torch.tensor([row['fast_tau_seconds'] for row in hypotheses], device='cuda')
        state = model.initial_state(count)
        persistent = set(model.state_dict())-{'readout.lateral_reference'}
        versions = [(name, id(value), value._version) for name, value in
                    list(model.named_parameters())+list(model.named_buffers())
                    if name in persistent]
        original_reference = model.readout.lateral_reference.clone()
        with torch.no_grad():
            held = model._drive_tensor(visual, visual_valid, body, valid, goals, features)
            held[:, model.vision_ports] = 0.
            held[:, model.steering_ports] = 0.
            for _ in range(400):
                state = dynamics.advance(held, state, .01, coupling=couplings, fast_tau=taus)
            settled = state.clone()
            for _ in range(50):
                state = dynamics.advance(held, state, .01, coupling=couplings, fast_tau=taus)
            drift = float((settled-state).abs().max())
            assert drift <= 3e-6
            raw, muscles = model._read_state_preactivation_tensor(state)
            references, shifts = relative_motor_reference(model.readout, muscles)
            maximum_readout_error = 0.
            for index in range(count):
                model.readout.lateral_reference.copy_(references[index])
                actual = model._read_state_tensor(state[index:index+1])[0][0]
                expected = raw[index].clone()
                expected[0] -= shifts[index]
                expected = expected.tanh()
                maximum_readout_error = max(maximum_readout_error,
                                            float((actual-expected).abs().max()))
                assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
            model.readout.lateral_reference.copy_(original_reference)
        assert versions == [(name, id(value), value._version) for name, value in
                            list(model.named_parameters())+list(model.named_buffers())
                            if name in persistent]
        model.assert_body_boundary_protected()
        assert torch.equal(model.readout.lateral_reference, original_reference)
        with load_array(source/'prototype-frequency-responses.npz', allow_pickle=False) as archived:
            times, raw_intents = archived['model_time'], archived['intents']
            raw_neutral = archived['neutral_intent']
        assert np.all(np.abs(raw_intents) < 1)
        shift_map = {row['name']: float(shifts[n]) for n, row in enumerate(hypotheses)}
        case_shifts = np.array([shift_map[row['hypothesis']] for row in before['cases']])
        calibrated = raw_intents.copy()
        calibrated[:, :, 0] = np.tanh(
            np.arctanh(raw_intents[:, :, 0].astype(np.float64))-case_shifts)
        neutral = raw_neutral.copy()
        neutral[:, 0] = np.tanh(np.arctanh(raw_neutral[:, 0].astype(np.float64))-case_shifts)
        selected = (times >= 12.) & (times < 20.)
        reports = []
        for index, row in enumerate(before['cases']):
            report = {key: row[key] for key in ('hypothesis', 'frequency_hz', 'amplitude_rad_s',
                                               'cadence_ticks', 'coupling', 'fast_tau_seconds')}
            response = fundamental(times[selected], -calibrated[selected, index, 0],
                                   row['frequency_hz'])
            report.update(opposing_intent_response=response,
                gain_per_ideal_yaw=response['amplitude']/row['amplitude_rad_s'],
                neutral_intent=neutral[index].tolist(),
                neutral_difference_from_original=(neutral[index]-neutral[0]).tolist())
            reports.append(report)
        candidate_references = [dict(**hypothesis,
            lateral_reference=references[n].cpu().tolist(),
            preactivation_shift=float(shifts[n]),
            motor_reference_intent_before=raw[n].tanh().cpu().tolist())
            for n, hypothesis in enumerate(hypotheses)]
        write_json(run/'reference-candidates.json', dict(
            schema='flybrain.fast_haltere_motor_reference.v1',
            original_checkpoint_sha256=config['checkpoint_sha256'], spec_sha256=sha256(spec),
            dynamics_contract=dynamics.contract, hypotheses=candidate_references,
            original_lateral_reference=original_reference.cpu().tolist(),
            matched_original_motor_only_reference=True, visual_policy_bias_not_zeroed=True,
            actor_qualified=False, physical_vehicle_control_allowed=False))
        np.savez_compressed(run/'calibrated-open-loop-readout.npz', model_time=times,
            intents=calibrated, neutral_intent=neutral,
            motor_reference_activity=muscles.cpu().numpy())
        result.update(status='completed_relative_motor_reference_calibration', cases=reports,
            reference_settling_neuron_drift=drift,
            actual_readout_equivalence_max_error=maximum_readout_error,
            original_checkpoint_unchanged=True, unchanged_fields_verified=True,
            allowed_reference_buffer='readout.lateral_reference',
            reference_restored_after_validation=True,
            physical_trajectory_simulated=False, physical_stability_repaired=False,
            candidate_reference_sha256=sha256(run/'reference-candidates.json'),
            archive_sha256=sha256(run/'calibrated-open-loop-readout.npz'))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
