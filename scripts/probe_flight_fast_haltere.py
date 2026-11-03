"""All predeclared contact-current hypotheses, preserving original identity and chemical weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_commands import COMMAND_GOAL_CONTRACT
from flybrain_drive.flight_fast_haltere import FastHaltereDynamics
from flybrain_drive.flight_frequency import fundamental
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    assert plan['schema'] == 'flybrain.fast_haltere_prototype_experiment.v1'
    assert plan['physical_vehicle_control_allowed'] is False
    config = read(plan['stream_config'])
    training = Path(config['training'])
    receipt = read(training/'result.json')
    spec = read(training/'manifest.json')['config']['plan']['spec']
    assert sha256(receipt['checkpoint']) == config['checkpoint_sha256']
    payload = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
    assert payload['goal_contract'] == COMMAND_GOAL_CONTRACT
    assert payload['spec_sha256'] == sha256(spec)
    root = Path(plan['body_recording'])
    assert read(root/'result.json')['status'] == 'completed_physical_yaw_trial'
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    reference = [row for row in events if row['kind'] == 'flight_axis_observation'
                 and row['phase'] == 'baseline'][-1]
    cases = [dict(hypothesis=h['name'], coupling=h['coupling'],
                  fast_tau_seconds=h['fast_tau_seconds'], frequency_hz=f,
                  amplitude_rad_s=a, cadence_ticks=c)
             for h in plan['hypotheses'] for a in plan['amplitudes_rad_s']
             for c in plan['cadence_ticks'] for f in plan['frequencies_hz']]
    baseline_count = sum(case['hypothesis'] == 'original' for case in cases)
    assert baseline_count == 24 and all(c['hypothesis'] == 'original' for c in cases[:24])
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, plan=plan,
        plan_sha256=sha256(args.plan), cases=cases,
        checkpoint_sha256=config['checkpoint_sha256'], spec_sha256=sha256(spec),
        reference_events_sha256=sha256(root/'events.jsonl'),
        reference_body=reference['body_values'], reference_valid=reference['body_valid'],
        model_time_only=True, selection='All hypotheses; no automatic winner or deployment'))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0,
        optimizer_updates=0, ui_actions_sent=0, actor_qualified=False,
        physical_vehicle_tested=False, physiology_verified=False)
    try:
        torch.set_num_threads(1)
        assert torch.cuda.is_available()
        model = VisualFlightCircuit(spec, glutamate_sign=payload['glutamate_sign'])
        model.load_state_dict(payload['model'], strict=True)
        model.to('cuda').eval().requires_grad_(False)
        model.assert_body_boundary_protected()
        dynamics = FastHaltereDynamics(model, plan['inventory'],
            maximum_microstep=plan['maximum_internal_microstep_seconds'])
        write_json(run/'hypothesis-contract.json', dynamics.contract)
        persistent = set(model.state_dict())
        versions = [(name, id(value), value._version) for name, value in
                    list(model.named_parameters())+list(model.named_buffers())
                    if name in persistent]
        count, dt = len(cases), plan['nominal_dt_seconds']
        assert model.spec['dynamics']['dt'] == dt == .01
        body = torch.tensor(reference['body_values'], dtype=torch.float32,
                            device='cuda').repeat(count, 1)
        valid = torch.tensor(reference['body_valid'], device='cuda').repeat(count, 1)
        assert valid[:, 4].all()
        visual = torch.zeros(count, 6, device='cuda')
        visual_valid = torch.zeros_like(visual, dtype=torch.bool)
        goals = torch.tensor([[0., .2]], device='cuda').repeat(count, 1)
        features = model.scene_command.mean[None].expand(count, -1)
        couplings = torch.tensor([case['coupling'] for case in cases], device='cuda')
        taus = torch.tensor([case['fast_tau_seconds'] for case in cases], device='cuda')
        state, original = model.initial_state(count), model.initial_state(baseline_count)
        maximum_state_error = torch.zeros((), device='cuda')

        def drive():
            return model._drive_tensor(visual, visual_valid, body, valid, goals, features)

        def advance(held):
            nonlocal state, original, maximum_state_error
            state = dynamics.advance(held, state, dt, coupling=couplings, fast_tau=taus)
            original = model._advance_elapsed_tensor(held[:baseline_count], original, dt)
            maximum_state_error = torch.maximum(maximum_state_error,
                (state[:baseline_count]-original).abs().max())

        with torch.no_grad():
            held = drive()
            for _ in range(round(plan['settling_seconds']/dt)):
                advance(held)
            neutral = model._read_state_tensor(state)[0].cpu().numpy().copy()
            times = np.arange(round(plan['stimulus_seconds']/dt)+1)*dt
            frequencies = np.array([case['frequency_hz'] for case in cases])
            amplitudes = np.array([case['amplitude_rad_s'] for case in cases])
            cadences = np.array([case['cadence_ticks'] for case in cases])
            body_reference = float(body[0, 4])
            ideal = body_reference+np.sin(2*np.pi*times[:, None]*frequencies)*amplitudes
            values, inputs = [], []
            current = np.full(count, body_reference)
            for step in range(len(times)):
                values.append(model._read_state_tensor(state)[0].cpu().numpy().copy())
                update = step % cadences == 0
                current[update] = ideal[step, update]
                inputs.append(current.copy())
                body[:, 4] = torch.tensor(current, dtype=torch.float32, device='cuda')
                held = drive()
                if step < len(times)-1:
                    advance(held)
                if step % 200 == 0:
                    write_json(run/'progress.json', dict(model_tick=step, total_ticks=len(times)-1))
            values, inputs = np.array(values), np.array(inputs)
        maximum_state_error = float(maximum_state_error)
        assert maximum_state_error <= 3e-6
        previous = Path(plan['original_frequency_probe'])
        prior = read(previous/'result.json')
        assert prior['status'] == 'completed_model_yaw_frequency_probe'
        assert sha256(previous/'model-frequency-responses.npz') == prior['archive_sha256']
        with load_array(previous/'model-frequency-responses.npz', allow_pickle=False) as old:
            np.testing.assert_array_equal(times, old['model_time'])
            np.testing.assert_allclose(values[:, :baseline_count], old['intents'],
                                       atol=3e-6, rtol=2e-5)
            prior_error = float(np.max(np.abs(values[:, :baseline_count]-old['intents'])))
        selected = ((times >= plan['stimulus_seconds']-plan['fit_final_seconds'])
                    & (times < times[-1]))
        reports = []
        for index, case in enumerate(cases):
            output = fundamental(times[selected], -values[selected, index, 0], case['frequency_hz'])
            supplied = fundamental(times[selected], inputs[selected, index], case['frequency_hz'])
            reports.append(dict(**case, opposing_intent_response=output,
                gain_per_ideal_yaw=output['amplitude']/case['amplitude_rad_s'],
                phase_relative_to_held_degrees=(output['phase_degrees']
                    -supplied['phase_degrees']+180) % 360-180,
                neutral_intent=neutral[index].tolist()))
        for hypothesis in plan['hypotheses']:
            for amplitude in plan['amplitudes_rad_s']:
                for cadence in plan['cadence_ticks']:
                    group = [row for row in reports if row['hypothesis'] == hypothesis['name']
                             and row['amplitude_rad_s'] == amplitude
                             and row['cadence_ticks'] == cadence]
                    phases = np.unwrap(np.deg2rad([row['opposing_intent_response']['phase_degrees']
                                                  for row in group]))
                    for row, phase in zip(group, phases, strict=True):
                        row['phase_to_ideal_unwrapped_degrees'] = float(np.rad2deg(phase))
                        row['gain_relative_to_half_hz'] = (
                            row['gain_per_ideal_yaw']/group[0]['gain_per_ideal_yaw'])
        assert versions == [(name, id(value), value._version) for name, value in
                            list(model.named_parameters())+list(model.named_buffers())
                            if name in persistent]
        model.assert_body_boundary_protected()
        assert sha256(receipt['checkpoint']) == config['checkpoint_sha256']
        np.savez_compressed(run/'prototype-frequency-responses.npz', model_time=times,
            intents=values, held_yaw=inputs, ideal_yaw=ideal, neutral_intent=neutral)
        result.update(status='completed_fast_haltere_hypothesis_probe', cases=reports,
            contract=dynamics.contract, full_neuron_baseline_max_error=maximum_state_error,
            prior_frequency_intent_max_error=prior_error, original_checkpoint_unchanged=True,
            persistent_parameter_versions_unchanged=True, direct_vehicle_residual=False,
            physical_stability_repaired=False,
            archive_sha256=sha256(run/'prototype-frequency-responses.npz'))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
