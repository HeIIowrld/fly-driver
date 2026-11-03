"""Controlled model-only yaw frequency response; no car, camera, optimizer or actuator access."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_commands import COMMAND_GOAL_CONTRACT
from flybrain_drive.flight_frequency import fundamental
from flybrain_drive.flight_navigation import VisualFlightCircuit
from flybrain_drive.graph import sha256

FREQUENCIES = (.5, 1., 2., 4., 8., 12.)
AMPLITUDES = (.005, .02)
CADENCES = (1, 4)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-config', required=True)
    parser.add_argument('--body-recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    stream = read(args.stream_config)
    training = Path(stream['training'])
    receipt = read(training/'result.json')
    spec = read(training/'manifest.json')['config']['plan']['spec']
    assert sha256(receipt['checkpoint']) == stream['checkpoint_sha256']
    payload = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True)
    assert payload['spec_sha256'] == sha256(spec)
    assert payload['goal_contract'] == COMMAND_GOAL_CONTRACT
    root = Path(args.body_recording)
    assert read(root/'result.json')['status'] == 'completed_physical_yaw_trial'
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    reference = [row for row in events if row['kind'] == 'flight_axis_observation'
                 and row['phase'] == 'baseline'][-1]
    cases = [dict(frequency_hz=f, amplitude_rad_s=a, cadence_ticks=c)
             for a in AMPLITUDES for c in CADENCES for f in FREQUENCIES]
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, cases=cases,
        checkpoint_sha256=stream['checkpoint_sha256'], spec_sha256=sha256(spec),
        reference_events_sha256=sha256(root/'events.jsonl'),
        reference_body=reference['body_values'], reference_valid=reference['body_valid'],
        time_domain='New synthetic model time; never recorded as native observations',
        scene=('Fixed training normalization mean; '
               'normalized scene_command evidence zero, bias retained'),
        nominal_dt=.01, settling_seconds=2., stimulus_seconds=20., fit_final_seconds=8.))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0,
        ui_actions_sent=0, optimizer_updates=0, physical_vehicle_tested=False,
        physiology_verified=False)
    try:
        torch.set_num_threads(1)
        assert torch.cuda.is_available()
        model = VisualFlightCircuit(spec, glutamate_sign=payload['glutamate_sign'])
        model.load_state_dict(payload['model'], strict=True)
        model.to('cuda').eval().requires_grad_(False)
        model.assert_body_boundary_protected()
        dt = model.spec['dynamics']['dt']
        assert dt == .01
        persistent = set(model.state_dict())
        named = [(name, value) for name, value in list(model.named_parameters())
                 + list(model.named_buffers()) if name in persistent]
        versions = [(name, id(value), value._version) for name, value in named]
        count = len(cases)
        body = torch.tensor(reference['body_values'], dtype=torch.float32,
                            device='cuda').repeat(count, 1)
        valid = torch.tensor(reference['body_valid'], device='cuda').repeat(count, 1)
        assert valid[:, 4].all()
        visual = torch.zeros(count, 6, device='cuda')
        visual_valid = torch.zeros_like(visual, dtype=torch.bool)
        goals = torch.tensor([[0., .2]], device='cuda').repeat(count, 1)
        features = model.scene_command.mean[None].expand(count, -1)
        state = model.initial_state(count)
        body_reference = float(body[0, 4])

        def drive():
            return model._drive_tensor(visual, visual_valid, body, valid, goals, features)

        with torch.no_grad():
            held = drive()
            for _ in range(200):
                state = model._advance_elapsed_tensor(held, state, dt)
            times = np.arange(2001)*dt
            frequencies = np.array([c['frequency_hz'] for c in cases])
            amplitudes = np.array([c['amplitude_rad_s'] for c in cases])
            cadences = np.array([c['cadence_ticks'] for c in cases])
            ideal = body_reference+np.sin(2*np.pi*times[:, None]*frequencies)*amplitudes
            values, inputs = [], []
            current = np.full(count, body_reference)
            for index in range(len(times)):
                output = model._read_state_tensor(state)[0].cpu().numpy().copy()
                values.append(output)
                update = index % cadences == 0
                current[update] = ideal[index, update]
                inputs.append(current.copy())
                body[:, 4] = torch.tensor(current, dtype=torch.float32, device='cuda')
                held = drive()
                if index < len(times)-1:
                    state = model._advance_elapsed_tensor(held, state, dt)
                if index % 400 == 0:
                    write_json(run/'progress.json', dict(model_tick=index, total_ticks=2000))
            values, inputs = np.array(values), np.array(inputs)
        assert np.isfinite(values).all()
        selected = (times >= 12.) & (times < 20.)
        reports = []
        for index, case in enumerate(cases):
            output = fundamental(times[selected], -values[selected, index, 0], case['frequency_hz'])
            supplied = fundamental(times[selected], inputs[selected, index], case['frequency_hz'])
            relative = (output['phase_degrees']-supplied['phase_degrees']+180) % 360-180
            reports.append(dict(**case, opposing_intent_response=output,
                held_input_fundamental=supplied,
                gain_per_held_yaw=output['amplitude']/supplied['amplitude'],
                phase_relative_to_held_degrees=relative,
                gain_per_ideal_yaw=output['amplitude']/case['amplitude_rad_s']))
        for amplitude in AMPLITUDES:
            for cadence in CADENCES:
                group = [r for r in reports if r['amplitude_rad_s'] == amplitude
                         and r['cadence_ticks'] == cadence]
                phases = np.unwrap(np.deg2rad([r['opposing_intent_response']['phase_degrees']
                                              for r in group]))
                for row, phase in zip(group, phases, strict=True):
                    row['phase_to_ideal_unwrapped_degrees'] = float(np.rad2deg(phase))
                    row['gain_relative_to_half_hz'] = (
                        row['gain_per_ideal_yaw']/group[0]['gain_per_ideal_yaw'])
        model.assert_body_boundary_protected()
        assert versions == [(name, id(value), value._version)
                            for name, value in list(model.named_parameters())
                            + list(model.named_buffers()) if name in persistent]
        # Nonpersistent CSR caches are created lazily. Check their actual numerical contents,
        # rather than treating cache initialization as a learned-parameter mutation.
        matrix = model.circuit.matrix.coalesce()
        for cache, expected in ((model.circuit._cached_matrix, matrix),
                                (model.circuit._cached_transpose,
                                 matrix.transpose(0, 1).coalesce())):
            actual = cache.to_sparse_coo().coalesce()
            assert torch.equal(actual.indices(), expected.indices())
            assert torch.equal(actual.values(), expected.values())
        assert sha256(receipt['checkpoint']) == stream['checkpoint_sha256']
        np.savez_compressed(run/'model-frequency-responses.npz', model_time=times,
            ideal_yaw=ideal, held_yaw=inputs, intents=values)
        result.update(status='completed_model_yaw_frequency_probe', cases=reports,
            neural_dt_seconds=dt, neural_tau_seconds=model.spec['dynamics']['tau'],
            model_state_resets_during_stimulus=0, parameter_versions_unchanged=True,
            nonpersistent_csr_cache_values_verified=True,
            physical_stability_repaired=False, actor_qualified=False,
            archive_sha256=sha256(run/'model-frequency-responses.npz'))
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)


if __name__ == '__main__':
    main()
