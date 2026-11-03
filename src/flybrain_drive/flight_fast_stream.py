"""Separately admitted fast-body stream; the original stream certificate covers its parent only."""

import json
from pathlib import Path

import torch

from .flight_fast_graph import GraphedFastHaltere
from .flight_fast_haltere import FastHaltereDynamics
from .flight_fast_integrity import verify_trial_evidence
from .flight_stream import FlightStream
from .flyvis_graphed_tail import install_graphed_visual_tail
from .graph import sha256

PARENT_SOURCES = ('flight_stream', 'flyvis_camera', 'flyvis_compat', 'virtual_body',
    'flight_circuit', 'flight_commands', 'flight_navigation', 'vehicle_circuit',
    'flight_learning', 'ac_csp_vision', 'ac_csp_geometry', 'ac_native_physics', 'sensor_contract')
FAST_SOURCES = (*PARENT_SOURCES, 'flight_fast_haltere', 'flight_fast_stream',
                'flight_fast_graph', 'flight_fast_integrity',
                'flyvis_frozen_tail', 'flyvis_graphed_tail',
                'flight_scratch_warmup', 'flight_yaw_trial')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def fast_definition(config_file):
    from scripts.record_ac_engine_ai_probe import verified_shadow

    config = read(config_file)
    if (config['schema'] != 'flybrain.fast_flight_stream.v1' or config['task_speed_kmh'] != 6
            or config['fine_neuron_tolerance'] != .001
            or config.get('fast_neural_backend', 'eager')
            not in ('eager', 'graphed_fast_haltere')):
        raise ValueError('Explicit fast-body stream limited to the declared6kmh component task')
    if (type(config.get('graph_fractional_intervals', False)) is not bool
            or config.get('graph_fractional_intervals')
            and config.get('fast_neural_backend') != 'graphed_fast_haltere'):
        raise ValueError('Fractional graph explicitly requires the graphed fast-body backend')
    for field in ('parent_config', 'prototype_plan', 'motor_reference'):
        if sha256(config[field]) != config[field+'_sha256']:
            raise ValueError('Fast stream input changed: '+field)
    parent, proof = verified_shadow(config['parent_config'], config['parent_verification'])
    for field in ('numerical_audit', 'original_history_verification'):
        if sha256(Path(config[field])/'result.json') != config[field+'_result_sha256']:
            raise ValueError('Fast component certificate changed: '+field)
    numerical = read(Path(config['numerical_audit'])/'result.json')
    history = read(Path(config['original_history_verification'])/'result.json')
    if (not numerical['all_cases_passed']
            or history['status'] != 'completed_fast_original_history_audits'):
        raise ValueError('Successful component numerical and original-history checks required')
    old_manifest = read(Path(config['original_history_verification'])/'manifest.json')
    file = 'src/flybrain_drive/flight_fast_haltere.py'
    if old_manifest['source_sha256'][file] != sha256(file):
        raise ValueError('Fast neuronal component changed since original-history verification')
    reference = read(config['motor_reference'])
    prototype = read(config['prototype_plan'])
    candidates = [row for row in prototype['hypotheses'] if row['name'] == config['hypothesis']]
    calibrated = [row for row in reference['hypotheses'] if row['name'] == config['hypothesis']]
    if len(candidates) != 1 or len(calibrated) != 1:
        raise ValueError('One explicit tested hypothesis and motor reference required')
    for key in ('name', 'coupling', 'fast_tau_seconds'):
        if candidates[0][key] != calibrated[0][key]:
            raise ValueError('Hypothesis/reference mismatch')
    if reference['original_checkpoint_sha256'] != parent['checkpoint_sha256']:
        raise ValueError('Exact original parent checkpoint required')
    return config, parent, proof, prototype, reference, calibrated[0]


class FastFlightStream:
    def __init__(self, config_file, *, native_identity, episode_id):
        config, _, proof, prototype, reference, candidate = fast_definition(config_file)
        self.fast_config = config
        self.config_sha256 = sha256(config_file)
        self.parent_proof = proof
        self.base = FlightStream(config['parent_config'], native_identity=native_identity,
                                 episode_id=episode_id)
        self.base.goals[0, 1] = config['task_speed_kmh']/30.
        self.dynamics = FastHaltereDynamics(self.base.flight, prototype['inventory'],
            maximum_microstep=prototype['maximum_internal_microstep_seconds'])
        if self.dynamics.contract != reference['dynamics_contract']:
            raise ValueError('Configured dynamics and calibrated reference contract differ')
        if reference['spec_sha256'] != self.base.flight.spec_sha256:
            raise ValueError('Configured original neuron specification changed')
        original_reference = torch.tensor(reference['original_lateral_reference'],
                                          device=self.base.device)
        if not torch.equal(original_reference, self.base.flight.readout.lateral_reference):
            raise ValueError('Original readout reference changed')
        with torch.no_grad():
            self.base.flight.readout.lateral_reference.copy_(torch.tensor(
                candidate['lateral_reference'], device=self.base.device))
        self.candidate = candidate
        self.fast_stepper = (GraphedFastHaltere(self.dynamics,
            coupling=candidate['coupling'], fast_tau=candidate['fast_tau_seconds'],
            graph_fractional=config.get('graph_fractional_intervals', False))
            if config.get('fast_neural_backend') == 'graphed_fast_haltere' else None)

        def fast_advance(drive, state, elapsed):
            if self.fast_stepper is not None:
                return self.fast_stepper.advance(drive, state, elapsed)
            return self.dynamics.advance(drive, state, elapsed,
                coupling=candidate['coupling'], fast_tau=candidate['fast_tau_seconds'])

        # Explicitly versioned replacement on this new private stream instance only.
        self.base.flight._advance_elapsed_tensor = fast_advance
        install_graphed_visual_tail(self.base)
        self.base.flight.assert_body_boundary_protected()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'base'), name)

    def advance(self, **observation):
        output = self.base.advance(**observation)
        output.update(fast_stream_schema=self.fast_config['schema'],
            fast_stream_config_sha256=self.config_sha256,
            fast_hypothesis=self.candidate['name'],
            fast_reference_sha256=self.fast_config['motor_reference_sha256'],
            fast_neural_backend=self.fast_config.get('fast_neural_backend', 'eager'),
            operator_nominal_graph_calls=(self.fast_stepper.nominal_steps
                                          if self.fast_stepper else 0),
            operator_fractional_eager_calls=(self.fast_stepper.fractional_steps
                                             if self.fast_stepper else 0),
            operator_fractional_graph_calls=(self.fast_stepper.fractional_graph_steps
                                             if self.fast_stepper else 0),
            graph_fractional_intervals=self.fast_config.get('graph_fractional_intervals', False),
            fast_yaw_mode=getattr(self, 'yaw_trial_mode', None))
        return output


def verified_fast_stream(config_file, verification):
    config, parent, parent_proof, _, _, _ = fast_definition(config_file)
    root = Path(verification)
    receipt, manifest = read(root/'result.json'), read(root/'manifest.json')
    if (receipt.get('status') != 'completed_fast_stream_original_replay'
            or not receipt.get('all_original_observations_replayed')
            or not receipt.get('full_and_zero_yaw_both_verified')
            or not receipt.get('parameters_fixed_during_each_replay')
            or not receipt.get('real_time_budget_passed')
            or receipt.get('unique_original_observations') != 1542
            or receipt.get('histories') != 6 or receipt.get('modes_per_history') != 2
            or receipt.get('hidden_actual_resets') != 0
            or manifest['config']['fast_config_sha256'] != sha256(config_file)
            or receipt['maximum_intent_error'] > 3e-6
            or receipt['maximum_exact_neuron_error'] > 3e-6
            or receipt['maximum_fine_neuron_error'] > config['fine_neuron_tolerance']):
        raise ValueError('Exact complete original-history fast-stream certificate required')
    for source in FAST_SOURCES:
        file = 'src/flybrain_drive/'+source+'.py'
        if manifest['source_sha256'].get(file) != sha256(file):
            raise ValueError('Verified fast-stream source changed: '+source)
    for file in ('scripts/verify_flight_fast_stream_trial.py',
                 'scripts/verify_flight_fast_stream.py',
                 'scripts/record_ac_engine_ai_probe.py'):
        if manifest['source_sha256'].get(file) != sha256(file):
            raise ValueError('Fast stream verifier changed: '+file)
    for entry in receipt['trials']:
        trial = Path(entry['path'])
        if sha256(trial/'result.json') != entry['result_sha256']:
            raise ValueError('Fast-stream trial certificate changed')
    verify_trial_evidence(config, config_file, receipt)
    return parent, dict(verification=str(root), result_sha256=sha256(root/'result.json'),
        fast_config_sha256=sha256(config_file), parent_verification=parent_proof,
        initialization_seed=parent_proof['initialization_seed'], hypothesis=config['hypothesis'])


def verified_fast_plan(plan):
    if not plan.get('fast_stream_config'):
        if plan.get('fast_stream_verification'):
            raise ValueError('Fast stream certificate without an explicit configuration')
        return None
    _, proof = verified_fast_stream(plan['fast_stream_config'], plan['fast_stream_verification'])
    config = read(plan['fast_stream_config'])
    if (config['parent_config'] != plan['inference_config']
            or config['parent_verification'] != plan['inference_verification']
            or plan.get('visual_backend') != 'graphed_visual_tail'):
        raise ValueError('Explicit same parent and graphed visual backend required for fast model')
    return proof
