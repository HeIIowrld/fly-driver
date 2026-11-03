"""Five fast-path hypotheses on unchanged full/zero-yaw original observation histories."""

import json
from pathlib import Path

import numpy as np
import torch

from .flight_fast_haltere import FastHaltereDynamics
from .graph import sha256


class FastOriginalReplay:
    def __init__(self, stream, plan_file, reference_file):
        if stream.yaw_trial_mode != 'full':
            raise ValueError('Original unmodified full-mode source required')
        plan = json.loads(Path(plan_file).read_text(encoding='utf8'))
        reference = json.loads(Path(reference_file).read_text(encoding='utf8'))
        model = stream.flight
        if (reference['schema'] != 'flybrain.fast_haltere_motor_reference.v1'
                or reference['original_checkpoint_sha256'] != stream.config['checkpoint_sha256']
                or reference['spec_sha256'] != model.spec_sha256):
            raise ValueError('Exact original checkpoint/spec and declared motor reference required')
        self.dynamics = FastHaltereDynamics(model, plan['inventory'],
            maximum_microstep=plan['maximum_internal_microstep_seconds'])
        if self.dynamics.contract != reference['dynamics_contract']:
            raise ValueError('Fast-path hypothesis/reference contract mismatch')
        self.hypotheses = plan['hypotheses']
        for row, calibrated in zip(self.hypotheses, reference['hypotheses'], strict=True):
            for key in ('name', 'coupling', 'fast_tau_seconds'):
                if row[key] != calibrated[key]:
                    raise ValueError('Declared hypothesis and reference differ')
        device = stream.goals.device
        self.couplings = torch.tensor([r['coupling'] for r in self.hypotheses],
                                      device=device).repeat_interleave(2)
        self.taus = torch.tensor([r['fast_tau_seconds'] for r in self.hypotheses],
                                 device=device).repeat_interleave(2)
        self.shifts = torch.tensor([r['preactivation_shift'] for r in reference['hypotheses']],
                                   device=device).repeat_interleave(2)
        self.state = model.initial_state(len(self.hypotheses)*2)
        self.previous = self.held = None
        self.rows = []
        self.maximum_state_error = 0.
        self.maximum_intent_error = 0.
        self.contract = dict(plan_sha256=sha256(plan_file), reference_sha256=sha256(reference_file),
            dynamics=self.dynamics.contract, original_clocks_preserved=True,
            maximum_internal_microstep_seconds=plan['maximum_internal_microstep_seconds'],
            physical_counterfactual_trajectory=False, actuator_messages_sent=0)

    @torch.no_grad()
    def observe(self, stream, clock, original_intent):
        if self.previous is not None:
            elapsed = clock-self.previous
            if not 0 < elapsed <= .08:
                raise ValueError('Original positive bounded interval required; no gap repair')
            nominal = stream.flight.spec['dynamics']['dt']
            count = int(elapsed//nominal)
            intervals = [nominal]*count
            if elapsed-count*nominal > 1e-12:
                intervals.append(elapsed-count*nominal)
            for dt in intervals:
                self.state = self.dynamics.advance(self.held, self.state, dt,
                    coupling=self.couplings, fast_tau=self.taus)
        error = float((self.state[0]-stream.flight_state[0]).abs().max())
        self.maximum_state_error = max(self.maximum_state_error, error)
        if error > 3e-6:
            raise ValueError('Original whole-neuron baseline mismatch')
        preactivation = stream.flight._read_state_preactivation_tensor(self.state)[0]
        preactivation[:, 0] -= self.shifts
        output = preactivation.tanh().cpu().numpy().copy()
        np.testing.assert_allclose(output[0], original_intent, atol=3e-6, rtol=2e-5)
        self.maximum_intent_error = max(self.maximum_intent_error,
            float(np.max(np.abs(output[0]-original_intent))))
        # All readout shifts are neuron-reference constants, not vehicle-control residuals.
        body = torch.tensor(np.array(stream.last_body.values, dtype=np.float32)[None],
                            device=stream.goals.device)
        body[:, 4] = 0.
        valid = torch.tensor(np.array(stream.last_body.valid)[None], device=stream.goals.device)
        zero = stream.flight._drive_tensor(stream.visual_zero, stream.visual_valid, body, valid,
                                          stream.goals, stream.last_features)
        self.held = torch.cat([stream.flight_held, zero], dim=0).repeat(len(self.hypotheses), 1)
        self.rows.append(output.reshape(len(self.hypotheses), 2, 2))
        self.previous = clock
