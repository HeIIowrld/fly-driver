"""Offline original-clock flight replays sharing observed vision, never a simulated car."""

import numpy as np
import torch

MODES = ('full', 'zero_yaw', 'reversed_yaw', 'scene_command_lesion',
         'zero_yaw_scene_command_lesion')


class FlightCounterfactualReplay:
    def __init__(self, stream):
        if stream.yaw_trial_mode != 'full':
            raise ValueError('Unmodified full flight drive required for counterfactual diagnosis')
        self.state = stream.flight.initial_state(len(MODES))
        self.previous = self.held = None
        self.maximum_error = 0.
        self.rows = []

    @torch.no_grad()
    def observe(self, stream, clock, original_intent):
        if self.previous is not None:
            elapsed = clock-self.previous
            if not 0 < elapsed <= .08:
                raise ValueError('Original positive interval required; no gap repair')
            nominal = stream.flight.spec['dynamics']['dt']
            count = int(elapsed//nominal)
            intervals = [nominal]*count
            if elapsed-count*nominal > 1e-12:
                intervals.append(elapsed-count*nominal)
            for dt in intervals:
                self.state = stream.flight._advance_elapsed_tensor(self.held, self.state, dt)
        output = stream.flight._read_state_tensor(self.state)[0].cpu().numpy().copy()
        np.testing.assert_allclose(output[0], original_intent, atol=3e-6, rtol=2e-5)
        self.maximum_error = max(self.maximum_error,
                                 float(np.max(np.abs(output[0]-original_intent))))
        # Intervene in disposable input copies, retaining original values and all validity bits.
        device = stream.goals.device
        body = torch.tensor(np.array(stream.last_body.values, dtype=np.float32),
                            device=device).repeat(len(MODES), 1)
        body[1, 4] = body[4, 4] = 0.
        body[2, 4] = -body[2, 4]
        valid = torch.tensor(np.array(stream.last_body.valid), device=device).repeat(len(MODES), 1)
        visual = stream.visual_zero.expand(len(MODES), -1)
        visual_valid = stream.visual_valid.expand(len(MODES), -1)
        goals = stream.goals.expand(len(MODES), -1)
        features = stream.last_features.expand(len(MODES), -1)
        self.held = stream.flight._drive_tensor(visual, visual_valid, body, valid, goals, features)
        # Existing declared lesion zeros normalized scene evidence, retaining its learned bias.
        # It is an adapter intervention, not a biological blindness or retinal lesion claim.
        self.held[3:] = stream.flight._drive_tensor(visual[3:], visual_valid[3:], body[3:],
            valid[3:], goals[3:], features[3:], lesion='scene_command')
        self.rows.append(output)
        self.previous = clock
        return output
