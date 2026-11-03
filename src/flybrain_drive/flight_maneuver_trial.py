"""Persistent physical-command flight component, never an autonomous visual driver."""

import torch

from .ac_image_buffer import ExistingNamedMapping
from .ac_native_physics import decode_static_identity_prefix
from .flight_actuator import canonical_controls
from .flight_maneuver import MANEUVER_CONTRACT, PhysicalManeuverCircuit
from .racing_goal import RACING_GOAL_CONTRACT, racing_goal
from .virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix


class NeuralPhysicalProbe:
    def __init__(self, spec, checkpoint, *, episode_id):
        if not torch.cuda.is_available() or not episode_id:
            raise ValueError('Explicit GPU component episode required')
        torch.set_num_threads(1)
        payload = torch.load(checkpoint, map_location='cuda', weights_only=False)
        self.model = PhysicalManeuverCircuit(spec, payload).to('cuda').eval()
        self.kwargs = dict(command_contract=MANEUVER_CONTRACT, goal_contract=RACING_GOAL_CONTRACT)
        with ExistingNamedMapping('Local\\acpmf_static', 200) as mapping:
            identity = decode_static_identity_prefix(mapping.read())
        self.body = NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
                                         native_identity=identity)
        self.episode_id, self.clock, self.held = episode_id, None, None
        self.goals = torch.tensor(racing_goal(100, 1), device='cuda')
        # Declared synthetic quiet initialization, not a fabricated sensor sample.
        quiet = torch.zeros(1, 21, device='cuda')
        valid = torch.zeros_like(quiet, dtype=torch.bool)
        valid[:, [0, 1, 4, 16]] = True
        command = torch.zeros(1, 2, device='cuda')
        with torch.no_grad():
            state = self.model.initial_state(1)
            for _ in range(100):
                _, muscles, state = self.model.advance(command, quiet, valid, self.goals,
                                                       state, .01, **self.kwargs)
            readout = self.model.backbone.readout
            turn, forward = readout.turn_log_gain.clone(), readout.forward_log_gain.clone()
            readout.calibrate_reference(muscles[0], turn_scale=1., forward_scale=1.)
            readout.turn_log_gain.copy_(turn)
            readout.forward_log_gain.copy_(forward)
            self.state = state.detach()
        self.reference = {k: v.detach().cpu().tolist()
                          for k, v in self.model.backbone.readout.state_dict().items()}

    @torch.no_grad()
    def request(self, sample, physical_command, decision_clock):
        observed = self.body.encode(decode_navigation_body_prefix(
            sample.prefix_bytes, sample.received_monotonic_seconds), episode_id=self.episode_id,
            decision_monotonic_seconds=decision_clock)
        body = torch.tensor(observed.values.copy()[None], device='cuda', dtype=torch.float32)
        mask = torch.tensor(observed.valid.copy()[None], device='cuda')
        command = torch.tensor([physical_command], device='cuda', dtype=torch.float32)
        self.model.drive(command, body, mask, self.goals, **self.kwargs)
        elapsed = None if self.clock is None else decision_clock-self.clock
        if elapsed is not None:
            previous_command, previous_body, previous_mask = self.held
            _, _, self.state = self.model.advance(previous_command, previous_body, previous_mask,
                self.goals, self.state, elapsed, **self.kwargs)
        intent, muscles = self.model.backbone._read_state_tensor(self.state)
        intent, muscles = intent[0].cpu().numpy(), muscles[0].cpu().numpy()
        control = canonical_controls(intent)
        self.clock, self.held = decision_clock, (command, body, mask)
        return control, dict(decision_clock=decision_clock, original_elapsed_seconds=elapsed,
            native_packet=sample.packet_id,
            original_native_receipt=sample.received_monotonic_seconds,
            body=observed.values.tolist(), body_valid=observed.valid.tolist(),
            physical_command=list(physical_command), intent=intent.tolist(),
            muscles=muscles.tolist(), canonical_request=control.tolist(),
            first_state='synthetic_quiet1s' if elapsed is None else None,
            current_command_held_for_next_interval=True, numerical_state_reset=False)
