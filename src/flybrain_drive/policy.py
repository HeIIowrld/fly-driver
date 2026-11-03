"""Persistent recurrent actor/critic. Graph actor reads only explicit output ports."""

import numpy as np
import torch
from torch import nn


def tensor_observation(observation, device=None):
    return {
        key: torch.as_tensor(value, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


class Encoder(nn.Module):
    def __init__(self, visual_schema="legacy", motion_input=False):
        super().__init__()
        if visual_schema not in ("legacy", "retina_v1"):
            raise ValueError("Unknown encoder visual schema")
        if motion_input and visual_schema != "retina_v1":
            raise ValueError("Motion input requires retina_v1")
        self.visual_schema, self.motion_input = visual_schema, motion_input
        self.vision = nn.Sequential(
            nn.Conv2d(6 if visual_schema == "legacy" else 18, 16, 5, stride=2),
            nn.Tanh(),
            nn.Conv2d(16, 16, 3, stride=2),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        extras = int(visual_schema == "retina_v1")
        self.fusion = nn.Sequential(nn.Linear(64 + 38 * 2 + 4 + extras, 64), nn.Tanh())

    def forward(self, observation):
        images = observation["images"].float() / 255
        camera_valid = observation["camera_valid"].float()
        images = images * camera_valid[:, :, None, None, None]
        if self.visual_schema == "retina_v1":
            spatial = observation["retina_valid"].float() * camera_valid[:, :, None, None]
            images = images * spatial[..., None]
            motion = observation["motion"].float() * observation["motion_valid"].float()
            motion = motion * spatial[..., None]
            if not self.motion_input:
                motion = torch.zeros_like(motion)
            images = torch.cat((images, motion), dim=-1)
        images = images.permute(0, 1, 4, 2, 3).flatten(1, 2)
        vision = self.vision(images)
        mask = observation["body_valid"].float()
        body = observation["body"].float() * mask
        fields = [vision, body, mask, camera_valid, observation["camera_age"].float()]
        if self.visual_schema == "retina_v1":
            fields.append(observation["previous_yaw_action"].float())
        return self.fusion(torch.cat(fields, dim=1))


class SparseCircuit(nn.Module):
    def __init__(self, graph, dt):
        super().__init__()
        self.size = len(graph["ids"])
        type_names, type_ids = np.unique(graph["types"], return_inverse=True)
        t = len(type_names)
        pre, post = graph["pre"], graph["post"]
        self.register_buffer("pre", torch.tensor(pre))
        self.register_buffer("post", torch.tensor(post))
        self.register_buffer("node_type", torch.tensor(type_ids))
        self.register_buffer("pair_type", torch.tensor(type_ids[pre] * t + type_ids[post]))
        weights = graph["weights"].astype(float)
        incoming = np.bincount(post, weights=weights, minlength=self.size)
        normalized = weights / np.maximum(incoming[post], 1)
        self.register_buffer("structural_weight", torch.tensor(normalized, dtype=torch.float32))
        self.register_buffer("input_ports", torch.tensor(np.flatnonzero(graph["roles"] == "input")))
        input_mask = torch.tensor(graph["roles"] == "input")
        self.register_buffer("input_mask", input_mask, persistent=False)
        self.register_buffer(
            "output_ports", torch.tensor(np.flatnonzero(graph["roles"] == "output"))
        )
        self.gain = nn.Parameter(torch.ones(t * t))
        self.log_tau = nn.Parameter(torch.full((t,), -2.6))
        self.bias = nn.Parameter(torch.zeros(t))
        self.input_adapter = nn.Linear(64, len(self.input_ports))
        self.dt = dt

    def forward(self, embedding, state):
        injection = torch.zeros_like(state).index_add(
            1, self.input_ports, self.input_adapter(embedding)
        )
        return self.step_with_drive(injection, state)

    def step_with_drive(self, injection, state):
        """Advance persistent state with explicit input-only neuronal drive."""
        if (state.ndim != 2 or state.shape[1] != self.size or injection.shape != state.shape
                or not torch.isfinite(injection).all()
                or torch.any(injection[:, ~self.input_mask])):
            raise ValueError("External neuronal drive may enter only declared input ports")
        weights = self.structural_weight * self.gain[self.pair_type]
        message = torch.zeros_like(state).index_add(1, self.post, state[:, self.pre] * weights)
        tau = 0.01 + nn.functional.softplus(self.log_tau[self.node_type])
        alpha = 1 - torch.exp(-self.dt / tau)
        state = (1 - alpha) * state + alpha * torch.tanh(
            message + injection + self.bias[self.node_type]
        )
        return state[:, self.output_ports], state


class RecurrentPolicy(nn.Module):
    def __init__(
        self, kind="gru", graph=None, dt=0.01, action_size=2,
        visual_schema="legacy", motion_input=False, tv_active=True, critic_feature_grad=True,
        sensory_encoder=None,
    ):
        super().__init__()
        if action_size not in (2, 3):
            raise ValueError("Only two-intent or three-intent policies are supported")
        if sensory_encoder is None:
            self.encoder = Encoder(visual_schema, motion_input)
        else:
            from .sensory_policy import SensoryFeatureEncoder

            self.encoder = SensoryFeatureEncoder(**sensory_encoder)
        if kind == "gru":
            self.core = nn.GRUCell(64, 64)
            self.state_size = self.output_size = 64
        elif kind == "graph" and graph is not None:
            self.core = SparseCircuit(graph, dt)
            self.state_size = self.core.size
            self.output_size = len(self.core.output_ports)
        else:
            raise ValueError("Policy must be gru or graph with an explicit graph")
        self.kind = kind
        self.critic_feature_grad = bool(critic_feature_grad)
        self.actor = nn.Linear(self.output_size, action_size)
        self.critic = nn.Linear(self.output_size, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))
        mask = torch.ones(action_size)
        if action_size == 3:
            with torch.no_grad():
                self.actor.weight[2].zero_()
                self.actor.bias[2].zero_()
            mask[2] = float(tv_active)
        self.register_buffer("action_mask", mask, persistent=False)

    def initial_state(self, batch=1):
        return torch.zeros(batch, self.state_size, device=self.log_std.device)

    def tick(self, embedding, state, reset=None):
        if reset is not None:
            state = state * (1 - reset.float().reshape(-1, 1))
        if self.kind == "gru":
            state = self.core(embedding, state)
            output = state
        else:
            output, state = self.core(embedding, state)
        critic_input = output if self.critic_feature_grad else output.detach()
        return self.actor(output) * self.action_mask, self.critic(critic_input).squeeze(-1), state

    def forward(self, observation, state, reset=None):
        return self.tick(self.encoder(observation), state, reset)

    def sequence(self, observation, reset):
        """Full history recomputed with current weights, starting at a declared reset."""
        embeddings = self.encoder(observation)
        state = self.initial_state()
        means, values = [], []
        for i in range(len(embeddings)):
            mean, value, state = self.tick(embeddings[i : i + 1], state, reset[i : i + 1])
            means.append(mean[0])
            values.append(value[0])
        return torch.stack(means), torch.stack(values)

    def sequence_batch(self, observation, reset, initial_state=None, return_state=False):
        """Replay T×N trajectories; a detached boundary state supports truncated BPTT."""
        length, count = reset.shape
        flattened = {key: value.flatten(0, 1) for key, value in observation.items()}
        embeddings = self.encoder(flattened).reshape(length, count, -1)
        state = self.initial_state(count) if initial_state is None else initial_state
        if state.shape != (count, self.state_size):
            raise ValueError("Initial recurrent state has the wrong batch/state shape")
        means, values = [], []
        for tick in range(length):
            mean, value, state = self.tick(embeddings[tick], state, reset[tick])
            means.append(mean)
            values.append(value)
        result = (torch.stack(means), torch.stack(values))
        return (*result, state) if return_state else result


def policy_from_config(config, graph=None):
    environment = config.get("environment", {})
    mode = environment.get("actuation_mode", "legacy")
    policy = RecurrentPolicy(
        config.get("policy", "gru"), graph, environment.get("control_dt", 0.01),
        action_size=2 if mode == "legacy" else 3,
        visual_schema=environment.get("visual_schema", "legacy"),
        motion_input=bool(config.get("motion_input", False)),
        tv_active=mode == "residual",
        critic_feature_grad=config.get("critic_feature_grad", True),
        sensory_encoder=config.get("sensory_encoder"),
    )
    initial_std = config.get("initial_log_std", -0.5)
    values = torch.as_tensor(initial_std, dtype=torch.float32).expand_as(policy.log_std)
    if not torch.isfinite(values).all() or torch.any((values < -5) | (values > 1)):
        raise ValueError("initial_log_std must be finite and between -5 and 1")
    with torch.no_grad():
        policy.log_std.copy_(values)
    return policy


def log_probability(mean, log_std, latent, mask=None):
    distribution = torch.distributions.Normal(mean, log_std.clamp(-5, 1).exp())
    # Stable Jacobian of tanh, including actions arbitrarily close to saturation.
    correction = 2 * (np.log(2) - latent - nn.functional.softplus(-2 * latent))
    result = distribution.log_prob(latent) - correction
    return (result if mask is None else result * mask).sum(-1)
