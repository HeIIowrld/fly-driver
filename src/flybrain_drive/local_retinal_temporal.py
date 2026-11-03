"""Episode-preserving replay and verified sensory teacher data for local retinal circuits."""

import json
from pathlib import Path

import numpy as np
import torch

from .episode_storage import load_array
from .graph import sha256
from .sensory_policy import SensoryFeatureEncoder


def load_teacher_sequences(path, calibration):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf8"))
    metadata = json.loads((path / "dataset.json").read_text(encoding="utf8"))
    if (metadata.get("teacher_beta") != 1 or metadata.get("student_state_sha256") is not None
            or manifest["config"]["policy_configuration"]["sensory_encoder"] != calibration
            or set(metadata["actor_fields"]) != SensoryFeatureEncoder.fields):
        raise ValueError("Expert-only source and matching sensory calibration required")
    episodes = []
    for record in metadata["episodes"]:
        file = (path / record["file"]).resolve()
        if not file.is_relative_to(path.resolve()) or sha256(file) != record["sha256"]:
            raise ValueError("Teacher episode path or hash differs from manifest")
        with load_array(file, allow_pickle=False) as data:
            n = len(data["action"])
            if (n != record["steps"] or not data["teacher_selected"].all()
                    or not np.isfinite(data["action"]).all()
                    or np.any(np.abs(data["action"]) > 1)
                    or not np.all(np.diff(data["observation_time"]) > 0)
                    or np.any(np.diff(data["exposure_ids"], axis=0) < 0)):
                raise ValueError("Invalid expert labels or temporal ordering")
            obs = {k[4:]: torch.tensor(data[k]) for k in data.files if k.startswith("obs_")}
            if set(obs) != SensoryFeatureEncoder.fields or any(len(v) != n for v in obs.values()):
                raise ValueError("Only declared actor fields with matching lengths are allowed")
            episodes.append({"observation": obs,
                             "label": torch.tensor(data["action"][:, 1]), "source": record})
    if not episodes or sum(len(e["label"]) for e in episodes) != metadata["decisions"]:
        raise ValueError("Nonempty dataset and exact decision count required")
    return episodes, manifest, metadata


def episode_chunk(episodes, start, length):
    """Pad ended columns with invalid inputs; never reinterpret padding as a new episode."""
    batch = len(episodes)
    if not batch or min(start, length) < 0 or length == 0:
        raise ValueError("Nonempty episodes and positive chunk length required")
    prototype = episodes[0]["observation"]
    obs = {k: v.new_zeros((length, batch, *v.shape[1:])) for k, v in prototype.items()}
    # Eye geometry is calibration, including at inactive padded steps.
    obs["ray_directions"][:] = prototype["ray_directions"][0]
    labels = episodes[0]["label"].new_zeros((length, batch))
    active = labels.new_zeros((length, batch))
    for column, episode in enumerate(episodes):
        count = min(length, max(0, len(episode["label"]) - start))
        if count:
            for key in obs:
                obs[key][:count, column] = episode["observation"][key][start:start + count]
            labels[:count, column] = episode["label"][start:start + count]
            active[:count, column] = 1
    return obs, labels, active


def replay_chunk(model, observation, active, state):
    """Numerical state passes through every active tick; gradient detach is caller-owned."""
    if (active.ndim != 2 or not torch.all((active == 0) | (active == 1))
            or any(v.shape[:2] != active.shape for v in observation.values())
            or state.shape != (active.shape[1], model.circuit.size)):
        raise ValueError("Time/batch activity and recurrent-state shapes differ")
    length, batch = active.shape
    flattened = {k: v.flatten(0, 1) for k, v in observation.items()}
    drives = model.prepare_drive(flattened).reshape(length, batch, -1)
    predictions = []
    for tick in range(length):
        prediction, updated = model.advance(drives[tick], state)
        state = torch.where(active[tick, :, None].bool(), updated, state)
        predictions.append(prediction * active[tick])
    return torch.stack(predictions), state


def replay_metrics(model, episodes, chunk_length=512):
    state = model.initial_state(len(episodes))
    errors, output_energy, count, maximum = 0., 0., 0, 0.
    traces = [[] for _ in episodes]
    with torch.no_grad():
        for start in range(0, max(len(e["label"]) for e in episodes), chunk_length):
            obs, labels, active = episode_chunk(episodes, start, chunk_length)
            prediction, state = replay_chunk(model, obs, active, state)
            errors += float(((prediction - labels).square() * active).sum())
            output_energy += float((prediction.square() * active).sum())
            count += int(active.sum())
            maximum = max(maximum, float(prediction.abs().max()))
            for column, values in enumerate(traces):
                values.append(prediction[:, column][active[:, column].bool()].numpy())
    return {"rmse": (errors / count)**.5, "output_rms": (output_energy / count)**.5,
            "max_abs_output": maximum, "decisions": count}, [np.concatenate(t) for t in traces]
