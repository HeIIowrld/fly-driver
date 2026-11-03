"""Training-only preservation of a frozen policy on explicit successful episodes."""

import copy
import math

import torch

from .multiview_learning import episode_chunk, replay_chunk, supervised_selection_score


@torch.no_grad()
def prepare_policy_anchors(model, episodes, episode_hashes, chunk, ticks):
    if (not isinstance(episode_hashes, list) or not episode_hashes
            or len(set(episode_hashes)) != len(episode_hashes)):
        raise ValueError('Explicit unique successful episode hashes required')
    selected = [e for e in episodes if e['source']['sha256'] in episode_hashes]
    if (len(selected) != len(episode_hashes)
            or {e['source']['sha256'] for e in selected} != set(episode_hashes)):
        raise ValueError('Successful anchor episodes must occur exactly once in training')
    # CSR storage cannot be deep-copied by PyTorch; omit derived caches without
    # mutating the learner. The frozen reference rebuilds them from its copied COO.
    circuit = getattr(model, 'circuit', None)
    memo = {id(value): None for name in ('_cached_matrix', '_cached_transpose')
            if torch.is_tensor(value := getattr(circuit, name, None))}
    reference = copy.deepcopy(model, memo).eval().requires_grad_(False)
    if hasattr(reference.circuit, '_cache_key'):
        reference.circuit._cache_key = None
    anchors = {}
    for episode in selected:
        state = reference.initial_state(1)
        predictions = []
        for offset in range(0, len(episode['action']), chunk):
            obs, _, _, active = episode_chunk([episode], offset, chunk)
            prediction, state = replay_chunk(reference, obs, active, state, ticks)
            predictions.append(prediction[:, 0].detach())
        target = torch.cat(predictions)[:len(episode['action'])].clone()
        anchors[episode['source']['sha256']] = target
    return anchors


def policy_anchor_loss(prediction, batch, offset, anchors):
    """Sum chunk contributions to one mean over all declared anchor decisions/actions."""
    denominator = sum(value.numel() for value in anchors.values())
    if not denominator:
        return prediction.new_zeros(())
    loss = prediction.new_zeros(())
    for column, episode in enumerate(batch):
        target = anchors.get(episode['source']['sha256'])
        if target is None:
            continue
        target = target[offset:offset+len(prediction)]
        if len(target):
            loss = loss + (prediction[:len(target), column]-target).square().sum()/denominator
    return loss


@torch.no_grad()
def replay_anchor_mse(model, episodes, anchors, chunk, ticks):
    error = model.initial_state(1).new_zeros(())
    for episode in episodes:
        if episode['source']['sha256'] not in anchors:
            continue
        state = model.initial_state(1)
        for offset in range(0, len(episode['action']), chunk):
            obs, _, _, active = episode_chunk([episode], offset, chunk)
            prediction, state = replay_chunk(model, obs, active, state, ticks)
            error += policy_anchor_loss(prediction, [episode], offset, anchors)
    return float(error)


def anchored_selection_score(metrics, coefficient):
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 100:
        raise ValueError('Finite policy anchor coefficient in [0,100] required')
    score = supervised_selection_score(metrics)
    if coefficient:
        error = metrics.get('policy_anchor_mse')
        if error is None or not math.isfinite(error) or error < 0:
            raise ValueError('Declared successful-policy preservation score required')
        score += coefficient*error
    return score
