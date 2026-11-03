"""Replace only supervised labels; keep measured actor arrays and clocks bit-for-bit."""

import numpy as np

from .multiview_sensory import MultiViewPacker


def relabel_episode(arrays, actions):
    expected = {'obs_'+key for key in MultiViewPacker.fields} | {
        'action', 'action_valid', 'observation_time'}
    if set(arrays) != expected:
        raise ValueError('Only sealed pose-free actor arrays and action metadata accepted')
    actions = np.asarray(actions, dtype=np.float32)
    if (actions.shape != arrays['action'].shape or actions.ndim != 2 or actions.shape[1] != 3
            or not np.isfinite(actions).all() or np.any(np.abs(actions[:, 0]) > 1)
            or np.any((actions[:, 1:] < 0) | (actions[:, 1:] > 1))):
        raise ValueError('One bounded native three-head label per recorded decision required')
    result = {key: value.copy() for key, value in arrays.items()}
    result['action'] = actions.copy()
    return result
