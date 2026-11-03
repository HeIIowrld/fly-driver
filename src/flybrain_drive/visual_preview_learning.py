"""Training-only moments for a visual auxiliary task; no controls, body or goals as inputs."""

import numpy as np


def training_moments(episodes, *, include_spatial, feature_floor, target_floor, feature_count=576):
    if any(e['role'] not in ('train', 'development') for e in episodes):
        raise ValueError('No evaluation access or implicit role conversion')
    selected = [e for e in episodes if e['role'] == 'train'
                and (include_spatial or e['group'] == 'original')]
    if not selected:
        raise ValueError('Complete declared training histories required')
    x, y, valid = (np.concatenate([e[k] for e in selected]) for k in ('x', 'y', 'valid'))
    floor = np.asarray(target_floor, float)
    if (feature_count not in (576, 896) or x.ndim != 2 or x.shape[1] != feature_count
            or y.shape != (len(x), 6)
            or valid.shape != y.shape or valid.dtype != bool or floor.shape != (6,)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not valid.any(0).all() or not np.isfinite(floor).all() or not np.all(floor > 0)
            or not np.isfinite(feature_floor) or not feature_floor > 0):
        raise ValueError('Finite visual-only arrays and valid six-head preview targets required')
    return dict(x=x, y=y, valid=valid, feature_mean=x.mean(0, dtype=float),
        feature_scale=x.astype(float).std(0).clip(feature_floor),
        target_mean=np.array([y[valid[:, h], h].mean() for h in range(6)]),
        target_scale=np.maximum([y[valid[:, h], h].std() for h in range(6)], floor),
        fitting_histories=[e['id'] for e in selected])
