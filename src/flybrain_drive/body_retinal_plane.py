"""Body-attitude registration of retinal rays onto an explicitly virtual horizontal plane.

Axes are forward/left/up; pitch is nose-up and roll is left-up. The plane passes
through the vehicle body reference, not a measured road surface. No world pose,
heading, track map, semantic road mask or inferred depth is consumed.
"""

import numpy as np


def level_rotation(pitch, roll):
    pitch, roll = np.asarray(pitch, float), np.asarray(roll, float)
    if pitch.ndim != 1 or pitch.shape != roll.shape or not np.isfinite([pitch, roll]).all():
        raise ValueError('Finite matched pitch/roll histories required')
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    r = np.zeros((len(pitch), 3, 3))
    r[:, 0, 0], r[:, 0, 1], r[:, 0, 2] = cp, -sp*sr, -sp*cr
    r[:, 1, 1], r[:, 1, 2] = cr, -sr
    r[:, 2, 0], r[:, 2, 1], r[:, 2, 2] = sp, cp*sr, cp*cr
    return r


def plane_intersections(directions, origins, pitch, roll, attitude_valid, *,
                        forward_bounds=(-30., 30.), left_bounds=(-15., 15.)):
    directions, origins = np.asarray(directions, float), np.asarray(origins, float)
    r = level_rotation(pitch, roll)
    attitude_valid = np.asarray(attitude_valid)
    if (directions.ndim != 3 or directions.shape[-1] != 3
            or origins.shape != (directions.shape[0], 3)
            or not np.isfinite(directions).all() or not np.isfinite(origins).all()
            or not np.allclose(np.linalg.norm(directions, axis=-1), 1., atol=1e-6)
            or attitude_valid.shape != (len(r),) or attitude_valid.dtype != bool):
        raise ValueError('Registered unit rays, origins and explicit attitude validity required')
    for bounds in (forward_bounds, left_bounds):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or not bounds[0] < bounds[1]:
            raise ValueError('Finite ordered virtual plane bounds required')
    ray = np.einsum('tij,vnj->tvni', r, directions)
    origin = np.einsum('tij,vj->tvi', r, origins)
    supported = (ray[..., 2] < -1e-6) & (origin[:, :, None, 2] > 0)
    supported &= attitude_valid[:, None, None]
    distance = np.zeros_like(ray[..., 2])
    np.divide(-origin[:, :, None, 2], ray[..., 2], out=distance, where=supported)
    xy = origin[:, :, None, :2] + distance[..., None]*ray[..., :2]
    supported &= (distance > 0) & np.isfinite(xy).all(axis=-1)
    for h, bounds in enumerate((forward_bounds, left_bounds)):
        supported &= (xy[..., h] >= bounds[0]) & (xy[..., h] <= bounds[1])
    return np.where(supported[..., None], xy, np.nan), supported


def pool_on_plane(activity, xy, supported, centres, sigma, *, radius=2.5):
    """Pool actual scalar activity per camera; do not rotate directional cell types.

    All arguments are torch tensors on the same device. Unsupported bins return
    zero and a false support mask. A positive mask means ray-plane support only,
    not visible ground, road membership or independent depth validation.
    """
    import torch

    if (activity.ndim != 4 or xy.shape != (*activity.shape[:2], activity.shape[-1], 2)
            or supported.shape != xy.shape[:-1] or supported.dtype != torch.bool
            or centres.ndim != 2 or centres.shape[1] != 2 or sigma.shape != (2,)
            or not torch.isfinite(activity).all() or not torch.isfinite(centres).all()
            or not torch.isfinite(sigma).all() or not (sigma > 0).all()
            or not np.isfinite(radius) or radius <= 0
            or not torch.isfinite(xy[supported]).all()):
        raise ValueError('Finite activity, supported coordinates and pooling geometry required')
    safe = torch.where(supported[..., None], xy, torch.zeros_like(xy))
    delta = (safe[:, :, None]-centres[None, None, :, None])/sigma
    squared = delta.square().sum(-1)
    weight = torch.exp(-.5*squared)*supported[:, :, None]*(squared <= radius**2)
    mass = weight.sum(-1)
    present = mass > 1e-12
    weight = weight/mass.clamp_min(1e-12)[..., None]
    values = torch.einsum('tvcn,tvkn->tvck', activity, weight)
    return values, present


def fuse_scalar_views(values, present):
    """Equal-camera scalar fusion; missing cameras never masquerade as observations."""
    import torch

    if (values.ndim != 4 or present.shape != (*values.shape[:2], values.shape[-1])
            or present.dtype != torch.bool or not torch.isfinite(values).all()):
        raise ValueError('Per-view scalar fields and support masks required')
    count = present.sum(1)
    fused = (values*present[:, :, None]).sum(1)/count.clamp_min(1)[:, None]
    return fused, count > 0
