"""Align consecutive front images with the locally measured vehicle motion."""

import cv2
import numpy as np


def flat_ground_points(uv, mount):
    direction=(np.asarray(mount["forward_body"])
               +(2.*uv[:, :1]-1.)*np.asarray(mount["right_body"])/mount["projection_x"]
               +(1.-2.*uv[:, 1:])*np.asarray(mount["up_body"])/mount["projection_y"])
    origin=np.asarray(mount["origin_body_m"])
    if (direction[:,2]>=-1e-6).any():
        raise ValueError("Ground warp anchors must lie below the fixed camera horizon")
    return origin-direction*(origin[2]/direction[:,2:3])


def project(points,mount):
    delta=points-mount["origin_body_m"]
    depth=delta@mount["forward_body"]
    if (depth<=0).any():
        return None
    return np.column_stack((.5+.5*mount["projection_x"]*(delta@mount["right_body"])/depth,
                            .5-.5*mount["projection_y"]*(delta@mount["up_body"])/depth))


def predicted_warp(dimensions,mount,motion):
    uv=np.array([[.22,.62],[.78,.62],[.22,.89],[.78,.89]])
    rotation,travel=motion
    previous=flat_ground_points(uv,mount)@rotation.T+travel
    old_uv=project(previous,mount)
    if old_uv is None:
        return None
    matrix=cv2.getPerspectiveTransform((uv*dimensions).astype(np.float32),
                                       (old_uv*dimensions).astype(np.float32))
    return matrix if np.isfinite(matrix).all() and abs(np.linalg.det(matrix))>1e-8 else None


def motion_aligned_tracks(current,previous,uv,mount,motion,track_pixels):
    """Track residual motion in warped pixels, then recover original prior pixels."""
    h,w=current.shape
    dims=np.array([w-1,h-1])
    matrix=predicted_warp(dims,mount,motion)
    if matrix is None:
        return uv.copy(),np.zeros(len(uv),dtype=bool)
    aligned=cv2.warpPerspective(previous,matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    warped_points,valid=track_pixels(current,aligned,uv)
    old_px=cv2.perspectiveTransform((warped_points*dims).astype(np.float32)[None],matrix)[0]
    old_uv=old_px/dims
    margin=np.minimum(old_uv,1.-old_uv).min(axis=1)
    valid &= np.isfinite(old_uv).all(axis=1)&(margin>.06)
    return old_uv,valid
