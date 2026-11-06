"""Diagnostic pixel/pose checks; renderer truth is never an actor feature."""
import io

import numpy as np
from PIL import Image

from .ac_csp_geometry import matrix


def optical_camera(view):
    """Recorded OpenGL view basis -> top-left OpenCV pixel basis."""
    v=matrix(view['view_matrix_diagnostic']);p=matrix(view['projection_matrix_diagnostic'])
    w,h=view['width'],view['height']
    # Row-vector view matrices store world-space basis vectors as columns.
    rotation=np.diag([1.,-1.,-1.])@v[:3,:3].T
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=5e-5):
        raise ValueError('Recorded camera basis is not orthonormal')
    center=np.asarray(view['renderer_only_world_position'],dtype=float)
    intrinsics=np.array([[p[0,0]*w/2,0,w/2],[0,p[1,1]*h/2,h/2],[0,0,1.]])
    return intrinsics,rotation,center


def fundamental(camera_one,camera_two):
    k1,r1,c1=camera_one;k2,r2,c2=camera_two
    relative=r2@r1.T;translation=r2@(c1-c2)
    if np.linalg.norm(translation)<1e-5:
        raise ValueError('Translating camera baseline required for epipolar check')
    x,y,z=translation
    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.linalg.inv(k2).T@skew@relative@np.linalg.inv(k1)


def sampson_pixel_errors(first,second,f):
    first=np.asarray(first,dtype=float);second=np.asarray(second,dtype=float)
    if first.shape!=second.shape or first.ndim!=2 or first.shape[1]!=2:
        raise ValueError('Matched pixel coordinates required')
    a=np.c_[first,np.ones(len(first))];b=np.c_[second,np.ones(len(second))]
    fa=a@f.T;ftb=b@f
    denominator=(fa[:,:2]**2).sum(1)+(ftb[:,:2]**2).sum(1)
    if np.any(denominator<=1e-20):
        raise ValueError('Degenerate epipolar observations')
    return abs((b*fa).sum(1))/np.sqrt(denominator)


def image_gray(content):
    with Image.open(io.BytesIO(content)) as image:
        pixels=np.asarray(image.convert('L'))
    if pixels.shape!=(180,320):
        raise ValueError('Original 320x180 source image required')
    return pixels


def tracked_pixels(first,second):
    import cv2
    mask=np.zeros_like(first);mask[8:145,8:312]=255
    points=cv2.goodFeaturesToTrack(first,maxCorners=500,qualityLevel=.01,minDistance=4,
        mask=mask,blockSize=5)
    if points is None or len(points)<30:
        return np.empty((0,2)),np.empty((0,2))
    settings=dict(winSize=(21,21),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    moved,status,_=cv2.calcOpticalFlowPyrLK(first,second,points,None,**settings)
    if moved is None or status is None:
        return np.empty((0,2)),np.empty((0,2))
    returned,back_status,_=cv2.calcOpticalFlowPyrLK(second,first,moved,None,**settings)
    if returned is None or back_status is None:
        return np.empty((0,2)),np.empty((0,2))
    a=points[:,0];b=moved[:,0]
    valid=(status[:,0]!=0)&(back_status[:,0]!=0)&(np.linalg.norm(a-returned[:,0],axis=1)<=.75)
    valid&=(b[:,0]>=0)&(b[:,0]<320)&(b[:,1]>=0)&(b[:,1]<180)
    return a[valid],b[valid]


def source_render_witness(view):
    before=view['source_body_before_render'];after=view['source_body_after_render']
    clock=[before['source_clock_started'],before['source_clock_finished'],
        view['render_started_clock'],view['render_call_finished_clock'],
        after['source_clock_started'],after['source_clock_finished']]
    if (not np.isfinite(clock).all() or any(a>b for a,b in zip(clock,clock[1:]))
            or before['schema']!='flybrain.csp.source_body.render_witness.v1'
            or after['schema']!=before['schema'] or before['physics_available'] is not True
            or after['physics_available'] is not True
            or before['physics_timestamp_ms']!=after['physics_timestamp_ms']
            or not 0<=before['sim_time_ms']-before['physics_timestamp_ms']<=5
            or not 0<=after['sim_time_ms']-after['physics_timestamp_ms']<=5
            or clock[3]-clock[2]>.005):
        raise ValueError('Body source timestamp/render bracket not qualified')
    for key in ('local_velocity','local_angular_velocity','acceleration_g'):
        a,b=np.asarray(before[key]),np.asarray(after[key])
        if a.shape!=(3,) or not np.isfinite(a).all() or not np.array_equal(a,b):
            raise ValueError('Source body changed inside an equal-timestamp render bracket')
    return dict(render_seconds=clock[3]-clock[2],
        physics_timestamp_ms=before['physics_timestamp_ms'],
        sim_physics_lag_ms=before['sim_time_ms']-before['physics_timestamp_ms'])


def preserved_source_render_metadata(view):
    """Carry original render/body evidence to online diagnostics without admitting it."""
    import copy
    keys=('source_body_before_render','source_body_after_render','render_started_clock','render_call_finished_clock')
    present=[key in view for key in keys]
    # Legacy camera metadata still has render clocks without source witnesses.
    if not any(present[:2]):return None
    if not all(present):raise ValueError('Partial source render/body metadata')
    return {key:copy.deepcopy(view[key]) for key in keys}
