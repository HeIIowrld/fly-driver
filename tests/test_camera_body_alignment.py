import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flybrain_drive.camera_body_alignment import fundamental,sampson_pixel_errors,optical_camera


def project(points,camera):
    k,r,c=camera;p=(points-c)@r.T@k.T
    return p[:,:2]/p[:,2:3]


def test_pixel_geometry_identifies_real_pose_and_rejects_shifted_rotation():
    rng=np.random.default_rng(780203)
    xyz=rng.uniform([-3,-2,8],[3,2,80],size=(200,3))
    k=np.array([[170,0,160],[0,170,90],[0,0,1.]])
    first=(k,np.eye(3),np.zeros(3))
    second=(k,Rotation.from_rotvec([0,.04,0]).as_matrix(),np.array([0.,0.,1.]))
    a,b=project(xyz,first),project(xyz,second)
    assert sampson_pixel_errors(a,b,fundamental(first,second)).max()<1e-10
    wrong=(k,Rotation.from_rotvec([.12,.16,0]).as_matrix(),second[2])
    # Wrong geometry exceeds the registered 2px residual ceiling.
    assert np.median(sampson_pixel_errors(a,b,fundamental(first,wrong)))>2
    offset=np.array([256.,-4.,512.])
    shifted1=(k,first[1],first[2]+offset);shifted2=(k,second[1],second[2]+offset)
    assert np.allclose(fundamental(first,second),fundamental(shifted1,shifted2))


def test_zero_baseline_is_not_reported_as_motion_synchronization():
    camera=(np.eye(3),np.eye(3),np.zeros(3))
    with pytest.raises(ValueError,match='Translating'):
        fundamental(camera,camera)


def test_recorded_row_vector_camera_basis_is_converted_to_pixel_basis():
    rotation=Rotation.from_rotvec([.1,.4,-.2]).as_matrix()
    center=np.array([50.,2.,-120.])
    view=np.eye(4);view[:3,:3]=(np.diag([1.,-1.,-1.])@rotation).T
    view[3,:3]=-center@view[:3,:3]
    projection=np.diag([170/160,170/90,1.,1.])
    stringify=lambda v:str(tuple(tuple(float(x) for x in row) for row in v))
    k,r,c=optical_camera(dict(view_matrix_diagnostic=stringify(view),
        projection_matrix_diagnostic=stringify(projection),width=320,height=180,
        renderer_only_world_position=center.tolist()))
    assert np.allclose(r,rotation) and np.array_equal(c,center)
    assert np.allclose(k,[[170,0,160],[0,170,90],[0,0,1]])
