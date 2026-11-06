import numpy as np
import pytest
from flybrain_drive.diagnostic_road_pose import project_road_pose,paired_pose_bounds


def test_exact_closed_road_projection_and_heading_wrap():
    line=np.array([[0,0],[10,0],[10,10],[0,10],[0,0]])
    result=project_road_pose([[4,1],[10.5,4],[4,11]],[2*np.pi,np.pi/2,-np.pi],line)
    np.testing.assert_allclose(result[:,:3],[[1,0,1],[-.5,0,.5],[-1,0,1]],atol=1e-12)


def test_brackets_require_all_cross_endpoint_combinations():
    a=np.array([[[0,.01,20],[.2,.02,21]]]);b=np.array([[[.1,-.01,21],[.15,-.02,22]]])
    np.testing.assert_allclose(paired_pose_bounds(a,b),[[.15,.04,2]])


def test_heading_endpoint_distance_is_circular():
    a=np.array([[[0,np.pi-.01,1],[0,np.pi-.02,1]]]);b=np.array([[[0,-np.pi+.01,1],[0,-np.pi+.02,1]]])
    assert abs(paired_pose_bounds(a,b)[0,1]-.04)<1e-12


@pytest.mark.parametrize('line',[[[0,0],[1,0],[1,0],[0,1]],[[0,0],[1,float('nan')],[0,1]]])
def test_invalid_road_is_not_repaired(line):
    with pytest.raises(ValueError):project_road_pose([[0,0]],[0],line)
