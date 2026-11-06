import numpy as np
import pytest
from flybrain_drive.motion_spatial_training_data import duplicate_coarse_motion,training_motion_normalization


def test_coarse_control_only_duplicates_existing_motion_information():
    scene=np.arange(576,dtype=np.float32)[None];value=duplicate_coarse_motion(scene).reshape(1,4,8,8,8)
    np.testing.assert_array_equal(value[...,::2,::2],scene.reshape(1,4,9,16)[:,:,1:].reshape(1,4,8,4,4))
    np.testing.assert_array_equal(value[...,1::2,1::2],value[...,::2,::2])
    changed=scene.copy();changed.reshape(1,4,9,16)[:,:,0]=1e8
    np.testing.assert_array_equal(duplicate_coarse_motion(changed),value.reshape(1,2048))


def test_normalization_uses_only_train_supervised_camera_events():
    features=[np.array([[0.]*2048,[50.]*2048]),np.full((1,2048),2.),np.full((1,2048),1e20)]
    indices=[np.array([0,1]),np.array([0]),np.array([0])];masks=[np.array([True,False]),np.array([True]),np.array([True])]
    mean,scale=training_motion_normalization(features,indices,masks)
    np.testing.assert_array_equal(mean,np.ones(2048,dtype=np.float32));np.testing.assert_array_equal(scale,np.ones(2048,dtype=np.float32))


def test_constant_train_channel_keeps_explicit_scale_floor():
    f=[np.zeros((1,2048))]*3;j=[np.array([0])]*3;m=[np.array([True])]*3
    reference,scale=training_motion_normalization(f,j,m)
    np.testing.assert_array_equal(reference,np.zeros(2048));np.testing.assert_allclose(scale,.005)


@pytest.mark.parametrize('problem',['index','mask','nonfinite'])
def test_invalid_train_join_is_not_repaired(problem):
    f=[np.zeros((1,2048)) for _ in range(3)];j=[np.array([0]) for _ in range(3)];m=[np.array([True]) for _ in range(3)]
    if problem=='index':j[0][0]=2
    if problem=='mask':m[0]=np.array([1])
    if problem=='nonfinite':f[0][0,0]=float('nan')
    with pytest.raises(ValueError):training_motion_normalization(f,j,m)
