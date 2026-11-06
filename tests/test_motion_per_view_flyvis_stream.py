from dataclasses import replace
import numpy as np
import pytest
import torch
from test_per_view_flyvis_stream import make,kw
from flybrain_drive.motion_per_view_flyvis_stream import MotionPerViewFlyvisStream,MotionVisualFlightObservation


def new_stream():
    old,network=make();return MotionPerViewFlyvisStream(old.network,old.state,old.pool,old.projector,old.indices,old.geometry,episode_id=old.episode_id,generation=old.generation,device='cpu'),network


def test_motion_extension_keeps_original_scene_static_and_neural_schedule():
    baseline,n1=make();motion,n2=new_stream()
    a=baseline.process(**kw());b=motion.process(**kw())
    np.testing.assert_array_equal(a.scene,b.scene);np.testing.assert_array_equal(a.static,b.static)
    assert len(n1.inputs)==len(n2.inputs) and motion.previous==baseline.previous
    assert b.motion_features.shape==(2048,) and b.coarse_motion_reference.shape==(512,)
    assert not b.motion_features.flags.writeable


def test_real_nullspace_motion_is_retained_without_changing_original_coarse_signal():
    stream,_=new_stream()
    stream.state.nodes.activity[:,stream.indices[:,16]]=1.
    obs=stream.process(**kw())
    assert obs.motion_features.max()>0
    assert not obs.coarse_motion_reference.any()
    assert not obs.scene.reshape(4,9,16)[:,1:].any()


def test_hash_mismatch_cannot_silently_repair_motion_transport():
    stream,_=new_stream();obs=stream.process(**kw())
    with pytest.raises(ValueError):replace(obs,motion_features=np.ones(2048,dtype=np.float32))


def test_motion_extraction_does_not_commit_speculative_prefix():
    stream,_=new_stream();stream.process(**kw());original=stream.state;stamp=stream.previous
    stream.prepare_held_prefix();assert stream.state is original and stream.previous==stamp
    args=kw();args.update(capture_id=2,render_start_lowers=tuple(t+.04 for t in args['render_start_lowers']),camera_received=args['camera_received']+.04)
    stream.process(**args);assert stream.prefix is None
