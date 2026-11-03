import json
import struct

import numpy as np
import pytest
import torch
from torch import nn

from flybrain_drive.ac_native_physics import decode_physics_prefix
from flybrain_drive.flight_geometry_feedback import CausalGeometryVisualPort
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.visual_self_motion import VisualSelfMotionPool


class Base(nn.Module):
    def forward(self, traces, body, valid):
        return torch.tensor([[.2,-.1]])


def make_port():
    with open('configs/ac-csp-yaw-retina.json',encoding='utf8') as f:
        config=json.load(f)
    uv=np.array([(q,r) for q in range(-15,16)
                 for r in range(max(-15,-15-q),min(15,15-q)+1)])
    projector=FlyvisCameraProjector(config,uv[:,0],uv[:,1])
    return CausalGeometryVisualPort(Base(),VisualSelfMotionPool(config,projector.directions_body),2)


def sample(index,speed=0.):
    raw=bytearray(308)
    struct.pack_into('<i',raw,0,index)
    struct.pack_into('<f',raw,28,speed)
    return decode_physics_prefix(bytes(raw),10.+.04*index)


def test_stationary_first_eleven_and_causal_nonzero_output():
    port=make_port(); motion=np.ones((4,8,721),dtype=np.float32)
    valid=np.ones(4,dtype=bool)
    for i in range(11):
        assert not port.observe(motion,valid,sample(i),capture_id=i,decision_time=10.+.04*i+.001,camera_receipt_time=10.+.04*i+.001)
        assert torch.equal(port(None,None,None),torch.tensor([[.2,-.1]]))
    assert port.reference.value is not None
    changed=motion.copy();changed[0,1,:]=2.
    assert port.observe(changed,valid,sample(11,speed=30),capture_id=11,decision_time=10.441,camera_receipt_time=10.441)
    assert port.current[0,2] != 0
    assert torch.equal(port(None,None,None),torch.tensor([[.2,-.1]]))
    with torch.no_grad(): port.projection.weight[0,2]=1.
    # The prior observation was consumed even though its then-neutral weight
    # happened to produce zero; it must not be reused for another decision.
    assert torch.equal(port(None,None,None),torch.tensor([[.2,-.1]]))
    assert port.observe(changed,valid,sample(12,speed=30),capture_id=12,decision_time=10.481,camera_receipt_time=10.481)
    assert not torch.equal(port(None,None,None),torch.tensor([[.2,-.1]]))
    assert torch.equal(port(None,None,None),torch.tensor([[.2,-.1]]))
    assert not port.observe(changed,np.array([True,False,True,True]),sample(13),
                            capture_id=13,decision_time=10.521,camera_receipt_time=10.521)
    assert torch.equal(port(None,None,None),torch.tensor([[.2,-.1]]))
    port.reset_episode()
    assert port.reference.value is None and port.current is None


def test_missing_and_stale_body_never_create_reference():
    port=make_port(); motion=np.ones((4,8,721),dtype=np.float32); valid=np.ones(4,dtype=bool)
    for i in range(11):
        assert not port.observe(motion,valid,sample(i,speed=30),capture_id=i,decision_time=10.+.04*i+.001,camera_receipt_time=10.+.04*i+.001)
    assert port.reference.value is None
    assert not port.observe(motion,valid,sample(11),capture_id=11,decision_time=10.6,camera_receipt_time=10.6)
    assert port.reason == 'body_stale'
    assert not port.observe(motion,valid,sample(12),capture_id=11,decision_time=10.601,camera_receipt_time=10.601)
    assert port.reason == 'repeated_camera_capture'
    with pytest.raises(ValueError,match='Nondecreasing'):
        port.observe(motion,valid,sample(12),capture_id=10,decision_time=10.641,camera_receipt_time=10.641)


def test_duplicate_native_packet_does_not_refresh_original_receipt():
    port=make_port(); motion=np.ones((4,8,721),dtype=np.float32); valid=np.ones(4,dtype=bool)
    first=sample(0)
    assert not port.observe(motion,valid,first,capture_id=0,decision_time=10.001,camera_receipt_time=10.001)
    restamped=decode_physics_prefix(first.prefix_bytes,10.1)
    assert not port.observe(motion,valid,restamped,capture_id=1,decision_time=10.101,camera_receipt_time=10.101)
    assert port.reason == 'body_stale'
    raw=bytearray(first.prefix_bytes);struct.pack_into('<f',raw,28,1.)
    changed=decode_physics_prefix(bytes(raw),10.11)
    with pytest.raises(ValueError,match='different original bytes'):
        port.observe(motion,valid,changed,capture_id=2,decision_time=10.111,camera_receipt_time=10.111)


def test_camera_freshness_resets_partial_reference_and_disables_correction():
    port=make_port(); motion=np.ones((4,8,721),dtype=np.float32); valid=np.ones(4,dtype=bool)
    for i in range(3):
        now=10.+.04*i+.001
        assert not port.observe(motion,valid,sample(i),capture_id=i,decision_time=now,camera_receipt_time=now-.001)
    assert len(port.reference.rows)==3
    assert not port.observe(motion,valid,sample(3),capture_id=3,decision_time=10.121,camera_receipt_time=10.06)
    assert port.reason=='camera_stale' and not port.reference.rows and port.current is None
    assert not port.observe(motion,valid,sample(4),capture_id=4,decision_time=10.161,camera_receipt_time=10.162)
    assert port.reason=='camera_future' and not port.reference.rows
    with pytest.raises(ValueError,match='camera receipt'):
        port.observe(motion,valid,sample(5),capture_id=5,decision_time=10.201,camera_receipt_time=float('nan'))
    port.reset_episode()
    for i in range(11):
        now=10.+.04*i+.001
        port.observe(motion,valid,sample(i),capture_id=i,decision_time=now,camera_receipt_time=now-.001)
    assert port.reference.value is not None
    assert not port.observe(motion,valid,sample(11),capture_id=11,decision_time=10.441,camera_receipt_time=10.38)
    assert port.reason=='camera_stale' and port.reference.value is not None and port.current is None
    assert torch.equal(port(None,None,None),torch.tensor([[.2,-.1]]))
