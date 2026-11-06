from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from flybrain_drive.flyvis_camera import VIEW_ORDER
from flybrain_drive.per_view_flyvis_stream import PerViewFlyvisStream


class Projector:
    uv=np.zeros((17,2))
    def sample(self,rgb): return np.full(17,rgb[0,0,0]/255,dtype=np.float32)


class Network:
    def __init__(self): self.inputs=[]
    def simulate(self,x,*,dt,initial_state,as_states):
        self.inputs.append(x[:,0,0,:].clone())
        return [initial_state for _ in range(x.shape[1])]


class Pool(nn.Module):
    def __init__(self,old): super().__init__();self.register_buffer('weight',torch.tensor(old))
    def forward(self,x): return (x.tanh()@self.weight.T).reshape(1,576)


def make():
    old=np.concatenate([np.eye(16),np.zeros((16,1))],1).astype(np.float32)
    geo=dict(old=old,inverse=np.linalg.pinv(old),grid8_narrow=np.ones((64,17),dtype=np.float32)/17)
    state=SimpleNamespace(nodes=SimpleNamespace(activity=torch.zeros(4,136)))
    net=Network();stream=PerViewFlyvisStream(net,state,Pool(old),Projector(),np.arange(136).reshape(8,17),geo,episode_id='one',generation='g',device='cpu')
    return stream,net


def kw():
    return dict(pixels={name:np.full((180,320,3),(i+1)*20,dtype=np.uint8) for i,name in enumerate(VIEW_ORDER)},
        render_start_lowers=(1.03,1.01,1.04,1.02),camera_received=1.05,capture_id=1,generation='g')


def test_capture_order_does_not_install_future_views_early():
    stream,net=make();obs=stream.process(**kw())
    # Left is first, then rear, then front, then right. First integration
    # still holds front/right/rear at their original .5 prior.
    first=net.inputs[0][:,0].numpy()
    np.testing.assert_allclose(first,[.5,40/255,.5,.5],rtol=0,atol=1e-7)
    assert obs.render_start_lowers==(1.03,1.01,1.04,1.02)
    assert obs.scene.shape==(576,) and obs.static.shape==(320,)
    np.testing.assert_array_equal(obs.static[:64],obs.scene.reshape(4,9,16)[:,0].reshape(64))
    assert not obs.scene.flags.writeable and not obs.static.flags.writeable


@pytest.mark.parametrize('problem',['duplicate_clock','generation','blank','future'])
def test_invalid_group_does_not_advance_state_and_poison_is_explicit(problem):
    stream,net=make();args=kw()
    if problem=='duplicate_clock':args['render_start_lowers']=(1.,1.,1.01,1.02)
    if problem=='generation':args['generation']='other'
    if problem=='blank':args['pixels']['rear'][:]=0
    if problem=='future':args['camera_received']=1.035
    with pytest.raises(ValueError):stream.process(**args)
    assert not net.inputs and stream.previous is None and stream.poisoned
    with pytest.raises(RuntimeError):stream.process(**kw())


def test_duplicate_group_cannot_advance_recurrence_twice():
    stream,net=make();stream.process(**kw());before=len(net.inputs)
    with pytest.raises(ValueError):stream.process(**kw())
    assert len(net.inputs)==before and stream.poisoned


def test_speculative_prefix_does_not_commit_unobserved_time():
    stream,net=make();stream.process(**kw());state=stream.state;previous=stream.previous
    stream.prepare_held_prefix()
    assert stream.state is state and stream.previous==previous
    with pytest.raises(ValueError):stream.prepare_held_prefix()
    args=kw();args['capture_id']=2;args['render_start_lowers']=tuple(t+.04 for t in args['render_start_lowers']);args['camera_received']+=.04
    stream.process(**args)
    assert stream.previous==max(args['render_start_lowers']) and stream.prefix is None
