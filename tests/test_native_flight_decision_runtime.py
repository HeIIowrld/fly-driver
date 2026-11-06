import numpy as np
import pytest
import torch
from torch import nn
from flybrain_drive.multisensory_body_feedback_port import MultisensoryBodyFeedbackPort
from flybrain_drive.native_flight_decision_runtime import NativeFlightDecisionRuntime
from flybrain_drive.native_multisensory_extension import NativeExtensionSample, NativeExtensionObservation
from flybrain_drive.sac_unified_student import MuscleVehicleReadout
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


class Command(nn.Module):
    def forward(self,x): return x,x


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.motor=nn.Module();self.motor.command=Command()
        self.visual=nn.Linear(588,2);self.body_feedback=nn.Linear(21,2);self.count=0
    def reset_episode(self): self.count=0
    def sensory_decision(self,*,scene_features,body_context,**kwargs):
        self.visual(torch.cat([scene_features.new_zeros(1,12),scene_features],1))
        self.body_feedback(torch.tensor(body_context.decision_body.values[None],dtype=torch.float32,device=scene_features.device))
        self.motor.command(scene_features.new_zeros(1,1));self.count+=1
        return None,scene_features.new_ones(1,6)*self.count


class Encoder:
    def encode(self,sample,*,episode_id,decision_monotonic_seconds):
        return NativeExtensionObservation(episode_id,sample.packet_id,sample.received_monotonic_seconds,
            decision_monotonic_seconds,np.zeros(19),np.ones(19,dtype=bool),'h',False,(False,)*4)


def make(clock=None):
    model=Model()
    kwargs={} if clock is None else dict(monotonic=clock)
    runtime=NativeFlightDecisionRuntime(model,MuscleVehicleReadout(6),nn.Linear(320,1),nn.Linear(256,2),
        MultisensoryBodyFeedbackPort(torch.zeros(19),torch.ones(19),2),Encoder(),**kwargs)
    runtime.begin_episode('one')
    return runtime,model


def inputs(t=1.,packet=1,receipt=None):
    receipt=t-.001 if receipt is None else receipt
    body=NavigationBodyObservation('one',packet,receipt,t,np.zeros(21),np.ones(21,dtype=bool),'b')
    ctx=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(t-.02,)*4,camera_receipt_time=t-.01,decision_time=t)
    return dict(decision_time=t,scene_features=np.zeros(576),static_features=np.zeros(320),body_context=ctx,
        native_sample=NativeExtensionSample(packet,receipt,'a'*64,np.zeros(19),0.,0.,0.))


def test_recurrence_preserved_until_explicit_episode_restart():
    runtime,model=make()
    first=runtime.decide(**inputs());second=runtime.decide(**inputs(1.01,2))
    assert first['muscles'][0]==1 and second['muscles'][0]==2 and model.count==2
    runtime.begin_episode('one');assert runtime.decide(**inputs())['muscles'][0]==1
    runtime.close();assert not model.visual._forward_hooks and not model.body_feedback._forward_hooks


@pytest.mark.parametrize('problem',['packet','receipt','stale','clock','scene','episode'])
def test_input_mismatch_requires_explicit_restart(problem):
    runtime,model=make();kw=inputs()
    if problem=='packet': kw['native_sample']=NativeExtensionSample(9,.999,'a'*64,np.zeros(19),0.,0.,0.)
    if problem=='receipt': kw['native_sample']=NativeExtensionSample(1,.998,'a'*64,np.zeros(19),0.,0.,0.)
    if problem=='stale': kw=inputs(receipt=.94)
    if problem=='clock': runtime.decide(**inputs())
    if problem=='scene': kw['static_features'][0]=1
    if problem=='episode': runtime.begin_episode('two')
    before=model.count
    with pytest.raises(ValueError): runtime.decide(**kw)
    assert model.count==before and runtime.poisoned
    with pytest.raises(RuntimeError): runtime.decide(**inputs(1.02,3))
    runtime.close()


def test_delayed_output_poisoned_without_silent_rewind():
    runtime,model=make(iter([0.,.06]).__next__)
    with pytest.raises(ValueError,match='output exceeds'): runtime.decide(**inputs())
    assert model.count==1 and runtime.poisoned
    with pytest.raises(RuntimeError): runtime.decide(**inputs(1.01,2))
    runtime.close()


def test_cpu_runtime_does_not_inherit_external_default_device():
    runtime,model=make()
    # Models such as FlyVis can establish a global accelerator default. A meta
    # default exercises the same placement leak without requiring a test GPU.
    with torch.device('meta'):
        output=runtime.decide(**inputs())
    assert output['prediction'].shape==(3,) and model.count==1
    runtime.close()
