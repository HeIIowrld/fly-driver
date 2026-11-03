import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from flybrain_drive.flight_maneuver import MANEUVER_CONTRACT, ManeuverCurrents, PhysicalManeuverCircuit
from flybrain_drive.flight_body_feedback_adapter import CenteredVehicleBodyFeedbackAdapter
from flybrain_drive.flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from flybrain_drive.flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT
from flybrain_drive.unified_flight_sensorimotor import UnifiedFlightSensorimotor
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation


class SyntheticBackbone(nn.Module):
    """Small synthetic graph fixture, never a measured connectome."""
    def __init__(self):
        super().__init__()
        self.neurons = 8
        self.spec = {'dynamics': {'dt': .005}}
        for key, values in [('vision_ports',[0,1]),('body_ports',[2]),
                            ('steering_ports',[3,4,5,6]),('goal_ports',[7])]:
            self.register_buffer(key,torch.tensor(values))
        self.register_buffer('visual_scale',torch.ones(6))
        self.body = nn.Linear(6,1)
        self.vision = nn.Linear(588,2)
        self.scene_pool = nn.Identity()
        self.circuit = nn.Module()
        self.circuit.register_buffer('matrix',torch.eye(8).to_sparse())

    def _advance_elapsed_tensor(self,drive,state,dt):
        return state + drive*dt

    def _read_state_tensor(self,state):
        return state[:,:2],state[:,2:4]


class SyntheticMotor(PhysicalManeuverCircuit):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = SyntheticBackbone()
        self.neurons = 8
        self.command = ManeuverCurrents([1, -1, 1, -1], 1)

    def initial_state(self,batch):
        return torch.zeros(batch,8)

    def drive(self,command,body,valid,goals,**kwargs):
        if not valid[:,[4,16]].all() or command[0,1] > goals[0,1]*100/3.6:
            raise ValueError('bad physical input')
        x = torch.zeros(len(body),8)
        x[:,2] = .1
        x[:,7] = command[:,1]/30
        return x


def context(now, speed=0.):
    values = np.zeros(21,dtype=np.float64)
    values[16] = speed
    valid = np.ones(21,dtype=bool)
    body = NavigationBodyObservation('synthetic',1,now-.01,now,values,valid,'fixture')
    return per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.03,)*4,
        camera_receipt_time=now-.02,decision_time=now)


def make_model(body_feedback=None):
    motor = SyntheticMotor()
    visual = copy.deepcopy(motor.backbone)
    with torch.no_grad():
        visual.vision.weight.zero_()
        visual.vision.weight[:,12] = 1
        visual.vision.bias.zero_()
    temporal = OpponentTemporalMotionBoundary(torch.zeros(576),torch.ones(576),2)
    boundary = PerViewVisualBodyContextBoundary(2)
    return UnifiedFlightSensorimotor(motor,visual,temporal,boundary,body_feedback=body_feedback)


def test_optional_centered_body_boundary_removes_full_validity_offset_and_marks_loss():
    centered = make_model(CenteredVehicleBodyFeedbackAdapter(1))
    legacy = make_model()
    legacy.load_state_dict(centered.state_dict())
    with torch.no_grad():
        centered.body_feedback.projection.weight[:, 14:].fill_(1.)
        legacy.body_feedback.projection.weight.copy_(centered.body_feedback.projection.weight)
    scene = torch.zeros(1,576)
    invoke(centered,1.,scene); invoke(legacy,1.,scene)
    assert centered.previous_drive[0,2] < legacy.previous_drive[0,2]
    available = centered.previous_drive[0,2].clone()
    centered.reset_episode()
    ctx = context(1.)
    ctx.decision_body.valid[0] = False
    centered.decision(decision_time=1.,command=torch.tensor([[0.,100/3.6]]),
        racing_goals=torch.tensor([[0.,1.]]),scene_features=scene,
        scene_valid=torch.ones(1,4,dtype=torch.bool),body_context=ctx)
    assert centered.previous_drive[0,2] < available
    with pytest.raises(ValueError, match='body feedback boundary'):
        make_model(CenteredVehicleBodyFeedbackAdapter(2))


def invoke(model,now,scene):
    return model.decision(decision_time=now,command=torch.tensor([[0.,100/3.6]]),
        racing_goals=torch.tensor([[0.,1.]]),scene_features=scene,
        scene_valid=torch.ones(1,4,dtype=torch.bool),body_context=context(now))


def test_one_state_high_speed_and_visual_effect_with_causal_hold():
    a,b = make_model(),make_model()
    b.load_state_dict(a.state_dict())
    scene = torch.zeros(1,576)
    changed = scene.clone(); changed[0,0] = 1
    invoke(a,1.,scene); invoke(b,1.,changed)
    assert torch.equal(a.state,b.state)  # first decision does not invent elapsed time
    intent_a,_ = invoke(a,1.04,scene)
    intent_b,_ = invoke(b,1.04,changed)
    assert not torch.equal(intent_a,intent_b)
    assert a.state.shape == (1,8) and a.state[0,7] > 0
    old_state = a.state.clone()
    with pytest.raises(ValueError,match='bad physical input'):
        a.decision(decision_time=1.08,command=torch.tensor([[0.,101/3.6]]),
            racing_goals=torch.tensor([[0.,1.]]),scene_features=scene,
            scene_valid=torch.ones(1,4,dtype=torch.bool),body_context=context(1.08))
    assert torch.equal(a.state,old_state) and a.previous_time == 1.04
    with pytest.raises(ValueError,match='advancing'):
        invoke(a,1.04,scene)
    a.reset_episode()
    assert a.state is None and a.previous_time is None


def test_graph_mismatch_is_rejected():
    motor = SyntheticMotor(); visual = copy.deepcopy(motor.backbone)
    visual.vision_ports[0] = 7
    with pytest.raises(ValueError,match='ports differ'):
        UnifiedFlightSensorimotor(motor,visual,
            OpponentTemporalMotionBoundary(torch.zeros(576),torch.ones(576),2),
            PerViewVisualBodyContextBoundary(2))


def test_late_failure_does_not_partially_advance_the_state(monkeypatch):
    model = make_model()
    scene = torch.zeros(1,576)
    invoke(model,1.,scene)
    old_state, old_time = model.state.clone(),model.previous_time
    def fail(*args,**kwargs):
        raise ValueError('injected context failure')
    monkeypatch.setattr(model.view_context,'forward',fail)
    with pytest.raises(ValueError,match='injected'):
        invoke(model,1.04,scene)
    assert torch.equal(model.state,old_state) and model.previous_time == old_time


def test_training_retains_one_state_gradient_and_truncation_preserves_values():
    model = make_model()
    scene = torch.zeros(1,576)
    args = dict(command=torch.tensor([[0.,100/3.6]]), racing_goals=torch.tensor([[0.,1.]]),
        scene_features=scene, scene_valid=torch.ones(1,4,dtype=torch.bool))
    model.training_decision(decision_time=1.,body_context=context(1.),**args)
    _,muscles = model.training_decision(decision_time=1.04,body_context=context(1.04),**args)
    loss = (muscles[0,0]-.02).square()
    loss.backward()
    grad = model.body_feedback.projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    old = model.state.detach().clone()
    drive = model.previous_drive.detach().clone()
    model.detach_training_state()
    assert torch.equal(model.state,old) and torch.equal(model.previous_drive,drive)
    assert model.state.grad_fn is None and model.previous_drive.grad_fn is None
    assert model.previous_time == 1.04
    model.training_decision(decision_time=1.08,body_context=context(1.08),**args)
    assert model.previous_drive.grad_fn is not None
    model.training_decision(decision_time=1.12,body_context=context(1.12),**args)
    assert model.state.grad_fn is not None


def test_research_training_matches_inference_and_can_fit_synthetic_target():
    training,inference = make_model(),make_model()
    inference.load_state_dict(training.state_dict())
    scene = torch.zeros(1,576)
    args = dict(command=torch.tensor([[0.,100/3.6]]), racing_goals=torch.tensor([[0.,1.]]),
        scene_features=scene, scene_valid=torch.ones(1,4,dtype=torch.bool))
    for now in (1.,1.04,1.08):
        trained = training.training_decision(decision_time=now,body_context=context(now),**args)
        ordinary = inference.decision(decision_time=now,body_context=context(now),**args)
        assert all(torch.equal(a,b) for a,b in zip(trained,ordinary))
    optimizer = torch.optim.SGD(training.body_feedback.parameters(),lr=10.)
    losses = []
    for _ in range(8):
        training.reset_episode()
        optimizer.zero_grad()
        training.training_decision(decision_time=1.,body_context=context(1.),**args)
        _,muscles = training.training_decision(decision_time=1.04,body_context=context(1.04),**args)
        loss = (muscles[0,0]-.02).square()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_episode_restart_resets_stateful_visual_reference():
    model=make_model()
    class StatefulContext(nn.Module):
        def __init__(self):
            super().__init__()
            self.reference='previous episode'
        def reset_episode(self):
            self.reference=None
    context_boundary=StatefulContext()
    model.view_context=context_boundary
    model.reset_episode()
    assert context_boundary.reference is None
    assert model.state is None and model.previous_time is None


def test_sensory_path_matches_zero_yaw_and_current_speed_and_preserves_gradients():
    sensory, explicit = make_model(), make_model()
    explicit.load_state_dict(sensory.state_dict())
    scene = torch.zeros(1, 576)
    args = dict(racing_goals=torch.tensor([[0., 1.]]), scene_features=scene,
                scene_valid=torch.ones(1, 4, dtype=torch.bool))
    for now in (1., 1.04, 1.08):
        ctx = context(now, speed=3.)
        output = sensory.sensory_training_decision(decision_time=now, body_context=ctx, **args)
        reference = explicit.training_decision(decision_time=now, body_context=ctx,
                                                command=torch.tensor([[0., 3.]]), **args)
        assert all(torch.equal(a, b) for a, b in zip(output, reference))
    output[1].sum().backward()
    assert sensory.body_feedback.projection.weight.grad is not None
    assert sensory.state.grad_fn is not None


def test_sensory_path_rejects_missing_speed_and_nonoriginal_command_boundary():
    model = make_model()
    ctx = context(1.)
    ctx.decision_body.valid.setflags(write=True)
    ctx.decision_body.valid[16] = False
    ctx.decision_body.valid.setflags(write=False)
    args = dict(decision_time=1., racing_goals=torch.tensor([[0., 1.]]),
                scene_features=torch.zeros(1, 576), scene_valid=torch.ones(1, 4, dtype=torch.bool),
                body_context=ctx)
    with pytest.raises(ValueError, match='body speed'):
        model.sensory_decision(**args)
    assert model.state is None
    model.motor.command = nn.Identity()
    args['body_context'] = context(1.)
    with pytest.raises(ValueError, match='Original physical maneuver'):
        model.sensory_decision(**args)


def test_original_maneuver_currents_are_neutral_for_measured_speed():
    boundary = ManeuverCurrents([1, -1, 1, -1], 2)
    body = torch.zeros(1, 21)
    body[0, 16] = 3.
    turn, power = boundary(torch.tensor([[0., 3.]]), body, torch.ones(1, 21, dtype=torch.bool),
                           torch.tensor([[0., 1.]]), command_contract=MANEUVER_CONTRACT,
                           goal_contract=RACING_GOAL_CONTRACT)
    assert torch.count_nonzero(turn) == 0
    torch.testing.assert_close(power, torch.full((1, 2), .05))


def test_interval_yaw_neutral_is_same_graph_and_missing_explicit():
    from flybrain_drive.graph import sha256
    from flybrain_drive.interval_yaw_visual_boundary import IntervalYawVisualBoundary
    from flybrain_drive.view_yaw_interval import ViewYawInterval
    a,b = make_model(),make_model()
    b.load_state_dict(a.state_dict())
    digest=sha256('configs/ac-virtual-body-v1.json')
    b.interval_yaw_boundary=IntervalYawVisualBoundary(2,body_contract_sha256=digest)
    scene=torch.zeros(1,576)
    for i in range(4):
        now=1.+i*.04;ctx=context(now)
        start=now-.07 if i==0 else b.previous_render_lowers[0]
        interval=ViewYawInterval(start,now-.03,now-.02,now,.2,.2*(now-.03-start),
                                 1,2,now-.08,now-.025,digest)
        legacy=a.sensory_decision(decision_time=now,racing_goals=torch.tensor([[0.,1.]]),
            scene_features=scene,scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=ctx)
        actual=b.experimental_sensory_interval_decision(decision_time=now,racing_goals=torch.tensor([[0.,1.]]),
            scene_features=scene,scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=ctx,
            interval_yaw=(interval,None,None,None) if i else (None,None,None,None))
        assert torch.equal(legacy[0],actual[0]) and torch.equal(a.state,b.state)
    with pytest.raises(ValueError,match='Four explicit'):
        b.experimental_sensory_interval_decision(decision_time=1.16,racing_goals=torch.tensor([[0.,1.]]),
            scene_features=scene,scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=context(1.16),
            interval_yaw=None)


def test_interval_yaw_signal_causal_and_bad_provenance_rolls_back():
    from dataclasses import replace
    from flybrain_drive.graph import sha256
    from flybrain_drive.interval_yaw_visual_boundary import IntervalYawVisualBoundary
    from flybrain_drive.view_yaw_interval import ViewYawInterval
    a,b=make_model(),make_model();b.load_state_dict(a.state_dict())
    digest=sha256('configs/ac-virtual-body-v1.json')
    a.interval_yaw_boundary=IntervalYawVisualBoundary(2,body_contract_sha256=digest)
    b.interval_yaw_boundary=IntervalYawVisualBoundary(2,body_contract_sha256=digest)
    with torch.no_grad():b.interval_yaw_boundary.projection.weight.fill_(.5)
    for i in range(5):
        now=1.+i*.04;scene=torch.zeros(1,576)
        if i>=1:scene[0,32]=1.
        start=now-.07 if i==0 else b.previous_render_lowers[0]
        y=ViewYawInterval(start,now-.03,now-.02,now,.4,.4*(now-.03-start),1,2,now-.08,now-.025,digest)
        for model in (a,b):
            model.experimental_sensory_interval_decision(decision_time=now,racing_goals=torch.tensor([[0.,1.]]),
                scene_features=scene,scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=context(now),
                interval_yaw=(y,None,None,None) if i else (None,None,None,None))
    assert not torch.equal(a.state,b.state)
    before=b.state.clone();last=b.previous_time;now=1.2
    bad=ViewYawInterval(now-.07,now-.03,now-.02,now,.4,.4*.04,1,2,now-.08,now+.01,digest)
    with pytest.raises(ValueError,match='provenance'):
        b.experimental_sensory_interval_decision(decision_time=now,racing_goals=torch.tensor([[0.,1.]]),
            scene_features=scene,scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=context(now),
            interval_yaw=(bad,None,None,None))
    assert torch.equal(b.state,before) and b.previous_time==last
    wrong_start=b.previous_render_lowers[0]-.01
    bad_start=ViewYawInterval(wrong_start,now-.03,now-.02,now,.4,.4*(now-.03-wrong_start),
                              1,2,wrong_start-.01,now-.025,digest)
    with pytest.raises(ValueError,match='previous view render'):
        b.experimental_sensory_interval_decision(decision_time=now,racing_goals=torch.tensor([[0.,1.]]),
            scene_features=scene,scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=context(now),
            interval_yaw=(bad_start,None,None,None))
    assert torch.equal(b.state,before) and b.previous_time==last


def test_interval_training_gradients_follow_same_recurrent_graph_and_detach():
    from flybrain_drive.graph import sha256
    from flybrain_drive.interval_yaw_visual_boundary import IntervalYawVisualBoundary
    from flybrain_drive.view_yaw_interval import ViewYawInterval
    torch.manual_seed(21)
    model=make_model()
    digest=sha256('configs/ac-virtual-body-v1.json')
    model.interval_yaw_boundary=IntervalYawVisualBoundary(2,body_contract_sha256=digest)
    for i in range(6):
        now=1.+i*.04
        scene=torch.zeros(1,576)
        if i>=1: scene[0,32]=1.
        previous=model.previous_render_lowers
        yaw=(None,)*4
        if previous is not None:
            start=previous[0]
            yaw=(ViewYawInterval(start,now-.03,now-.02,now,.4,.4*(now-.03-start),
                                  1,2,start-.01,now-.025,digest),None,None,None)
        intent,muscles=model.experimental_sensory_interval_training_decision(
            decision_time=now,racing_goals=torch.tensor([[0.,1.]]),scene_features=scene,
            scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=context(now),interval_yaw=yaw)
    intent.sum().backward()
    grad=model.interval_yaw_boundary.projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)>0
    prior=model.state.detach().clone()
    model.detach_training_state()
    assert torch.equal(model.state,prior) and model.state.grad_fn is None
    with torch.no_grad(), pytest.raises(ValueError,match='autograd'):
        model.experimental_sensory_interval_training_decision(
            decision_time=1.24,racing_goals=torch.tensor([[0.,1.]]),scene_features=scene,
            scene_valid=torch.ones((1,4),dtype=torch.bool),body_context=context(1.24),interval_yaw=(None,)*4)


def test_recurrent_speed_boundary_preserves_neutral_and_trains_same_state():
    from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
    a, b = make_model(), make_model()
    b.load_state_dict(a.state_dict())
    b.speed_command_boundary = FlightStateSpeedCommand(2)
    args = dict(racing_goals=torch.tensor([[0., 1.]]), scene_features=torch.zeros(1, 576),
                scene_valid=torch.ones(1, 4, dtype=torch.bool))
    for now in (1., 1.04, 1.08):
        first = a.sensory_training_decision(decision_time=now, body_context=context(now, speed=3.), **args)
        second = b.experimental_recurrent_speed_training_decision(
            decision_time=now, body_context=context(now, speed=3.), **args)
        assert all(torch.equal(x, y) for x, y in zip(first, second))
        assert torch.equal(a.state, b.state)
    b.state[0, 7].backward()
    grad = b.speed_command_boundary.projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    saved = b.state.detach().clone()
    b.detach_training_state()
    assert torch.equal(saved, b.state) and b.state.grad_fn is None
    b.reset_episode()
    assert b.state is None


def test_recurrent_speed_boundary_uses_only_previous_state_and_obeys_ceiling():
    from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
    model = make_model()
    model.speed_command_boundary = FlightStateSpeedCommand(2, max_delta_m_s=10.)
    with torch.no_grad():
        model.speed_command_boundary.projection.bias.fill_(10.)
    args = dict(racing_goals=torch.tensor([[0., .1]]), scene_features=torch.zeros(1, 576),
                scene_valid=torch.ones(1, 4, dtype=torch.bool))
    command = model._recurrent_speed_command(args['scene_features'], context(1., speed=1.), args['racing_goals'])
    assert command[0, 1].item() == pytest.approx(10. / 3.6)
    model.experimental_recurrent_speed_decision(decision_time=1., body_context=context(1., speed=1.), **args)
    assert model.state is not None
    with torch.no_grad(), pytest.raises(ValueError, match='autograd'):
        model.experimental_recurrent_speed_training_decision(
            decision_time=1.04, body_context=context(1.04, speed=1.), **args)

def test_airflow_enters_same_state_only_with_explicit_valid_sensor():
    from flybrain_drive.flight_airflow_feedback_adapter import AirflowFeedbackAdapter
    from flybrain_drive.relative_airflow_sensor import missing_airflow,relative_airflow
    model=make_model();model.airflow_feedback=AirflowFeedbackAdapter(1)
    with torch.no_grad():model.airflow_feedback.projection.weight[0,0]=1.
    def wind(now,speed,verified=True):
        return relative_airflow(wind_world_m_s=np.zeros(3),vehicle_world_m_s=np.array([speed,0.,0.]),body_axes_world=np.eye(3),wind_receipt=now-.01,body_receipt=now-.01,decision=now,source_verified=verified)
    def step(now,air):
        return model.experimental_airflow_sensory_decision(decision_time=now,racing_goals=torch.tensor([[0.,1.]]),scene_features=torch.zeros(1,576),scene_valid=torch.ones(1,4,dtype=torch.bool),body_context=context(now),airflow=air)
    step(1.,wind(1.,10.));with_wind=model.previous_drive.clone()
    step(1.04,wind(1.04,10.));state=model.state.clone()
    model.reset_episode();step(1.,missing_airflow(1.));without_wind=model.previous_drive.clone()
    step(1.04,missing_airflow(1.04));assert not torch.equal(state,model.state)
    assert with_wind[0,2] != without_wind[0,2]
    model.reset_episode();step(1.,wind(1.,10.,False));assert torch.equal(model.previous_drive,without_wind)
    old=model.state.clone()
    with pytest.raises(ValueError,match='same-decision'):
        step(1.04,missing_airflow(1.))
    assert torch.equal(model.state,old) and model.previous_time==1.

def test_airflow_boundary_has_training_gradient_in_same_recurrent_state():
    from flybrain_drive.flight_airflow_feedback_adapter import AirflowFeedbackAdapter
    from flybrain_drive.relative_airflow_sensor import relative_airflow
    model=make_model();model.airflow_feedback=AirflowFeedbackAdapter(1)
    for now in (1.,1.04):
        air=relative_airflow(wind_world_m_s=np.zeros(3),vehicle_world_m_s=np.array([10.,0.,0.]),body_axes_world=np.eye(3),wind_receipt=now-.01,body_receipt=now-.01,decision=now,source_verified=True)
        _,muscles=model.experimental_airflow_sensory_training_decision(decision_time=now,racing_goals=torch.tensor([[0.,1.]]),scene_features=torch.zeros(1,576),scene_valid=torch.ones(1,4,dtype=torch.bool),body_context=context(now),airflow=air)
    muscles[0,0].backward();grad=model.airflow_feedback.projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0

def test_recurrent_speed_and_missing_airflow_share_exact_original_state():
    from flybrain_drive.flight_airflow_feedback_adapter import AirflowFeedbackAdapter
    from flybrain_drive.flight_state_speed_command import FlightStateSpeedCommand
    from flybrain_drive.relative_airflow_sensor import missing_airflow
    from flybrain_drive.unified_airflow_speed import recurrent_speed_airflow_decision
    original=make_model();original.speed_command_boundary=FlightStateSpeedCommand(2)
    combined=copy.deepcopy(original);combined.airflow_feedback=AirflowFeedbackAdapter(1)
    with torch.no_grad():combined.airflow_feedback.projection.weight.fill_(.5)
    args=dict(racing_goals=torch.tensor([[0.,1.]]),scene_features=torch.zeros(1,576),scene_valid=torch.ones(1,4,dtype=torch.bool))
    for now in (1.,1.04,1.08):
        ctx=context(now,speed=2.)
        expected=original.experimental_recurrent_speed_decision(decision_time=now,body_context=ctx,**args)
        actual=recurrent_speed_airflow_decision(combined,decision_time=now,body_context=ctx,airflow=missing_airflow(now),**args)
        assert all(torch.equal(a,b) for a,b in zip(actual,expected,strict=True))
        assert torch.equal(combined.state,original.state)


def test_explicit_synthetic_load_adapter_uses_same_flight_circuit():
    from flybrain_drive.synthetic_load_feedback import SyntheticLoadBodyFeedbackAdapter
    baseline = make_model()
    ports = len(baseline.motor.backbone.body_ports)
    adapter = SyntheticLoadBodyFeedbackAdapter(ports)
    model = make_model(body_feedback=adapter)
    assert model.body_feedback is adapter
    assert model.body_feedback.load_projection.weight.count_nonzero().item() == 0
