"""Research-only unified visual/body input to one physical flight graph state.

Port transfers are engineering hypotheses. This module has no simulator or
actuator interface and does not qualify a predictor or a physical timing source.
"""
import copy
import math

import torch
from torch import nn

from .bilateral_retinal_contrast_boundary import BilateralRetinalContrastBoundary
from .flight_airflow_feedback_adapter import AirflowFeedbackAdapter
from .flight_body_feedback_adapter import (
    CenteredSpeedBodyFeedbackAdapter, CenteredVehicleBodyFeedbackAdapter, VehicleBodyFeedbackAdapter,
)
from .flight_embodied_navigation import modulate_current
from .synthetic_load_feedback import SyntheticLoadBodyFeedbackAdapter
from .flight_maneuver import MANEUVER_CONTRACT, ManeuverCurrents, PhysicalManeuverCircuit
from .flight_motion_forecast import CausalMotionForecast
from .flight_opponent_temporal_boundary import OpponentTemporalMotionBoundary
from .flight_state_speed_command import FlightStateSpeedCommand, SymmetricFlightStateSpeedCommand
from .interval_yaw_visual_boundary import IntervalYawVisualBoundary
from .flight_visual_body_temporal_context import PerViewVisualBodyContextBoundary
from .flyvis_camera import integration_intervals
from .packed_view_body import pack_view_body
from .retinal_spatial_contrast_boundary import RetinalSpatialContrastBoundary
from .racing_goal import RACING_GOAL_CONTRACT
from .temporal_body_context import (HostFreshnessLimits, PerViewTemporalBodyContext,
    per_view_temporal_body_context, require_decision_host_freshness)


def require_identical_flight_graph(motor, visual_backbone):
    """Reject graph, dynamics, port or protected-body changes during transfer."""
    if not isinstance(motor, PhysicalManeuverCircuit):
        raise ValueError('Explicit physical maneuver flight graph required')
    source, target = visual_backbone, motor.backbone
    for key in ('vision_ports', 'body_ports', 'steering_ports', 'goal_ports'):
        if not torch.equal(getattr(source,key).detach().cpu(),getattr(target,key).detach().cpu()):
            raise ValueError('Flight graph ports differ: '+key)
    if source.neurons != target.neurons or source.spec['dynamics'] != target.spec['dynamics']:
        raise ValueError('Flight graph dimensions or dynamics differ')
    one = source.circuit.matrix.detach().to_sparse_coo().coalesce().cpu()
    two = target.circuit.matrix.detach().to_sparse_coo().coalesce().cpu()
    if (one.shape != two.shape or not torch.equal(one.indices(),two.indices())
            or not torch.equal(one.values(),two.values())):
        raise ValueError('Original chemical graph differs')
    for key in ('weight','bias'):
        if not torch.equal(getattr(source.body,key).detach().cpu(),getattr(target.body,key).detach().cpu()):
            raise ValueError('Protected flight body boundary differs')


class UnifiedFlightSensorimotor(nn.Module):
    """One recurrent flight state with visual, body, and physical command ports.

    The caller must provide actor-derived commands; the module cannot establish
    their provenance or turn teacher/map truth into an admissible actor input.
    """

    def __init__(self, motor, visual_backbone, temporal, view_context, *, motion_forecast=None,
                 body_feedback=None, spatial_contrast=None, freshness_limits=None,
                 interval_yaw_boundary=None, speed_command_boundary=None, airflow_feedback=None):
        super().__init__()
        require_identical_flight_graph(motor,visual_backbone)
        if (not isinstance(temporal,OpponentTemporalMotionBoundary)
                or not isinstance(view_context,PerViewVisualBodyContextBoundary)
                or temporal.projection.out_features != len(motor.backbone.vision_ports)
                or view_context.projection.out_features != len(motor.backbone.vision_ports)):
            raise ValueError('Exact compatible visual-temporal ports required')
        if freshness_limits is not None and type(freshness_limits) is not HostFreshnessLimits:
            raise ValueError('Explicit host freshness limits required')
        self.freshness_limits = freshness_limits
        self.motor = motor
        self.visual = copy.deepcopy(visual_backbone.vision)
        self.scene_pool = copy.deepcopy(visual_backbone.scene_pool)
        self.register_buffer('visual_scale',visual_backbone.visual_scale.detach().clone())
        self.temporal = temporal
        self.view_context = view_context
        if (interval_yaw_boundary is not None and
                (type(interval_yaw_boundary) is not IntervalYawVisualBoundary
                 or interval_yaw_boundary.projection.out_features != len(motor.backbone.vision_ports))):
            raise ValueError('Explicit compatible diagnostic interval-yaw boundary required')
        self.interval_yaw_boundary = interval_yaw_boundary
        if speed_command_boundary is not None and type(speed_command_boundary) not in (FlightStateSpeedCommand, SymmetricFlightStateSpeedCommand):
            raise ValueError('Explicit experimental flight-state speed boundary required')
        self.speed_command_boundary = speed_command_boundary
        if motion_forecast is not None and not isinstance(motion_forecast, CausalMotionForecast):
            raise ValueError('Explicit causal motion forecast required')
        self.motion_forecast = motion_forecast
        if (spatial_contrast is not None and
                (type(spatial_contrast) not in (RetinalSpatialContrastBoundary, BilateralRetinalContrastBoundary) or
                 spatial_contrast.projection.out_features != len(motor.backbone.vision_ports))):
            raise ValueError('Explicit compatible spatial contrast boundary required')
        self.spatial_contrast = spatial_contrast
        if body_feedback is None:
            body_feedback = VehicleBodyFeedbackAdapter(len(motor.backbone.body_ports))
        if (type(body_feedback) not in (VehicleBodyFeedbackAdapter, CenteredVehicleBodyFeedbackAdapter,
                                        CenteredSpeedBodyFeedbackAdapter, SyntheticLoadBodyFeedbackAdapter)
                or body_feedback.projection.out_features != len(motor.backbone.body_ports)):
            raise ValueError('Explicit compatible vehicle body feedback boundary required')
        self.body_feedback = body_feedback
        if airflow_feedback is not None and (type(airflow_feedback) is not AirflowFeedbackAdapter
                or airflow_feedback.projection.out_features != len(motor.backbone.body_ports)):
            raise ValueError('Compatible experimental airflow boundary required')
        self.airflow_feedback = airflow_feedback
        self.reset_episode()

    def reset_episode(self):
        # Optional stateful visual boundaries must not retain a previous episode's
        # stationary reference or source generation across an explicit restart.
        context = getattr(self, 'view_context', None)
        if context is not None and callable(getattr(context, 'reset_episode', None)):
            context.reset_episode()
        self.state = None
        self.traces = None
        self.previous_drive = None
        self.previous_scene = None
        self.previous_raw_scene = None
        self.previous_time = None
        self.previous_body_receipt = None
        self.previous_render_lowers = None

    @torch.no_grad()
    def decision(self, *, decision_time, command, racing_goals, scene_features,
                 scene_valid, body_context):
        return self._decision(decision_time=decision_time, command=command,
            racing_goals=racing_goals, scene_features=scene_features,
            scene_valid=scene_valid, body_context=body_context, retain_graph=False)

    def training_decision(self, *, decision_time, command, racing_goals, scene_features,
                          scene_valid, body_context):
        """Advance the same state with gradients; caller must truncate between windows."""
        if not torch.is_grad_enabled():
            raise ValueError('Training requires enabled autograd')
        return self._decision(decision_time=decision_time, command=command,
            racing_goals=racing_goals, scene_features=scene_features,
            scene_valid=scene_valid, body_context=body_context, retain_graph=True)

    def _neutral_sensory_command(self, scene_features, body_context):
        """Neutralize external yaw and speed error using the original current body.

        This is an experimental port configuration, not a learned speed policy or
        a guarantee that the resulting motor intent is neutral or safe.
        """
        if not isinstance(body_context, PerViewTemporalBodyContext):
            raise ValueError('Original per-view body context required')
        if type(getattr(self.motor, 'command', None)) is not ManeuverCurrents:
            raise ValueError('Original physical maneuver currents required for neutral speed error')
        current = body_context.decision_body
        if not current.valid[16] or not math.isfinite(current.values[16]) or current.values[16] < 0:
            raise ValueError('Valid nonnegative original body speed required')
        if not isinstance(scene_features, torch.Tensor):
            raise ValueError('Original visual tensor required')
        return scene_features.new_tensor([[0., float(current.values[16])]])

    def _recurrent_speed_command(self, scene_features, body_context, racing_goals):
        if self.speed_command_boundary is None:
            raise ValueError('Explicit experimental flight-state speed boundary required')
        neutral = self._neutral_sensory_command(scene_features, body_context)
        if self.state is None:
            muscles = scene_features.new_zeros((1, self.speed_command_boundary.muscle_count))
        else:
            _, muscles = self.motor.backbone._read_state_tensor(self.state)
        return self.speed_command_boundary(muscles, neutral[:, 1:2], racing_goals)

    @torch.no_grad()
    def experimental_recurrent_speed_decision(self, *, decision_time, racing_goals,
                                               scene_features, scene_valid, body_context):
        """Use only previous flight state and current body speed; diagnostic only."""
        command = self._recurrent_speed_command(scene_features, body_context, racing_goals)
        return self._decision(decision_time=decision_time, command=command,
            racing_goals=racing_goals, scene_features=scene_features,
            scene_valid=scene_valid, body_context=body_context, retain_graph=False)

    def experimental_recurrent_speed_training_decision(self, *, decision_time, racing_goals,
                                                        scene_features, scene_valid, body_context):
        """Train this same recurrent state; caller enforces dataset admissibility."""
        if not torch.is_grad_enabled():
            raise ValueError('Training requires enabled autograd')
        command = self._recurrent_speed_command(scene_features, body_context, racing_goals)
        return self._decision(decision_time=decision_time, command=command,
            racing_goals=racing_goals, scene_features=scene_features,
            scene_valid=scene_valid, body_context=body_context, retain_graph=True)

    @torch.no_grad()
    def sensory_decision(self, *, decision_time, racing_goals, scene_features,
                         scene_valid, body_context):
        """Experimental graph-only turn/power intent with no external maneuver target."""
        command = self._neutral_sensory_command(scene_features, body_context)
        return self._decision(decision_time=decision_time, command=command,
            racing_goals=racing_goals, scene_features=scene_features,
            scene_valid=scene_valid, body_context=body_context, retain_graph=False)

    @torch.no_grad()
    def experimental_sensory_interval_decision(self, *, decision_time, racing_goals,
                                               scene_features, scene_valid, body_context,
                                               interval_yaw):
        """Diagnostic-only interval input into the same recurrent flight graph."""
        if self.interval_yaw_boundary is None:
            raise ValueError('Explicit interval-yaw boundary required')
        if not isinstance(interval_yaw, tuple) or len(interval_yaw) != 4:
            raise ValueError('Four explicit interval entries required')
        command = self._neutral_sensory_command(scene_features, body_context)
        return self._decision(decision_time=decision_time, command=command,
            racing_goals=racing_goals, scene_features=scene_features,
            scene_valid=scene_valid, body_context=body_context, retain_graph=False,
            interval_yaw=interval_yaw)

    def sensory_training_decision(self, *, decision_time, racing_goals, scene_features,
                                  scene_valid, body_context):
        """Train the same sensory path while retaining the existing recurrent state."""
        if not torch.is_grad_enabled():
            raise ValueError('Training requires enabled autograd')
        command = self._neutral_sensory_command(scene_features, body_context)
        return self._decision(decision_time=decision_time, command=command,
            racing_goals=racing_goals, scene_features=scene_features,
            scene_valid=scene_valid, body_context=body_context, retain_graph=True)

    def experimental_sensory_interval_training_decision(self, *, decision_time, racing_goals,
                                                        scene_features, scene_valid, body_context,
                                                        interval_yaw):
        """Train the diagnostic interval port in the same recurrent state.

        The caller must independently enforce dataset admissibility. This method
        does not qualify host timestamps or authorize an actor deployment.
        """
        if not torch.is_grad_enabled():
            raise ValueError('Training requires enabled autograd')
        if self.interval_yaw_boundary is None:
            raise ValueError('Explicit interval-yaw boundary required')
        if not isinstance(interval_yaw, tuple) or len(interval_yaw) != 4:
            raise ValueError('Four explicit interval entries required')
        command = self._neutral_sensory_command(scene_features, body_context)
        return self._decision(decision_time=decision_time, command=command,
            racing_goals=racing_goals, scene_features=scene_features,
            scene_valid=scene_valid, body_context=body_context, retain_graph=True,
            interval_yaw=interval_yaw)

    @torch.no_grad()
    def experimental_airflow_sensory_decision(self, *, decision_time, racing_goals,
                                               scene_features, scene_valid, body_context, airflow):
        """Apply explicitly sourced airflow to the same state; diagnostic only."""
        if self.airflow_feedback is None:raise ValueError('Explicit airflow boundary required')
        command=self._neutral_sensory_command(scene_features,body_context)
        return self._decision(decision_time=decision_time,command=command,racing_goals=racing_goals,
            scene_features=scene_features,scene_valid=scene_valid,body_context=body_context,
            retain_graph=False,airflow=airflow)

    def experimental_airflow_sensory_training_decision(self, *, decision_time, racing_goals,
                                                        scene_features, scene_valid, body_context, airflow):
        if not torch.is_grad_enabled() or self.airflow_feedback is None:
            raise ValueError('Enabled gradients and explicit airflow boundary required')
        command=self._neutral_sensory_command(scene_features,body_context)
        return self._decision(decision_time=decision_time,command=command,racing_goals=racing_goals,
            scene_features=scene_features,scene_valid=scene_valid,body_context=body_context,
            retain_graph=True,airflow=airflow)

    def detach_training_state(self):
        """Truncate gradients without resetting the numerical recurrent state."""
        for name in ('state', 'previous_drive', 'previous_scene', 'previous_raw_scene'):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.detach().clone())
        if self.traces is not None:
            self.traces = tuple(value.detach().clone() for value in self.traces)

    def _decision(self, *, decision_time, command, racing_goals, scene_features,
                  scene_valid, body_context, retain_graph, interval_yaw=None, airflow=None):
        if (not isinstance(body_context,PerViewTemporalBodyContext)
                or body_context.decision_time != decision_time):
            raise ValueError('Original per-view context at decision time required')
        per_view_temporal_body_context(body_context.visual_contexts,body_context.decision_body,
            render_start_lowers=body_context.render_start_lowers,
            camera_receipt_time=body_context.camera_receipt_time,decision_time=decision_time)
        interval_tensors = None
        if interval_yaw is not None:
            if self.interval_yaw_boundary is None:
                raise ValueError('Explicit interval-yaw boundary required')
            interval_tensors = self.interval_yaw_boundary.pack(
                interval_yaw, body_context=body_context, decision_time=decision_time)
            for index, item in enumerate(interval_yaw):
                if item is not None and (self.previous_render_lowers is None or
                        item.start_render_lower != self.previous_render_lowers[index]):
                    raise ValueError('Interval must start at the previous view render bound')
        if self.freshness_limits is not None:
            if self.previous_time is not None and self.previous_body_receipt is None:
                raise ValueError('Held body receipt missing after priming')
            require_decision_host_freshness(body_context, self.freshness_limits,
                                            held_body_receipt=self.previous_body_receipt)
        if (not isinstance(scene_features,torch.Tensor) or scene_features.shape != (1,576)
                or not isinstance(scene_valid,torch.Tensor) or scene_valid.shape != (1,4)
                or scene_valid.dtype != torch.bool or not scene_valid.all()
                or not torch.isfinite(scene_features).all()):
            raise ValueError('All four finite original visual views required')
        if self.airflow_feedback is not None and airflow is None:
            raise ValueError('Enabled airflow requires explicit observation or missing mask')
        if self.airflow_feedback is None and airflow is not None:
            raise ValueError('Airflow boundary is not enabled')
        current = body_context.decision_body
        body = torch.as_tensor(current.values.copy(),dtype=scene_features.dtype,
                               device=scene_features.device).unsqueeze(0)
        valid = torch.as_tensor(current.valid.copy(),dtype=torch.bool,
                                device=scene_features.device).unsqueeze(0)
        history, history_valid = pack_view_body(body_context.visual_contexts,
            device=scene_features.device,dtype=scene_features.dtype)
        elapsed = None if self.previous_time is None else decision_time-self.previous_time
        if elapsed is not None and (not math.isfinite(elapsed) or not 0 < elapsed <= .08):
            raise ValueError('Original advancing decision time within 80ms required')
        raw_scene = scene_features
        if self.motion_forecast is not None:
            scene_features = self.motion_forecast(raw_scene, self.previous_raw_scene, elapsed)
        drive = self.motor.drive(command,body,valid,racing_goals,
            command_contract=MANEUVER_CONTRACT,goal_contract=RACING_GOAL_CONTRACT)
        state, traces = self.state, self.traces
        if state is None:
            if elapsed is not None:
                raise ValueError('Explicit fresh episode required')
            state = self.motor.initial_state(1)
            delta,traces = self.temporal.step(scene_features,scene_valid,None,None)
        else:
            intervals = integration_intervals(elapsed)
            for index,dt in enumerate(intervals):
                state = self.motor.backbone._advance_elapsed_tensor(self.previous_drive,state,dt)
                observed = scene_features if index == len(intervals)-1 else self.previous_scene
                delta,traces = self.temporal.step(observed,scene_valid,traces,dt)
        ports = self.motor.backbone.vision_ports
        base = self.visual(torch.cat([scene_features.new_zeros((1,12)),scene_features],dim=1)).sigmoid()
        drive = drive.index_copy(1,ports,base)
        drive = self.temporal.modulate(drive,ports,delta)
        if self.spatial_contrast is not None:
            drive = self.temporal.modulate(
                drive,ports,self.spatial_contrast(raw_scene,scene_valid))
        drive = self.temporal.modulate(drive,ports,self.view_context(traces,history,history_valid))
        if interval_tensors is not None:
            drive = self.temporal.modulate(drive,ports,
                self.interval_yaw_boundary(traces,*interval_tensors))
        body_ports = self.motor.backbone.body_ports
        drive = drive.index_copy(1,body_ports,
            modulate_current(drive[:,body_ports],self.body_feedback(body,valid)))
        if self.airflow_feedback is not None:
            delta_air=self.airflow_feedback(airflow,decision_time=decision_time,like=scene_features[:,:1])
            drive=drive.index_copy(1,body_ports,modulate_current(drive[:,body_ports],delta_air))
        intent,muscles = self.motor.backbone._read_state_tensor(state)
        keep = (lambda value: value.clone()) if retain_graph else (lambda value: value.detach().clone())
        self.state = keep(state)
        self.traces = tuple(keep(x) for x in traces)
        self.previous_drive = keep(drive)
        self.previous_scene = keep(scene_features)
        self.previous_raw_scene = keep(raw_scene)
        self.previous_render_lowers = tuple(body_context.render_start_lowers)
        self.previous_time = float(decision_time)
        self.previous_body_receipt = float(current.original_received_monotonic_seconds)
        return keep(intent),keep(muscles)
