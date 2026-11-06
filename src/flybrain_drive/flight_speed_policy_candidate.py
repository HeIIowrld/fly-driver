"""Strict research speed-policy overlay and original freshness guards, read-only."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from .graph import sha256
from .sac_unified_student import read
from .spatial_motion_dn_candidate import apply_spatial_motion_dn,SpatialMotionDNDecisionRuntime
from .flight_state_speed_command import FlightStateSpeedCommand
from .native_multisensory_extension import NativeExtensionSample
from .temporal_body_context import PerViewTemporalBodyContext,per_view_temporal_body_context
from .racing_goal import racing_goal
from .motion_spatial_runtime import validate_motion_binding


def apply_flight_speed_policy(checkpoint,components,calibration,motion):
 path=Path(checkpoint);r=read(path.parent/'result.json');m=read(path.parent/'manifest.json')['config'];p=m['plan']
 if r['status']!='completed_flight_speed_policy_fit' or sha256(path)!=r['checkpoint_sha256'] or any(sha256(p[k])!=p[k+'_sha256'] for k in ['speed_parent_checkpoint','speed_parent_optimizer','motion_parent_checkpoint','fixed_decoder']):raise ValueError('Completed hash-bound speed-policy inputs required')
 cp=torch.load(path,map_location='cpu',weights_only=True);keys={'schema','arm','current_calibration','body_port','fixed_decoder','fixed_decoder_sha256','parent_checkpoint_sha256','parent_port_checkpoint_sha256','motion_parent_checkpoint_sha256','speed_parent_checkpoint_sha256','speed_parent_optimizer_sha256','speed_policy','speed_max_delta_m_s','initial_gate','minimum_gate','actor_qualified'}
 if set(cp)!=keys or cp['schema']!='flybrain.flight_speed_policy_learning.experimental.v1' or cp['arm'] not in ['neutral_control','speed_policy'] or cp['arm']!=m['arm'] or cp['actor_qualified'] is not False or any(cp[k]!=p[k] for k in keys-{'schema','arm','current_calibration','body_port','speed_policy','actor_qualified'}):raise ValueError('Exact unqualified speed-policy schema required')
 if components[0].speed_command_boundary is not None:raise ValueError('Fresh original flight model without speed child required')
 private=(*components[:4],deepcopy(components[4]),*components[5:]);changed,current,receipt=apply_spatial_motion_dn(p['speed_parent_checkpoint'],private,deepcopy(calibration),motion);body=changed[4];policy=FlightStateSpeedCommand(6,max_delta_m_s=p['speed_max_delta_m_s'])
 for module,key,allowed in [(current,'current_calibration',{'motion_projection.weight'}),(body,'body_port',{'projection.weight'}),(policy,'speed_policy',{'projection.weight','projection.bias'})]:
  expected=module.state_dict();state=cp[key]
  if set(state)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() or k not in allowed and not torch.equal(v,expected[k]) for k,v in state.items()):raise ValueError('Exact frozen sensory normalization and finite speed policy required')
 old=torch.load(p['motion_parent_checkpoint'],map_location='cpu',weights_only=True)['body_port']['projection.weight'];w=cp['current_calibration']['motion_projection.weight'];delta=cp['body_port']['projection.weight']-old;sw=cp['speed_policy']['projection.weight'];sb=cp['speed_policy']['projection.bias']
 if w.norm()>p['visual_total_norm']+1e-7 or w.abs().max()>2+1e-7 or delta.norm()>p['native_total_norm']+1e-7 or delta.abs().max()>p['native_entry_delta']+1e-7 or sw.norm()>p['speed_weight_total_norm']+1e-7 or sw.abs().max()>p['speed_weight_absolute_bound']+1e-7 or sb.norm()>p['speed_bias_total_norm']+1e-7 or sb.abs().max()>p['speed_bias_absolute_bound']+1e-7 or sw[0,-1]>0 or cp['arm']=='neutral_control' and (torch.count_nonzero(sw)>0 or torch.count_nonzero(sb)>0):raise ValueError('Cumulative input bounds and nonpositive direct speed gain required')
 current.load_state_dict(cp['current_calibration']);body.load_state_dict(cp['body_port']);policy.load_state_dict(cp['speed_policy']);current.eval().requires_grad_(False);body.eval().requires_grad_(False);policy.eval().requires_grad_(False);changed[0].speed_command_boundary=policy
 return changed,current,dict(receipt,speed_policy_checkpoint_sha256=sha256(path),arm=cp['arm'],actor_qualified=False,vehicle_control_sent=False)


class FlightSpeedPolicyDecisionRuntime(SpatialMotionDNDecisionRuntime):
    def __init__(self,*components,policy_limits,**kwargs):
        policy=components[0].speed_command_boundary
        if type(policy) is not FlightStateSpeedCommand or policy.muscle_count!=6:raise ValueError('Explicit existing six-muscle speed command required')
        self.speed_policy=policy;self.policy_limits=dict(policy_limits);self._check_speed_policy();super().__init__(*components,**kwargs)
    def _check_speed_policy(self):
        p=self.policy_limits;module=self.speed_policy;w=module.projection.weight;b=module.projection.bias
        if module.max_delta_m_s!=p['speed_max_delta_m_s'] or any(t.device.type!='cpu' or t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in [w,b]) or w.detach().norm()>p['speed_weight_total_norm']+1e-7 or b.detach().norm()>p['speed_bias_total_norm']+1e-7 or w.detach().abs().max()>p['speed_weight_absolute_bound']+1e-7 or b.detach().abs().max()>p['speed_bias_absolute_bound']+1e-7 or w.detach()[0,-1]>0:raise ValueError('Exact finite bounded CPU speed command required')

    def decide(self,*,motion_observation,**kwargs):
        if self.closed or self.episode is None or self.poisoned:raise RuntimeError('Explicit episode restart required')
        started=self.monotonic()
        try:
            self._check_speed_policy()
            if self.model.speed_command_boundary is not self.speed_policy:raise ValueError('Speed command module changed')
            validate_motion_binding(motion_observation,episode_id=self.episode,scene_features=kwargs['scene_features'],body_context=kwargs['body_context'])
            self.motion_active=torch.tensor(motion_observation.features.copy()[None],dtype=torch.float32,device='cpu')
            result=self._decide_with_speed(**kwargs)
            elapsed=self.monotonic()-started
            age=kwargs['decision_time']-kwargs['body_context'].decision_body.original_received_monotonic_seconds
            self.last_computation=dict(input_body_age_seconds=age,inference_seconds=elapsed,output_body_age_seconds=age+elapsed,includes_motion_binding=True)
            if elapsed<0 or age+elapsed>.05:raise ValueError('Motion binding/inference output exceeds original body50ms age')
            result.update(inference_seconds=elapsed,estimated_output_body_age_seconds=age+elapsed)
            return result
        except Exception as error:
            self.last_rejection=dict(error=type(error).__name__,reason=str(error),computation=getattr(self,'last_computation',None))
            self.poisoned=True
            raise
        finally:self.motion_active=None

    def _decide_with_speed(self, *, decision_time, scene_features, static_features, body_context, native_sample):
        if self.closed or self.episode is None or self.poisoned:
            raise RuntimeError('Explicit episode start required before inference or after rejection')
        started = self.monotonic()
        self.last_computation = None
        try:
            if not isinstance(body_context,PerViewTemporalBodyContext) or not isinstance(native_sample,NativeExtensionSample):
                raise ValueError('Original four-view history and current native sample required')
            current = body_context.decision_body
            if (current.episode_id != self.episode or current.decision_monotonic_seconds != decision_time
                    or body_context.decision_time != decision_time
                    or self.last_time is not None and decision_time <= self.last_time):
                raise ValueError('Episode or strictly increasing decision clock differs')
            per_view_temporal_body_context(body_context.visual_contexts,current,
                render_start_lowers=body_context.render_start_lowers,
                camera_receipt_time=body_context.camera_receipt_time,decision_time=decision_time)
            if any(v is not None and v.episode_id != self.episode for v in body_context.visual_contexts):
                raise ValueError('Historical body belongs to another episode')
            age = decision_time-current.original_received_monotonic_seconds
            camera_age = decision_time-body_context.camera_receipt_time
            if not 0 <= age <= .05 or not 0 <= camera_age <= .15:
                raise ValueError('Current body50ms/camera150ms freshness required')
            scene = np.asarray(scene_features); static = np.asarray(static_features)
            if (scene.shape != (576,) or static.shape != (320,)
                    or not np.isfinite(scene).all() or not np.isfinite(static).all()):
                raise ValueError('Finite four-view576 and same-camera static320 required')
            # The first64 are an exact deterministic selection of the scene
            # features; an arbitrary unrelated spatial sidecar is inadmissible.
            if not np.array_equal(static[:64],scene.reshape(4,9,16)[:,0].reshape(64)):
                raise ValueError('Scene and static features are from different camera observations')
            obs = self.encoder.encode(native_sample,episode_id=self.episode,decision_monotonic_seconds=decision_time)
            if (obs.packet_id != current.packet_id
                    or obs.original_received_monotonic_seconds != current.original_received_monotonic_seconds):
                raise ValueError('Native19 and legacy21 must share packet and first receipt')
            self.active = dict(static=torch.tensor(static[None],dtype=torch.float32,device=self.device),
                body=torch.tensor(current.values[None],dtype=torch.float32,device=self.device),
                native=torch.tensor(obs.values.copy()[None],dtype=torch.float32,device=self.device),
                valid=torch.tensor(obs.valid.copy()[None],dtype=torch.bool,device=self.device),maximum_body_delta=0.)
            with torch.no_grad():
                _,muscle = self.model.experimental_recurrent_speed_decision(decision_time=decision_time,
                    racing_goals=torch.tensor(racing_goal(300,1),device=self.device),
                    scene_features=torch.tensor(scene[None],dtype=torch.float32,device=self.device),
                    scene_valid=torch.ones(1,4,dtype=torch.bool,device=self.device),body_context=body_context)
                prediction = self.head(muscle)[0].numpy().copy()
            elapsed = self.monotonic()-started
            self.last_computation = dict(input_body_age_seconds=age,inference_seconds=elapsed,output_body_age_seconds=age+elapsed)
            if elapsed < 0 or age+elapsed > .05:
                raise ValueError('Inference output exceeds original body50ms age')
            if (not np.isfinite(prediction).all() or abs(prediction[0])>1
                    or (prediction[1:]<0).any() or (prediction[1:]>1).any()):
                raise ValueError('Finite bounded three-axis muscle readout required')
            result = dict(prediction=prediction,muscles=muscle[0].numpy().copy(),
                body_packet=current.packet_id,body_first_receipt=current.original_received_monotonic_seconds,
                inference_seconds=elapsed,estimated_output_body_age_seconds=age+elapsed,
                maximum_body_delta=self.active['maximum_body_delta'],
                acceleration_out_of_assessed_scope=obs.acceleration_out_of_assessed_scope,
                thermal_out_of_training_support=obs.thermal_out_of_training_support,
                actor_qualified=False,vehicle_control_sent=False)
            self.last_time=decision_time
            return result
        except Exception as error:
            self.last_rejection = dict(error=type(error).__name__,reason=str(error),computation=getattr(self,'last_computation',None))
            # A rejection may occur after a recurrence step. Never continue from
            # potentially advanced state or silently rewind/retry that decision.
            self.poisoned = True
            raise
        finally:
            self.active = None
