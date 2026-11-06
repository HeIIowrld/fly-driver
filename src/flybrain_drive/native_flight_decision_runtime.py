"""Read-only execution boundary for the learned visual/native flight circuit.

Engineering adapters remain distinct from measured flight topology. No teacher,
world position, route, actuators or simulator lifecycle APIs enter this boundary.
"""
import time
import numpy as np
import torch
from .native_multisensory_extension import NativeExtensionSample
from .racing_goal import racing_goal
from .temporal_body_context import PerViewTemporalBodyContext, per_view_temporal_body_context


def prepare_fixed_cpu_backend(model):
    """Build constant sparse caches before admitting a sensor decision.

    The returned tensor is discarded. No model recurrence, adapter, clock,
    gradient or learned weight is advanced; this is backend initialization.
    """
    circuit = model.motor.backbone.circuit
    if circuit.sparse_backend != 'cached_csr_cpu' or circuit.matrix.device.type != 'cpu':
        raise ValueError('Declared fixed CPU inference backend required')
    durations = []
    with torch.no_grad():
        zero = torch.zeros(1,circuit.size,dtype=torch.float32,device='cpu')
        for _ in range(3):
            started = time.perf_counter()
            circuit._advance_tensor(zero,zero)
            durations.append(time.perf_counter()-started)
    return dict(backend='cached_csr_cpu',constant_zero_steps=3,
        initialization_seconds=durations,model_recurrent_steps=0,optimizer_updates=0)


class NativeFlightDecisionRuntime:
    def __init__(self, model, head, descending, optical, body_port, encoder, *, monotonic=time.perf_counter):
        self.model, self.head = model, head
        self.descending, self.optical, self.body_port = descending, optical, body_port
        self.encoder, self.monotonic = encoder, monotonic
        self.device = torch.device('cpu')
        for module in [model, head, descending, optical, body_port]:
            if any(t.device.type != 'cpu' for t in [*module.parameters(),*module.buffers()]):
                raise ValueError('Explicit fixed CPU flight runtime components required')
            module.eval().requires_grad_(False)
        self.episode = None; self.poisoned = False; self.active = None; self.closed = False
        self.handles = [model.motor.command.register_forward_hook(self._descending),
                        model.visual.register_forward_hook(self._optical),
                        model.body_feedback.register_forward_hook(self._body)]

    def _descending(self, module, args, value):
        steer, power = value
        return steer+self.descending(self.active['static']), power

    def _optical(self, module, args, value):
        return value+self.optical(self.active['static'][:,64:])

    def _body(self, module, args, value):
        if not torch.equal(args[0], self.active['body']):
            raise ValueError('Native sensation joined to another current body')
        delta = self.body_port(self.active['native'], self.active['valid'])
        self.active['maximum_body_delta'] = float(delta.abs().max())
        return value+delta

    def begin_episode(self, episode_id):
        if self.closed or not isinstance(episode_id,str) or not episode_id:
            raise ValueError('Open runtime and explicit episode identity required')
        self.model.reset_episode()
        self.encoder.episode = None; self.encoder.last = None; self.encoder.last_decision = None
        self.episode = episode_id; self.poisoned = False; self.active = None
        self.last_time = None; self.last_rejection = None

    def decide(self, *, decision_time, scene_features, static_features, body_context, native_sample):
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
                _,muscle = self.model.sensory_decision(decision_time=decision_time,
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

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles=[]; self.active=None; self.closed=True
