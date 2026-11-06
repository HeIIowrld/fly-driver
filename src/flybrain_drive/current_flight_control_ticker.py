"""Source-bound mailbox extension; the unchanged ticker alone advances recurrence."""
import base64,copy,hashlib,threading,time
from .ac_native_physics import PHYSICS_PREFIX
from .native_multisensory_extension import decode_native_extension
from .per_view_flyvis_stream import VisualFlightObservation
from .motion_spatial_runtime import MotionSpatialDecisionRuntime,MotionSpatialObservation
from .motion_per_view_flyvis_stream import MotionVisualFlightObservation
from .temporal_body_context import PerViewTemporalBodyContext,per_view_temporal_body_context
from .unified_live_ticker import defer_near_expiry
from .virtual_navigation_body import NavigationBodyEncoder,decode_navigation_body_prefix
import copy,math
from .native_flight_body_ticker import NativeFlightBodyTicker
from .current_flight_control_contract import FlightDecisionEnvelope, canonical3


class CurrentFlightControlTicker(NativeFlightBodyTicker):
    def __init__(self, runtime, *, run_id, generation, checkpoint_sha256, **kwargs):
        super().__init__(runtime, **kwargs)
        self.run_id, self.generation, self.checkpoint_sha256 = run_id, generation, checkpoint_sha256
        self.latest_decision = None
        self.decision_sequence = 0
        self.source = None
        self.pending_visual = None
        self.decision_mailbox = None

    def announce_next_visual(self,*,capture_id,bounds,camera_received,context_id):
        """Reserve a bounded handover to already qualified incoming real pixels."""
        now=self.clock()
        with self.condition:
            if (type(capture_id) is not int or capture_id<0 or len(bounds)!=4
                    or not all(type(t) in (float,int) and math.isfinite(t) and t>=0 for t in (*bounds,camera_received,now))
                    or camera_received>now or any(t>camera_received or now-t>.15 for t in bounds)
                    or self.latest_visual is not None and capture_id<=self.latest_visual[0].capture_id
                    or self.pending_visual is not None and capture_id<=self.pending_visual['capture_id']):
                raise ValueError('New qualified original camera announcement required')
            self.pending_visual=dict(capture_id=capture_id,bounds=tuple(bounds),camera_received=camera_received,
                                     context_id=context_id,announced_at=now)

    def publish_source(self, visual, context, *, uppers, first_copy_starts):
        if visual.generation != self.generation:
            raise ValueError("Original generation required")
        if len(uppers) != 4 or len(first_copy_starts) != 4:
            raise ValueError("Four view source witnesses required")
        for lo, hi, body in zip(visual.render_start_lowers, uppers, first_copy_starts, strict=True):
            if not body <= lo <= hi <= visual.camera_received or hi-body > .025:
                raise ValueError("Original source render bound differs")
        with self.condition:
            when = super().publish(visual, context)
            if self.pending_visual is not None and self.pending_visual['capture_id']==visual.capture_id:self.pending_visual=None
            self.source = dict(capture=visual.capture_id, version=self.version,
                               uppers=tuple(uppers), starts=tuple(first_copy_starts))
            return when

    def publish(self, *args, **kwargs):
        raise RuntimeError("Use source-witness publication")

    def step_once(self):
        if self.source_error is not None and self.source_error():raise RuntimeError('Native source failed; buffered rows are inadmissible')
        end=len(self.rows)
        while self.position<end:
            row=self.rows[self.position];authority=row['authority'];receipt=row['copy']['second_copy_end']
            raw=base64.b64decode(authority['prefix_base64'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=authority['prefix_sha256'] or receipt!=authority['received_monotonic_seconds']:
                raise ValueError('Original native bytes and receipt required')
            sample=decode_navigation_body_prefix(raw[:PHYSICS_PREFIX.size],receipt)
            extension=decode_native_extension(raw[:PHYSICS_PREFIX.size],receipt)
            if sample.legacy.packet_id!=authority['packet_id']:raise ValueError('Native packet identity differs')
            if self.latest is None:self.first=sample.legacy.packet_id;self.latest=sample;self.extension=extension
            elif sample.legacy.packet_id<self.latest.legacy.packet_id:raise ValueError('Native packet reset requires new episode')
            elif sample.legacy.packet_id==self.latest.legacy.packet_id:
                if sample.legacy.prefix_sha256!=self.latest.legacy.prefix_sha256:raise ValueError('Repeated packet changed bytes')
            else:self.latest=sample;self.extension=extension
            self.position+=1
        with self.condition:
            selected=self.latest_visual;version=self.version;selected_source=copy.deepcopy(self.source);pending=copy.deepcopy(self.pending_visual);consumed_version=self.consumed
        if selected is None or self.latest is None or self.latest.legacy.packet_id==self.first:return None
        now=self.clock()
        if now-self.latest.legacy.received_monotonic_seconds>.05:raise ValueError('Current native body stale at ticker')
        if self.previous_packet==self.latest.legacy.packet_id or self.last_update is not None and now-self.last_update<self.period:return None
        visual,historical,publication=selected
        if max(now-t for t in visual.render_start_lowers)>=.125:
            self.wait_events.append(dict(when=now,capture_id=visual.capture_id,publication=publication,view_age=max(now-t for t in visual.render_start_lowers),current_body_age=now-self.latest.legacy.received_monotonic_seconds))
        near_expiry=defer_near_expiry(now,visual.render_start_lowers,reserve=.025)
        # Do not spend the last view reserve on an obsolete scene when the
        # next, source-qualified camera group is already being computed.
        # Reserve two25ms inference slots before the original view deadline.
        # Actual pending wait stays bounded25ms; body50/view150 and
        # recurrent held-body checks still fail closed. No clock or state reset.
        if (pending is not None and version==consumed_version and pending['capture_id']>visual.capture_id
                and 0<=now-pending['announced_at']<=.025
                and self.runtime.model.previous_body_receipt is not None
                and 0<=now-self.runtime.model.previous_body_receipt<=.025
                and max(now-t for t in visual.render_start_lowers)>=.15-.050-self.period):
            self.wait_events.append(dict(when=now,capture_id=visual.capture_id,publication=publication,
                view_age=max(now-t for t in visual.render_start_lowers),current_body_age=now-self.latest.legacy.received_monotonic_seconds,
                reason='pending_new_scene_handover',pending=copy.deepcopy(pending)))
            return None
        if near_expiry:return None
        current=self.encoder.encode(self.latest,episode_id=self.episode,decision_monotonic_seconds=now)
        ctx=per_view_temporal_body_context(historical.visual_contexts,current,render_start_lowers=visual.render_start_lowers,camera_receipt_time=visual.camera_received,decision_time=now)
        held=self.runtime.model.previous_body_receipt
        event=dict(status='rejected',decision_time=now,body_packet=current.packet_id,body_first_receipt=current.original_received_monotonic_seconds,
            held_body_receipt=held,capture_id=visual.capture_id,render_start_lowers=list(visual.render_start_lowers),camera_received=visual.camera_received,publication=publication)
        computation_started=self.clock();thread_started=time.thread_time();process_started=time.process_time()
        def computation_receipt():
            return dict(started=computation_started,finished=self.clock(),thread_cpu_seconds=time.thread_time()-thread_started,process_cpu_seconds=time.process_time()-process_started)
        try:
            arguments=dict(decision_time=now,scene_features=visual.scene,static_features=visual.static,body_context=ctx,native_sample=self.extension)
            if isinstance(self.runtime,MotionSpatialDecisionRuntime):
                if not isinstance(visual,MotionVisualFlightObservation):raise ValueError('Motion-enabled runtime requires same-camera motion publication')
                arguments['motion_observation']=MotionSpatialObservation(self.episode,visual.camera_received,visual.render_start_lowers,visual.coarse_motion_reference,visual.motion_features)
                event['motion_features_sha256']=visual.motion_features_sha256
            decision=self.runtime.decide(**arguments)
            finished=self.clock();body_age=finished-current.original_received_monotonic_seconds;view_age=max(finished-t for t in visual.render_start_lowers)
            event.update(candidate_output_host_time=finished,candidate_body_age_seconds=body_age,candidate_view_age_seconds=view_age)
            if body_age>.05 or view_age>.15:raise ValueError('Native ticker output exceeds body50ms/view150ms')
            event.update(status='shadow_output',prediction=decision['prediction'].tolist(),output_host_time=finished,
                output_body_age_seconds=body_age,oldest_view_age_seconds=view_age,inference_seconds=decision['inference_seconds'],
                thermal_out_of_training_support=decision['thermal_out_of_training_support'],maximum_native_body_delta=decision['maximum_body_delta'])
            if (selected_source is None or selected_source['capture']!=event['capture_id']
                    or selected_source['version']!=version):raise ValueError('Pinned decision publication differs')
            self.decision_sequence+=1
            envelope=FlightDecisionEnvelope(self.run_id,self.episode,self.generation,self.checkpoint_sha256,
                self.decision_sequence,version,event['capture_id'],event['body_packet'],event['body_first_receipt'],
                tuple(event['render_start_lowers']),selected_source['uppers'],selected_source['starts'],
                event['camera_received'],event['publication'],event['decision_time'],event['output_host_time'],
                canonical3(event['prediction']))
            envelope.validate(now=self.clock(),run_id=self.run_id,episode_id=self.episode,
                              generation=self.generation,checkpoint_sha256=self.checkpoint_sha256)
            event['decision_envelope']=envelope.record()
            event['computation_profile']=computation_receipt()
            self.events.append(event);self.previous_packet=current.packet_id;self.last_update=now
            with self.condition:
                self.latest_decision=envelope
                self.consumed=version;self.condition.notify_all()
            if self.decision_mailbox is not None:self.decision_mailbox.publish(envelope)
            return event
        except Exception as error:
            event.update(reason=type(error).__name__+': '+str(error),computation_profile=computation_receipt(),runtime_rejection=copy.deepcopy(getattr(self.runtime,'last_rejection',None)));self.events.append(event);raise

    def latest_after(self, decision_id):
        if type(decision_id) is not int or decision_id < 0:
            raise ValueError("Explicit last consumed decision ID required")
        with self.condition:
            if self.error or self.stop_event.is_set():
                raise RuntimeError("Ticker terminal; no buffered output may be consumed")
            value = self.latest_decision
            if value is None or value.decision_id <= decision_id:
                return None
            value.validate(now=self.clock(), run_id=self.run_id, episode_id=self.episode,
                           generation=self.generation, checkpoint_sha256=self.checkpoint_sha256,
                           after_decision=decision_id)
            return value

    def _run(self):
        try:
            if self.flush_denormal:
                import torch
                self.numerics['supported']=torch.set_flush_denormal(True)
                if not self.numerics['supported']:raise RuntimeError('CPU flush-denormal mode unsupported')
            self.ready.set()
            while not self.stop_event.is_set():
                event=self.step_once()
                # Windows condition/event timeout granularity was measured at
                # ~16ms for a requested1ms. Standard Python sleep uses the
                # available short wait here; actual admission clocks still
                # enforce body50ms/view150ms, regardless of wake latency.
                elapsed=None if self.last_update is None else self.clock()-self.last_update
                time.sleep(.001 if event is None else (0 if elapsed is not None and elapsed>=self.period else .001))
        except Exception as error:
            self.error=dict(type=type(error).__name__,reason=str(error))
        finally:
            if self.decision_mailbox is not None:self.decision_mailbox.seal()
            if self.flush_denormal:
                import torch
                self.numerics['reset_to_nonflushing_before_thread_exit']=torch.set_flush_denormal(False)
            with self.condition:self.condition.notify_all()
