"""Single-owner body-rate inference for the learned native flight runtime.

Vision publication and native sampling never advance the circuit themselves.
A rejected tick is terminal until an explicit new run; no resets or actuators.
"""
import base64,copy,hashlib,threading,time
from .ac_native_physics import PHYSICS_PREFIX
from .native_multisensory_extension import decode_native_extension
from .per_view_flyvis_stream import VisualFlightObservation
from .motion_spatial_runtime import MotionSpatialDecisionRuntime,MotionSpatialObservation
from .motion_per_view_flyvis_stream import MotionVisualFlightObservation
from .temporal_body_context import PerViewTemporalBodyContext,per_view_temporal_body_context
from .unified_live_ticker import defer_near_expiry
from .virtual_navigation_body import NavigationBodyEncoder,decode_navigation_body_prefix


class NativeFlightBodyTicker:
    def __init__(self,runtime,*,native_rows,native_identity,episode_id,period=.01,clock=time.perf_counter,source_error=None,flush_denormal=False):
        if not isinstance(native_rows,list) or not .01<=period<=.04 or runtime.episode!=episode_id:
            raise ValueError('Explicit started runtime, original rows and bounded body cadence required')
        self.runtime,self.rows,self.episode,self.period,self.clock=runtime,native_rows,episode_id,period,clock
        if source_error is not None and not callable(source_error):raise ValueError('Explicit source-health callback required')
        if type(flush_denormal) is not bool:raise ValueError('Explicit CPU numerical-mode boolean required')
        self.flush_denormal=flush_denormal;self.numerics=dict(requested_flush_denormal=flush_denormal,scope='Owned newly created CPU ticker thread only; numerical execution approximation')
        self.source_error=source_error
        self.encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=native_identity)
        self.condition=threading.Condition();self.latest_visual=None;self.version=0;self.consumed=0
        self.position=0;self.latest=None;self.extension=None;self.first=None;self.previous_packet=None;self.last_update=None
        self.events=[];self.wait_events=[];self.error=None;self.stop_event=threading.Event();self.ready=threading.Event()
        self.thread=threading.Thread(target=self._run,name='native-flight-body-ticker',daemon=True)

    def publish(self,visual,context,*,timeout=.05):
        if not isinstance(visual,VisualFlightObservation) or not isinstance(context,PerViewTemporalBodyContext):
            raise ValueError('Original typed visual observation and view-body history required')
        if (visual.episode_id!=self.episode or context.decision_body.episode_id!=self.episode
                or visual.camera_received!=context.camera_receipt_time
                or visual.render_start_lowers!=context.render_start_lowers):
            raise ValueError('Visual/body episode or source clocks differ')
        if isinstance(self.runtime,MotionSpatialDecisionRuntime):
            if not isinstance(visual,MotionVisualFlightObservation) or hashlib.sha256(visual.motion_features.tobytes()).hexdigest()!=visual.motion_features_sha256:
                raise ValueError('Motion runtime requires intact typed motion publication')
        deadline=self.clock()+timeout
        with self.condition:
            while self.version!=self.consumed and not self.error and not self.stop_event.is_set():
                remaining=deadline-self.clock()
                if remaining<=0:raise TimeoutError('Previous native-flight scene is unconsumed')
                self.condition.wait(remaining)
            if self.error or self.stop_event.is_set():raise RuntimeError('Native body ticker unavailable')
            if self.latest_visual is not None and visual.capture_id<=self.latest_visual[0].capture_id:
                raise ValueError('Visual capture must advance')
            now=self.clock()
            if max(now-t for t in visual.render_start_lowers)>.15 or now<visual.camera_received:
                raise ValueError('Published visual source is stale or future')
            self.latest_visual=(copy.deepcopy(visual),copy.deepcopy(context),now)
            self.version+=1;self.condition.notify_all()
            return now

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
            selected=self.latest_visual;version=self.version
        if selected is None or self.latest is None or self.latest.legacy.packet_id==self.first:return None
        now=self.clock()
        if now-self.latest.legacy.received_monotonic_seconds>.05:raise ValueError('Current native body stale at ticker')
        if self.previous_packet==self.latest.legacy.packet_id or self.last_update is not None and now-self.last_update<self.period:return None
        visual,historical,publication=selected
        if max(now-t for t in visual.render_start_lowers)>=.125:
            self.wait_events.append(dict(when=now,capture_id=visual.capture_id,publication=publication,view_age=max(now-t for t in visual.render_start_lowers),current_body_age=now-self.latest.legacy.received_monotonic_seconds))
        if defer_near_expiry(now,visual.render_start_lowers,reserve=.025):return None
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
            event['computation_profile']=computation_receipt()
            self.events.append(event);self.previous_packet=current.packet_id;self.last_update=now
            with self.condition:
                self.consumed=version;self.condition.notify_all()
            return event
        except Exception as error:
            event.update(reason=type(error).__name__+': '+str(error),computation_profile=computation_receipt(),runtime_rejection=copy.deepcopy(getattr(self.runtime,'last_rejection',None)));self.events.append(event);raise

    def _run(self):
        try:
            if self.flush_denormal:
                import torch
                self.numerics['supported']=torch.set_flush_denormal(True)
                if not self.numerics['supported']:raise RuntimeError('CPU flush-denormal mode unsupported')
            self.ready.set()
            while not self.stop_event.is_set():
                self.step_once()
                # Windows condition/event timeout granularity was measured at
                # ~16ms for a requested1ms. Standard Python sleep uses the
                # available short wait here; actual admission clocks still
                # enforce body50ms/view150ms, regardless of wake latency.
                elapsed=None if self.last_update is None else self.clock()-self.last_update
                time.sleep(0 if elapsed is not None and elapsed>=self.period else .001)
        except Exception as error:
            self.error=dict(type=type(error).__name__,reason=str(error))
        finally:
            if self.flush_denormal:
                import torch
                self.numerics['reset_to_nonflushing_before_thread_exit']=torch.set_flush_denormal(False)
            with self.condition:self.condition.notify_all()

    def start(self):
        self.thread.start()
        if not self.ready.wait(3):raise TimeoutError('Native ticker startup unknown')

    def stop(self):
        self.stop_event.set()
        with self.condition:self.condition.notify_all()
        self.thread.join(5)
        if self.thread.is_alive():raise TimeoutError('Native body ticker remains live')
