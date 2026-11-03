"""Research-only body-rate owner for one guarded unified flight circuit."""
import base64
import copy
import hashlib
import math
import threading
import time

import torch

from .ac_native_physics import PHYSICS_PREFIX, decode_physics_prefix
from .temporal_body_context import PerViewTemporalBodyContext, per_view_temporal_body_context
from .unified_multirate_shadow import UnifiedMultirateShadow
from .virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix


class SceneMailbox:
    """One pending immutable scene; never silently overwrite or invent a time."""
    def __init__(self):
        self.condition=threading.Condition();self.value=None;self.version=0;self.consumed=0;self.closed=False

    def publish(self, scene, context, capture_id, *, timeout=.05):
        if (not isinstance(scene,torch.Tensor) or scene.device.type!='cpu' or scene.dtype!=torch.float32
                or scene.shape!=(1,576) or not torch.isfinite(scene).all()
                or not isinstance(context,PerViewTemporalBodyContext)
                or type(capture_id) is not int or capture_id<0 or not 0<timeout<=.05):
            raise ValueError('Finite CPU scene and original context required')
        frozen=scene.detach().clone()
        context=copy.deepcopy(context)
        if context.source_synchronization_verified is not False:
            raise ValueError('Unverified original host timing required')
        context=per_view_temporal_body_context(context.visual_contexts, context.decision_body,
            render_start_lowers=context.render_start_lowers, camera_receipt_time=context.camera_receipt_time,
            decision_time=context.decision_time)
        deadline=time.perf_counter()+timeout
        with self.condition:
            while self.value is not None and self.version!=self.consumed and not self.closed:
                remaining=deadline-time.perf_counter()
                if remaining<=0:raise TimeoutError('Previous scene not consumed')
                self.condition.wait(remaining)
            now=time.perf_counter()
            if self.closed:raise RuntimeError('Scene mailbox closed')
            if self.value is not None and capture_id <= self.value[4]:
                raise ValueError('Scene capture ID must strictly advance')
            if context.decision_time>now or max(now-x for x in context.render_start_lowers)>.15:
                raise ValueError('Scene future or stale at publication')
            self.version+=1;self.value=(self.version,frozen,context,now,capture_id);self.condition.notify_all()
            return now

    def latest(self):
        with self.condition:return self.value,self.consumed

    def acknowledge(self,version):
        with self.condition:
            if self.value is None or self.version!=version or self.consumed==version:
                raise ValueError('Scene acknowledgment changed')
            self.consumed=version;self.condition.notify_all()

    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()


def defer_near_expiry(now, render_lowers, *, maximum_age=.15, reserve=.01):
    """Skip work near expiry; never extend the actual freshness limit."""
    if (not math.isfinite(now) or len(render_lowers) != 4
            or not all(math.isfinite(t) and t <= now for t in render_lowers)
            or not 0 < reserve < maximum_age <= .15):
        raise ValueError('Causal view times and bounded freshness reserve required')
    age = max(now - t for t in render_lowers)
    if age > maximum_age:
        raise ValueError('View render bound stale while waiting for new scene')
    return age >= maximum_age - reserve


class UnifiedLiveBodyTicker:
    """A single owner reads native receipts and advances exactly one circuit state."""
    def __init__(self, shadow, *, native_rows, native_identity, episode_id, period=.01):
        if (not isinstance(shadow,UnifiedMultirateShadow) or not isinstance(native_rows,list)
                or not isinstance(episode_id,str) or not episode_id or not .01<=period<=.04):
            raise ValueError('Explicit shadow, original native rows and bounded period required')
        self.shadow,self.native_rows,self.episode_id,self.period=shadow,native_rows,episode_id,period
        self.encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=native_identity)
        self.mailbox=SceneMailbox();self.stop_event=threading.Event();self.ready=threading.Event()
        self.events=[];self.error=None;self.thread=threading.Thread(target=self._run,name='unified-body-ticker',daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(3) or self.error:raise RuntimeError('Body ticker failed to start: '+str(self.error))

    def publish(self,scene,context,capture_id):
        if self.error or not self.thread.is_alive():raise RuntimeError('Body ticker unavailable')
        return self.mailbox.publish(scene,context,capture_id)

    def _run(self):
        index=0;latest=None;first=None;previous_id=None;last_update=None
        try:
            self.ready.set()
            while not self.stop_event.is_set():
                end=len(self.native_rows)
                while index<end:
                    row=self.native_rows[index];raw=base64.b64decode(row['authority']['prefix_base64'],validate=True);receipt=row['copy']['second_copy_end']
                    if hashlib.sha256(raw).hexdigest()!=row['authority']['prefix_sha256'] or receipt!=row['authority']['received_monotonic_seconds']:
                        raise ValueError('Original native bytes and receipt required')
                    sample=decode_physics_prefix(raw[:PHYSICS_PREFIX.size],receipt)
                    if sample.packet_id!=row['authority']['packet_id']:raise ValueError('Native packet identity differs')
                    candidate=decode_navigation_body_prefix(sample.prefix_bytes,receipt)
                    if latest is None:first=sample.packet_id;latest=candidate
                    elif sample.packet_id<latest.legacy.packet_id:raise ValueError('Packet reset requires new episode')
                    elif sample.packet_id==latest.legacy.packet_id:
                        if candidate.legacy.prefix_sha256!=latest.legacy.prefix_sha256:raise ValueError('Repeated packet changed bytes')
                    else:latest=candidate
                    index+=1
                now=time.perf_counter();available,consumed=self.mailbox.latest()
                if available is None or latest is None or latest.legacy.packet_id==first:
                    time.sleep(.002);continue
                if self.shadow.guard.model.previous_time is not None and now-latest.legacy.received_monotonic_seconds>.05:
                    raise ValueError('Current body receipt stale at ticker')
                if latest.legacy.packet_id==previous_id or last_update is not None and now-last_update<self.period:
                    time.sleep(.002);continue
                version,scene,frame,publication,capture=available;fresh=version!=consumed
                # Avoid starting an output with an almost-expired held image. A new
                # scene can arrive while waiting; the existing guard still validates
                # held body and actual output freshness, and expiry remains terminal.
                if not fresh and defer_near_expiry(now,frame.render_start_lowers):
                    time.sleep(.002);continue
                body=self.encoder.encode(latest,episode_id=self.episode_id,decision_monotonic_seconds=now)
                if fresh:self.mailbox.acknowledge(version)
                held=self.shadow.guard.model.previous_body_receipt;started=time.perf_counter()
                try:
                    output=self.shadow.update(when=now,body=body,new_scene=scene if fresh else None,new_context=frame if fresh else None,published_at=publication if fresh else None)
                except Exception as error:
                    self.events.append(dict(status='rejected',decision=now,started=started,finished=time.perf_counter(),packet=body.packet_id,body_receipt=body.original_received_monotonic_seconds,held_body_receipt=held,capture_id=capture,reason=type(error).__name__+': '+str(error),vehicle_control=False));raise
                self.events.append(dict(status='priming' if output is None else 'shadow_output',decision=now,started=started,finished=time.perf_counter(),packet=body.packet_id,body_receipt=body.original_received_monotonic_seconds,held_body_receipt=held,capture_id=capture,visual_replaced=fresh,publication=publication,view_render_lowers=list(frame.render_start_lowers),intent=None if output is None else output.intent[0].tolist(),vehicle_control=False))
                previous_id,last_update=body.packet_id,now
        except Exception as error:
            self.error={'type':type(error).__name__,'reason':str(error)}
        finally:
            self.ready.set();self.mailbox.close()

    def stop(self):
        self.stop_event.set();self.mailbox.close();self.thread.join(timeout=5)
        if self.thread.is_alive():raise TimeoutError('Owned body ticker remains live')
