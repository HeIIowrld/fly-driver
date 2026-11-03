"""Process-backed multirate research shadow; no vehicle interface."""
import math
import time
import torch
from .temporal_body_context import PerViewTemporalBodyContext, per_view_temporal_body_context
from .unified_host_guard import GuardedIntent
from .unified_multirate_shadow import UnifiedMultirateShadow
from .virtual_navigation_body import NavigationBodyObservation


class ProcessUnifiedMultirateShadow(UnifiedMultirateShadow):
    """Parent enforces freshness; the worker alone owns numerical circuit state."""
    def __init__(self,guard,worker,*,goals,clock=time.perf_counter):
        super().__init__(guard,goals=goals)
        self.worker,self.clock=worker,clock
        self.worker_observations=[]

    def reset(self):
        try:
            self.worker.reset()
            super().reset()
        except Exception:
            self.guard.failed=True
            raise

    def update(self, *, when, body, new_scene=None, new_context=None, published_at=None):
        if self.guard.failed:raise RuntimeError('Guard failed; explicit restart required')
        try:
            if not isinstance(body,NavigationBodyObservation) or body.decision_monotonic_seconds!=when:
                raise ValueError('Original body at decision time required')
            replacing=new_scene is not None
            if replacing!=(new_context is not None) or replacing!=(published_at is not None):
                raise ValueError('New scene, context and publication required together')
            if replacing:
                if (not isinstance(new_scene,torch.Tensor) or new_scene.device.type!='cpu' or new_scene.dtype!=torch.float32 or new_scene.shape!=(1,576) or not torch.isfinite(new_scene).all() or not isinstance(new_context,PerViewTemporalBodyContext) or isinstance(published_at,bool) or not isinstance(published_at,(int,float)) or not math.isfinite(published_at) or not new_context.camera_receipt_time<=new_context.decision_time<=published_at<=when):
                    raise ValueError('Completed original causal CPU scene required')
                scene,frame,publication=new_scene.detach().clone(),new_context,float(published_at)
            else:scene,frame,publication=self.scene,self.frame,self.published_at
            if scene is None or frame is None or publication is None or publication>when:
                raise ValueError('Available original published scene required')
            context=per_view_temporal_body_context(frame.visual_contexts,body,render_start_lowers=frame.render_start_lowers,camera_receipt_time=frame.camera_receipt_time,decision_time=when)
            started=self.clock();model=self.guard.model;held=model.previous_body_receipt;priming=model.previous_time is None
            if not math.isfinite(started) or started<when or not priming and held is None:
                raise ValueError('Monotonic decision and held body required')
            ipc_started=time.perf_counter()
            try:
                response=self.worker.step(when=when,scene=scene.numpy().copy(),context=context)
            except Exception as error:
                self.worker_observations.append(dict(decision=when,ipc_started=ipc_started,ipc_finished=time.perf_counter(),error=type(error).__name__+': '+str(error)))
                raise
            self.worker_observations.append(dict(decision=when,ipc_started=ipc_started,ipc_finished=time.perf_counter(),worker_started=response.get('started'),worker_finished=response.get('finished')))
            completed=self.clock();limits=model.freshness_limits
            if not math.isfinite(completed) or completed<started:raise ValueError('Monotonic output time required')
            if not 0<=completed-body.original_received_monotonic_seconds<=limits.maximum_body_receipt_age_seconds:
                raise ValueError('Current body receipt stale at output')
            if held is not None and not 0<=completed-held<=limits.maximum_body_receipt_age_seconds:
                raise ValueError('Held body receipt stale at output')
            if not all(0<=completed-t<=limits.maximum_view_render_age_seconds for t in frame.render_start_lowers):
                raise ValueError('View render bound stale at output')
            # Mirror only clock metadata for the existing ticker. Numerical state stays in worker.
            model.previous_time=float(when);model.previous_body_receipt=float(body.original_received_monotonic_seconds)
            self.scene,self.frame,self.published_at=scene,frame,publication
            if priming:return None
            return GuardedIntent(intent=torch.from_numpy(response['intent'].copy()),muscles=torch.from_numpy(response['muscles'].copy()),decision_time=float(when),completed_time=float(completed))
        except Exception:
            self.guard.failed=True
            raise
