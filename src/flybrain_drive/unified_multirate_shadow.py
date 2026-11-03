"""Research-only held-vision adapter into one guarded flight circuit; no control."""
import math
import torch

from .temporal_body_context import PerViewTemporalBodyContext, per_view_temporal_body_context
from .unified_host_guard import UnifiedHostGuard
from .virtual_navigation_body import NavigationBodyObservation


class UnifiedMultirateShadow:
    """Consume immutable published scene and current body; suppress stale output."""
    def __init__(self, guard, *, goals):
        if not isinstance(guard, UnifiedHostGuard) or not isinstance(goals, torch.Tensor) or goals.shape != (1,2) or not torch.isfinite(goals).all():
            raise ValueError('Guarded unified model and finite actor goals required')
        if guard.model.previous_time is not None:
            raise ValueError('Fresh episode required')
        self.guard, self.goals = guard, goals.detach().clone()
        self.scene = self.frame = self.published_at = None

    def reset(self):
        self.guard.reset()
        self.scene = self.frame = self.published_at = None

    def update(self, *, when, body, new_scene=None, new_context=None, published_at=None):
        if self.guard.failed:
            raise RuntimeError('Guard failed; explicit restart required')
        if not isinstance(body, NavigationBodyObservation) or body.decision_monotonic_seconds != when:
            raise ValueError('Original body at decision time required')
        replacing = new_scene is not None
        if replacing != (new_context is not None) or replacing != (published_at is not None):
            raise ValueError('New scene, original context and publication must arrive together')
        if replacing:
            if (not isinstance(new_scene,torch.Tensor) or new_scene.device.type != 'cpu'
                    or new_scene.dtype != torch.float32 or new_scene.shape != (1,576)
                    or not torch.isfinite(new_scene).all() or not isinstance(new_context,PerViewTemporalBodyContext)
                    or isinstance(published_at,bool) or not isinstance(published_at,(int,float))
                    or not math.isfinite(published_at)
                    or not new_context.camera_receipt_time <= new_context.decision_time <= published_at <= when):
                raise ValueError('Completed finite CPU scene with original causal publication required')
            scene,frame,publication=new_scene.detach().clone(),new_context,float(published_at)
        else:
            scene,frame,publication=self.scene,self.frame,self.published_at
        if scene is None or frame is None or publication is None or publication > when:
            raise ValueError('Available original published scene required')
        context=per_view_temporal_body_context(frame.visual_contexts,body,
            render_start_lowers=frame.render_start_lowers,camera_receipt_time=frame.camera_receipt_time,decision_time=when)
        output=self.guard.step(decision_time=when,racing_goals=self.goals,scene_features=scene,
            scene_valid=torch.ones((1,4),dtype=torch.bool,device=scene.device),body_context=context)
        self.scene,self.frame,self.published_at=scene,frame,publication
        return output
