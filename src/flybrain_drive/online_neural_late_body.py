"""Visual-first neural shadow with fresh decision-body receipt; no actuation."""

import hashlib
import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .flight_stream import held_input_batches
from .flyvis_camera import VIEW_ORDER
from .online_neural_shadow import OnlineNeuralShadow
from .relative_airflow_sensor import missing_airflow


def output_host_freshness(*, has_intent, selected_body_age, held_body_age, render_ages):
    """Both the next-drive input and body held over the output interval must be fresh."""
    if (type(has_intent) is not bool or not has_intent or selected_body_age is None
            or held_body_age is None or len(render_ages) != 4):
        return False
    values = [selected_body_age, held_body_age, *render_ages]
    return bool(np.isfinite(values).all() and 0 <= selected_body_age <= .05
                and 0 <= held_body_age <= .05 and min(render_ages) >= 0
                and max(render_ages) <= .15)


class LateBodyNeuralShadow(OnlineNeuralShadow):
    """Advance vision at its own snapshot; choose the decision body after vision completes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clock = time.perf_counter
        self.held_body_receipt = None

    @torch.no_grad()
    def process_late(self, frame, event, select_after_vision, directory):
        if self.failed:
            raise RuntimeError('Failed neural shadow cannot be silently resumed')
        started, visual_time = self.clock(), event['visual_snapshot']
        try:
            if self.previous_time is not None and not 0 < visual_time-self.previous_time <= .15:
                raise ValueError('Original visual snapshot gap exceeds 150ms; no reset or filling')
            views = {view['view_id']: view for view in frame['views']}
            retinal, hashes = [], []
            for name in VIEW_ORDER:
                expected = f"{event['index']:04d}-{name}.png"
                if views[name].get('file') != expected:
                    raise ValueError('Exact same capture image identity required')
                data = (Path(directory)/expected).read_bytes()
                hashes.append(hashlib.sha256(data).hexdigest())
                with Image.open(io.BytesIO(data)) as picture:
                    retinal.append(self.projector.sample(np.asarray(picture.convert('RGB')).copy()))
            with torch.device('cuda'):
                current = torch.as_tensor(np.stack(retinal), dtype=torch.float32, device='cuda')
                if self.previous_time is not None:
                    for dt, count in held_input_batches(visual_time-self.previous_time):
                        while count:
                            n = min(count, 16)
                            source = self.held[:, None, None, :].expand(-1, n, -1, -1)
                            self.vis_state = self.network.simulate(source, dt=dt,
                                initial_state=self.vis_state, as_states=True)[-1]
                            count -= n
                motion = self.vis_state.nodes.activity[:, self.motion_indices]
                scene = self.model.backbone.scene_pool(torch.cat((current[:, None], motion), dim=1)[None])
                # Current body must be read after the visual work has actually completed.
                torch.cuda.synchronize()
                vision_finished = self.clock()
                context = select_after_vision()
                decision = event['decision']
                if decision < vision_finished:
                    raise ValueError('Fresh current-body selection must follow visual completion')
                required = [0, 1, 4, 16, 17, 18, 19, 20]
                valid = context is not None and bool(context.decision_body.valid[required].all())
                intent = None
                if not valid and self.flight_active:
                    self.flight_stopped = True
                prior_held_receipt = self.held_body_receipt
                if valid and not self.flight_stopped:
                    self.flight_active = True
                    intent, _ = self.stream.decision(decision_time=decision, visual=self.visual,
                        visual_valid=self.visual_valid, body_context=context, goals=self.goals,
                        scene_features=scene, scene_valid=self.scene_valid,
                        airflow_observation=missing_airflow(decision))
                    intent = intent.cpu().numpy()[0].tolist()
                    self.held_body_receipt = context.decision_body.original_received_monotonic_seconds
                torch.cuda.synchronize()
                self.held, self.previous_time = current.detach().clone(), visual_time
            finished = self.clock()
            receipt = None if context is None else context.decision_body.original_received_monotonic_seconds
            body_age = None if receipt is None else finished-receipt
            held_body_age = None if prior_held_receipt is None else finished-prior_held_receipt
            render_ages = [finished-x for x in event['render_start_lower']]
            fresh = output_host_freshness(has_intent=intent is not None,
                selected_body_age=body_age, held_body_age=held_body_age, render_ages=render_ages)
            return {'status': 'intent' if intent is not None else (
                'flight_stopped_after_body_gap' if self.flight_stopped else 'missing_required_body'),
                'intent': intent, 'started': started, 'vision_finished': vision_finished,
                'finished': finished, 'processing_seconds': finished-started,
                'camera_receipt_to_output_seconds': finished-event['camera_receipt'],
                'body_receipt_age_at_output_seconds': body_age,
                'previous_held_body_receipt': prior_held_receipt,
                'previous_held_body_receipt_age_at_output_seconds': held_body_age,
                'new_body_affects_future_interval_not_current_state': True,
                'view_render_age_upper_at_output_seconds': render_ages,
                'source_images_sha256': hashes, 'source_images_hashed_at_neural_read': True,
                'output_host_freshness_passed': bool(fresh), 'vehicle_control': False,
                'physical_sample_or_exposure_time_verified': False,
                'flight_segment_started': self.flight_active}
        except Exception:
            self.failed = True
            raise
