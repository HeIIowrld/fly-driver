"""Reusable four-source MuJoCo observation boundary for collection and evaluation."""

import time

from .mujoco_multiview import MuJoCoMultiViewRenderer
from .mujoco_sensory import MuJoCoSensoryBridge
from .multiview_sensory import MultiViewPacker
from .sensor_contract import ActorObservation, SampleTime, VisualViewSample


class MultiViewMuJoCoRuntime:
    def __init__(
        self, env, views, organs, *, periods, source_size=96, max_view_age=0.15, contrast_scale=10
    ):
        if len(periods) != len(views) or any(type(p) is not int or p < 1 for p in periods):
            raise ValueError("An explicit positive update period per source is required")
        self.env, self.views, self.periods = env, tuple(views), tuple(periods)
        self.packer = MultiViewPacker(
            views, organs, max_view_age=max_view_age, contrast_scale=contrast_scale
        )
        self.renderer = MuJoCoMultiViewRenderer(env, views, source_size)
        self.body = MuJoCoSensoryBridge(env, rows=2, columns=3)
        self.projectors = {v.source_id: v.projector() for v in views}
        self.episode, self.cached, self.ids = None, {}, {}

    def observe(self, observation, episode_id, sequence_id):
        if episode_id != self.episode:
            self.episode, self.cached = episode_id, {}
            self.ids = {v.source_id: -1 for v in self.views}
        scheduled = [
            v.source_id
            for v, period in zip(self.views, self.periods, strict=True)
            if v.source_id not in self.cached or sequence_id % period == 0
        ]
        images = self.renderer.render(scheduled)
        clock = float(self.env.data.time)
        for view in self.views:
            if view.source_id not in images:
                continue
            response, valid, _ = self.projectors[view.source_id].sample_pixels(
                images[view.source_id]
            )
            self.ids[view.source_id] += 1
            self.cached[view.source_id] = VisualViewSample(
                view.source_id,
                self.ids[view.source_id],
                response,
                valid,
                SampleTime(clock, time.monotonic()),
                view.sha256,
                "declared_sensor_model",
            )
        legacy = self.body.observe(
            self.env,
            observation,
            episode_id=episode_id,
            sequence_id=sequence_id,
            received_monotonic_seconds=time.monotonic(),
        )
        return ActorObservation(
            episode_id,
            sequence_id,
            clock,
            time.monotonic(),
            (),
            legacy.body,
            views=tuple(self.cached[v.source_id] for v in self.views),
        )

    def close(self):
        self.renderer.close()
