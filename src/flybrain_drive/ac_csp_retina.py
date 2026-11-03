"""Independent CSP camera receptors with conservative source ages and native organs."""

import copy
import hashlib
import threading

import numpy as np

from .ac_camera_mount import configured_mount
from .ac_pixel_retina import NativePixelRetina, digest

VIEW_ORDER = ('front', 'left', 'right', 'rear')


class CSPCameraRetina:
    sample_fields = {'rgb', 'capture_id', 'generation', 'render_started_lower', 'received'}

    def __init__(self, config):
        configured_mount(config)
        self.config = copy.deepcopy(config)
        if (config['schema'] != 'flybrain.ac.csp_retina.v1'
                or tuple(config['view_order']) != VIEW_ORDER
                or set(config['views']) != set(VIEW_ORDER)
                or config['clock_domain'] != 'host_monotonic_interval_bound'):
            raise ValueError('Explicit four-camera interval-clock contract required')
        self.body = NativePixelRetina(config['native_retina'])
        self.count, self.max_age = self.body.count, self.body.max_age
        self.directions = []
        for name in VIEW_ORDER:
            view = config['views'][name]
            px, py = view['projection_x'], view['projection_y']
            basis = np.array([view['forward_body'], view['right_body'], view['up_body']])
            if (not np.isfinite([px, py]).all() or min(px, py) <= 0
                    or basis.shape != (3, 3) or not np.isfinite(basis).all()
                    or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5)):
                raise ValueError('Finite camera projection and orthonormal source basis required')
            uv = self.body.uv
            rays = (basis[0] + (2*uv[:, 0:1]-1)/px*basis[1]
                    + (1-2*uv[:, 1:2])/py*basis[2])
            self.directions.append(rays/np.linalg.norm(rays, axis=1, keepdims=True))
        self.directions = np.array(self.directions)
        self.history, self.identities = {}, {}
        self.generation, self.last_decision = None, None
        self.pixel_cache, self.pixel_cache_lock = {}, threading.Lock()
        self.pixel_cache_hits, self.pixel_cache_misses = 0, 0

    def reset_episode(self):
        """Clear temporal sensor state only at an explicit episode or run restart.

        Pixel-cache entries are immutable, content-addressed preprocessing results;
        they contain no temporal state and may be reused across episodes.
        """
        self.history.clear()
        self.identities.clear()
        self.generation = None
        self.last_decision = None
        self.body.scalar_history.clear()

    def _pixel_values(self, pixels, pixel_hash):
        with self.pixel_cache_lock:
            value = self.pixel_cache.get(pixel_hash)
        if value is None:
            h, w = self.config['native_retina']['image_shape']
            value = self.body._sample(pixels, [0, 0, w, h])
            value.setflags(write=False)
            with self.pixel_cache_lock:
                if pixel_hash not in self.pixel_cache:
                    if len(self.pixel_cache) >= 64:
                        self.pixel_cache.pop(next(iter(self.pixel_cache)))
                    self.pixel_cache[pixel_hash] = value
        return value

    def precompute_pixels(self, pixels):
        """Pure pixel transduction only; safe on the acquisition worker before publication."""
        if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
                or pixels.shape != (*self.config['native_retina']['image_shape'], 3)):
            raise ValueError('Exact source RGB pixels required for preprocessing')
        self._pixel_values(pixels, hashlib.sha256(pixels.tobytes()).hexdigest())

    @property
    def required_native_fields(self):
        return self.body.required_native_fields

    @property
    def wheel_spec(self):
        return self.body.wheel_spec

    @property
    def yaw_spec(self):
        return self.body.yaw_spec

    @property
    def calibration(self):
        return dict(view_order=list(VIEW_ORDER),
            view_hashes={name: digest(dict(config=self.config, source=name))
                         for name in VIEW_ORDER},
            organ_size=132,
            organ_calibration_sha256=self.body.calibration['organ_calibration_sha256'],
            max_view_age=self.max_age, contrast_scale=self.body.scale,
            required_views=list(VIEW_ORDER), preprocessing_kind='csp_independent_pixel_receptors',
            clock_domain='host_monotonic_interval_bound', body_ray_registration_verified=False,
            projection_parameter_source='returned CSP matrices plus declared body-frame axes',
            exposure_synchronization_verified=False)

    def pack(self, samples, *, decision_time, native_scalars=None, native_receipt=None):
        if (set(samples) - set(VIEW_ORDER) or not np.isfinite(decision_time)
                or decision_time < 0 or self.last_decision is not None
                and decision_time <= self.last_decision):
            raise ValueError('Known visual sources and increasing host decisions required')
        shape = (4, self.count)
        rgb, contrast = np.zeros((*shape, 3)), np.zeros((*shape, 3))
        valid, contrast_valid = np.zeros(shape, bool), np.zeros(shape, bool)
        age = np.full(4, self.max_age*10)
        history, identities = self.history.copy(), self.identities.copy()
        generation = self.generation
        for index, name in enumerate(VIEW_ORDER):
            sample = samples.get(name)
            if sample is None:
                history.pop(name, None)
                continue
            if set(sample) != self.sample_fields:
                raise ValueError('Only pixels and causal visual identity/times accepted')
            pixels, stamp, receipt = (sample[k] for k in
                                     ('rgb', 'render_started_lower', 'received'))
            capture, producer = sample['capture_id'], sample['generation']
            if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
                    or pixels.shape != (*self.config['native_retina']['image_shape'], 3)
                    or type(capture) is not int or capture < 0
                    or not isinstance(producer, str) or not producer
                    or not np.isfinite([stamp, receipt]).all()
                    or not 0 <= stamp <= receipt <= decision_time):
                raise ValueError('Exact RGB shape and causal source identity/times required')
            if generation is None:
                generation = producer
            if generation != producer:
                raise ValueError('Camera producer changed; reset explicitly at episode boundary')
            identity = (capture, stamp, receipt, hashlib.sha256(pixels.tobytes()).hexdigest())
            old = identities.get(name)
            if old is not None and (capture < old[0] or capture == old[0] and identity != old
                    or capture > old[0] and stamp <= old[1]):
                raise ValueError('Visual source identity changed or moved backwards')
            identities[name] = identity
            age[index] = min(decision_time-stamp, self.max_age*10)
            if age[index] > self.max_age:
                history.pop(name, None)
                continue
            pixel_hash = identity[-1]
            with self.pixel_cache_lock:
                ready = pixel_hash in self.pixel_cache
            self.pixel_cache_hits += int(ready)
            self.pixel_cache_misses += int(not ready)
            value = self._pixel_values(pixels, pixel_hash)
            rgb[index], valid[index] = value, True
            prior = history.get(name)
            if prior is not None:
                old_id, old_stamp, old_value, old_rate, old_valid = prior
                if capture == old_id:
                    contrast[index], contrast_valid[index] = old_rate, old_valid
                elif stamp-old_stamp <= self.max_age:
                    contrast[index] = np.tanh((value-old_value)/(stamp-old_stamp)/self.body.scale)
                    contrast_valid[index] = True
            history[name] = (capture, stamp, value.copy(), contrast[index].copy(),
                             contrast_valid[index].copy())
        old_body = self.body.scalar_history
        self.body.scalar_history = old_body.copy()
        try:
            organs, organ_valid = self.body._organs(native_scalars, native_receipt, decision_time)
        except Exception:
            self.body.scalar_history = old_body
            raise
        self.history, self.identities, self.generation = history, identities, generation
        self.last_decision = decision_time
        return {key: np.asarray(value, dtype=np.float32) for key, value in dict(
            view_rgb=rgb, view_contrast=contrast, view_valid=valid,
            contrast_valid=contrast_valid, view_age=age, organs=organs,
            organ_valid=organ_valid).items()}
