"""Native RGB pixel-space ommatidia and host-clock scalar surrogates, not calibrated fly optics.

No simulation timestamps, mirror geometry, roadwheel angles or body-axis transforms are
invented here. Position/progress/map values are not accepted. Unsupported organ groups and
unobserved camera sources remain masked. Unit conversions are declared engineering models.
"""

import hashlib
import json
import math

import numpy as np

VIEW_ORDER = ('front', 'mirror_left', 'mirror_right', 'mirror_rear')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class NativePixelRetina:
    def __init__(self, config):
        self.config = json.loads(json.dumps(config))
        if (config['schema'] != 'flybrain.ac.pixel_retina.v1'
                or config['clock_domain'] != 'host_monotonic_receipt'
                or set(config['regions']) != set(VIEW_ORDER)
                or config['regions']['front'] is None):
            raise ValueError('Explicit pixel regions and host-clock contract required')
        self.count, self.columns = config['receptors_per_view'], config['columns']
        self.max_age, self.scale = config['max_view_age'], config['contrast_scale']
        sigma = config['acceptance_sigma_pixels']
        if (type(self.count) is not int or type(self.columns) is not int
                or not 1 <= self.columns <= self.count <= 10000
                or not np.isfinite([self.max_age, self.scale, sigma]).all()
                or not 0 < self.max_age <= 1 or self.scale <= 0 or not 0 < sigma <= 10):
            raise ValueError('Finite pixel retina layout required')
        index = np.arange(self.count)
        row, col = index // self.columns, index % self.columns
        rows = math.ceil(self.count / self.columns)
        self.uv = np.c_[(col + .5 + np.where(row % 2, .25, -.25)) / self.columns,
                       (row + .5) / rows]
        addresses = np.c_[np.ones(self.count), 1 - 2*self.uv[:, 0], 1 - 2*self.uv[:, 1]]
        addresses /= np.linalg.norm(addresses, axis=1, keepdims=True)
        # Unit embedding supplies source-local neighborhoods, never physical body rays.
        self.local_addresses = np.tile(addresses[None], (4, 1, 1))
        h, w = config['image_shape']
        for region in config['regions'].values():
            if region is not None:
                x0, y0, x1, y1 = region
                if (any(type(v) is not int for v in region)
                        or not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h):
                    raise ValueError('Pixel region outside exact source shape')
        expected = dict(suspensionTravel=18, tyreCoreTemperature=54, wheelLoad=90)
        specs = config['scalar_organs']
        if (len(specs) != 3 or {s['native_field'] for s in specs} != set(expected)
                or any(s['feature_offset'] != expected[s['native_field']]
                       or not np.isfinite([s[k] for k in (
                           'minimum', 'maximum', 'width', 'change_scale', 'offset')]).all()
                       or s['minimum'] >= s['maximum'] or min(s['width'], s['change_scale']) <= 0
                       for s in specs)):
            raise ValueError('Explicit native population coding and 132-feature layout required')
        self.wheel_spec = config.get('wheel_rotation_organs')
        if self.wheel_spec is not None:
            w = self.wheel_spec
            centers = np.asarray(w['centers'], dtype=float)
            if (w['native_field'] != 'wheelAngularSpeed' or w['feature_offset'] != 0
                    or w['wheel_pairs'] != [[0, 2], [1, 3]]
                    or w['encoding'] != 'recruited_tanh_magnitude'
                    or centers.shape != (7,) or not np.isfinite(centers).all()
                    or centers[0] != 0 or np.any(np.diff(centers) <= 0)
                    or not np.isfinite([w['width'], w['change_scale']]).all()
                    or min(w['width'], w['change_scale']) <= 0):
                raise ValueError('Declared wheel magnitude coding in reserved slots 0..17 required')
        self.yaw_spec = config.get('yaw_rotation_organs')
        if self.yaw_spec is not None:
            y = self.yaw_spec
            if (y['native_field'] != 'localAngularVel' or y['axis_index'] != 1
                    or y['feature_offset'] != 126 or y['encoding'] != 'signed_tanh_pair'
                    or not np.isfinite([y['gain'], y['scale']]).all()
                    or y['gain'] != 1. or y['scale'] <= 0):
                raise ValueError('Reviewed native yaw axis1 and reserved slots126..127 required')
        self.previous_frame, self.last_decision = None, None
        self.visual_history, self.scalar_history = {}, {}
        self.sampling_plans = {}

    @property
    def required_native_fields(self):
        return {'suspensionTravel', 'tyreCoreTemperature', 'wheelLoad'} | (
            {'wheelAngularSpeed'} if self.wheel_spec is not None else set()) | (
            {'localAngularVel'} if self.yaw_spec is not None else set())

    @property
    def calibration(self):
        organs = self.config['scalar_organs'] if self.wheel_spec is None else dict(
            scalar_organs=self.config['scalar_organs'], wheel_rotation_organs=self.wheel_spec)
        if self.yaw_spec is not None:
            organs = dict(scalar_organs=self.config['scalar_organs'],
                          wheel_rotation_organs=self.wheel_spec, yaw_rotation_organs=self.yaw_spec)
        return dict(view_order=list(VIEW_ORDER),
            view_hashes={name: digest(dict(config=self.config, source=name))
                         for name in VIEW_ORDER},
            organ_size=132, organ_calibration_sha256=digest(organs),
            max_view_age=self.max_age, contrast_scale=self.scale,
            required_views=['front'], preprocessing_kind='native_pixel_lattice',
            clock_domain='host_monotonic_receipt', body_ray_registration_verified=False,
            exposure_synchronization_verified=False)

    def _sample(self, rgb, region):
        sigma = self.config['acceptance_sigma_pixels']
        key = (*region, sigma)
        plan = self.sampling_plans.get(key)
        if plan is None:
            plan = self._sampling_plan(region, sigma)
            self.sampling_plans[key] = plan
        result = np.zeros((self.count, 3))
        for ix, iy, jx, jy, ax, ay, wx, wy in plan:
            # Keep the original arithmetic order for bit-for-bit pixel replay.
            value = (rgb[iy, ix]*(1-ax)*(1-ay) + rgb[iy, jx]*ax*(1-ay)
                     + rgb[jy, ix]*(1-ax)*ay + rgb[jy, jx]*ax*ay)
            result += value * wx * wy / 255
        return result

    def _sampling_plan(self, region, sigma):
        x0, y0, x1, y1 = region
        xy = self.uv * [x1-x0-1, y1-y0-1] + [x0, y0]
        plan = []
        for dx, wx in [(-math.sqrt(3)*sigma, 1/6), (0, 2/3), (math.sqrt(3)*sigma, 1/6)]:
            for dy, wy in [(-math.sqrt(3)*sigma, 1/6), (0, 2/3), (math.sqrt(3)*sigma, 1/6)]:
                x = np.clip(xy[:, 0]+dx, x0, x1-1)
                y = np.clip(xy[:, 1]+dy, y0, y1-1)
                ix, iy = x.astype(int), y.astype(int)
                jx, jy = np.minimum(ix+1, x1-1), np.minimum(iy+1, y1-1)
                ax, ay = (x-ix)[:, None], (y-iy)[:, None]
                plan.append((ix, iy, jx, jy, ax, ay, wx, wy))
        return plan

    def _organs(self, fields, sample_time, decision):
        values, masks = np.zeros(132), np.zeros(132, bool)
        if fields is None:
            self.scalar_history.clear()
            return values, masks
        if set(fields) - self.required_native_fields:
            raise ValueError('Only declared native organ inputs accepted')
        if type(sample_time) not in (int, float) or not np.isfinite(sample_time):
            raise ValueError('Explicit host body receipt required')
        if not 0 <= decision-sample_time <= self.max_age:
            self.scalar_history.clear()
            return values, masks
        for spec in self.config['scalar_organs']:
            raw = fields.get(spec['native_field'])
            if raw is None:
                self.scalar_history.pop(spec['native_field'], None)
                continue
            raw = np.asarray(raw, dtype=float)
            if raw.shape != (4,):
                raise ValueError('FL/FR/RL/RR scalar channels required')
            valid = np.isfinite(raw)
            converted = np.where(valid, raw, 0) + spec['offset']
            if spec['native_field'] != 'suspensionTravel':
                valid &= converted >= 0
            centers = np.linspace(spec['minimum'], spec['maximum'], 7)
            population = np.zeros((4, 9))
            mask = np.zeros((4, 9), bool)
            population[:, :7] = np.exp(-.5*((converted[:, None]-centers)/spec['width'])**2)
            mask[:, :7] = valid[:, None]
            previous = self.scalar_history.get(spec['native_field'])
            if previous is not None:
                stamp, old, old_valid, rates, rate_valid = previous
                if sample_time < stamp:
                    raise ValueError('Host body sample moved backwards')
                if sample_time == stamp:
                    if not np.array_equal(converted, old) or not np.array_equal(valid, old_valid):
                        raise ValueError('Reused body sample changed')
                    population[:, 7:], mask[:, 7:] = rates, rate_valid
                elif sample_time-stamp <= self.max_age:
                    rate = (converted-old)/(sample_time-stamp)/spec['change_scale']
                    population[:, 7:] = np.tanh(np.c_[np.maximum(rate, 0), np.maximum(-rate, 0)])
                    mask[:, 7:] = (valid & old_valid)[:, None]
            self.scalar_history[spec['native_field']] = (
                sample_time, converted.copy(), valid.copy(), population[:, 7:].copy(),
                mask[:, 7:].copy())
            start = spec['feature_offset']
            values[start:start+36], masks[start:start+36] = (population*mask).ravel(), mask.ravel()
        if self.wheel_spec is not None:
            values[:18], masks[:18] = self._wheel_organs(fields, sample_time)
        if self.yaw_spec is not None and fields.get('localAngularVel') is not None:
            angular = np.asarray(fields['localAngularVel'], dtype=float)
            if angular.shape != (3,):
                raise ValueError('Three native angular velocity channels required')
            finite = np.isfinite(angular)
            safe = np.where(finite, angular, 0.)
            previous = self.scalar_history.get('localAngularVel')
            if previous is not None:
                stamp, old, old_valid = previous
                if sample_time < stamp:
                    raise ValueError('Host angular velocity sample moved backwards')
                if sample_time == stamp and (not np.array_equal(safe, old)
                                             or not np.array_equal(finite, old_valid)):
                    raise ValueError('Reused angular velocity sample changed')
            self.scalar_history['localAngularVel'] = (sample_time, safe.copy(), finite.copy())
            yaw = angular[self.yaw_spec['axis_index']]*self.yaw_spec['gain']
            if np.isfinite(yaw):
                values[126:128] = np.tanh(np.maximum([yaw, -yaw], 0)/self.yaw_spec['scale'])
                masks[126:128] = True
        elif self.yaw_spec is not None:
            self.scalar_history.pop('localAngularVel', None)
        return values, masks

    def _wheel_organs(self, fields, sample_time):
        key = 'wheelAngularSpeed'
        raw = fields.get(key)
        if raw is None:
            self.scalar_history.pop(key, None)
            return np.zeros(18), np.zeros(18, bool)
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (4,):
            raise ValueError('Four FL/FR/RL/RR native wheel rotation values required')
        valid = np.isfinite(raw)
        raw = np.where(valid, raw, 0.)
        pairs = np.asarray(self.wheel_spec['wheel_pairs'])
        magnitude = np.abs(raw[pairs]).mean(1)
        pair_valid = valid[pairs].all(1)
        population, mask = np.zeros((2, 9)), np.zeros((2, 9), bool)
        population[:, :7] = np.tanh(np.maximum(
            magnitude[:, None]-self.wheel_spec['centers'], 0.)/self.wheel_spec['width'])
        mask[:, :7] = pair_valid[:, None]
        previous = self.scalar_history.get(key)
        if previous is not None:
            stamp, old, old_valid, rates, rate_valid = previous
            if sample_time < stamp:
                raise ValueError('Host wheel sample moved backwards')
            if sample_time == stamp:
                if not np.array_equal(raw, old) or not np.array_equal(valid, old_valid):
                    raise ValueError('Reused wheel sample changed')
                population[:, 7:], mask[:, 7:] = rates, rate_valid
            elif sample_time-stamp <= self.max_age:
                rate = (magnitude-np.abs(old[pairs]).mean(1))/(sample_time-stamp)
                rate /= self.wheel_spec['change_scale']
                population[:, 7:] = np.tanh(np.c_[np.maximum(rate, 0), np.maximum(-rate, 0)])
                mask[:, 7:] = (pair_valid & old_valid[pairs].all(1))[:, None]
        self.scalar_history[key] = (sample_time, raw.copy(), valid.copy(),
                                   population[:, 7:].copy(), mask[:, 7:].copy())
        return (population*mask).ravel(), mask.ravel()

    def pack(self, rgb, *, capture_id, callback_time, decision_time,
             native_scalars=None, native_receipt=None):
        if (rgb.dtype != np.uint8 or rgb.shape != (*self.config['image_shape'], 3)
                or type(capture_id) is not int or capture_id < 0
                or not np.isfinite([callback_time, decision_time]).all()
                or callback_time > decision_time
                or self.last_decision is not None and decision_time <= self.last_decision):
            raise ValueError('Exact RGB, causal host order and increasing decisions required')
        identity = (capture_id, callback_time, hashlib.sha256(rgb.tobytes()).hexdigest())
        old = self.previous_frame
        if old is not None and (capture_id < old[0] or capture_id == old[0] and identity != old
                or capture_id > old[0] and callback_time <= old[1]):
            raise ValueError('Image capture identity changed or moved backwards')
        shape = (4, self.count)
        colors, contrast = np.zeros((*shape, 3)), np.zeros((*shape, 3))
        valid, rate_valid = np.zeros(shape, bool), np.zeros(shape, bool)
        ages = np.full(4, self.max_age*10)
        next_visual = self.visual_history.copy()
        for view, name in enumerate(VIEW_ORDER):
            region = self.config['regions'][name]
            if region is None:
                continue
            ages[view] = min(decision_time-callback_time, self.max_age*10)
            if ages[view] > self.max_age:
                next_visual.pop(name, None)
                continue
            value = self._sample(rgb, region)
            colors[view], valid[view] = value, True
            previous = next_visual.get(name)
            if previous is not None:
                prior_id, stamp, prior, derivative, derivative_valid = previous
                if prior_id != capture_id:
                    derivative = np.tanh((value-prior)/(callback_time-stamp)/self.scale)
                    derivative_valid = np.ones(self.count, bool)
                contrast[view], rate_valid[view] = derivative, derivative_valid
            next_visual[name] = (capture_id, callback_time, value.copy(),
                                 contrast[view].copy(), rate_valid[view].copy())
        old_scalars = self.scalar_history
        self.scalar_history = old_scalars.copy()
        try:
            organs, organ_valid = self._organs(native_scalars, native_receipt, decision_time)
        except Exception:
            self.scalar_history = old_scalars
            raise
        self.visual_history = next_visual
        self.previous_frame, self.last_decision = identity, decision_time
        return {k: np.asarray(v, dtype=np.float32) for k, v in dict(
            view_rgb=colors, view_contrast=contrast, view_valid=valid,
            contrast_valid=rate_valid, view_age=ages, organs=organs,
            organ_valid=organ_valid).items()}
