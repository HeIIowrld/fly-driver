"""Opt-in CSP retina using source-clock render intervals, with host age bounds."""

import numpy as np

from .ac_csp_retina import VIEW_ORDER, CSPCameraRetina


class CSPProducerTimeRetina(CSPCameraRetina):
    """Keep legacy freshness checks; use same-generation source time for temporal contrast.

    The producer render-start clock is not a physical exposure timestamp. Callers must
    preserve it with each original frame and may not infer it from host-clock bounds.
    """

    source_field = 'producer_render_started'

    def __init__(self, config):
        super().__init__(config)
        self.source_history = {}
        self.source_identities = {}

    def reset_episode(self):
        """Reset source-clock history together with the base sensor state."""
        super().reset_episode()
        self.source_history.clear()
        self.source_identities.clear()

    @property
    def calibration(self):
        return {**super().calibration, 'temporal_clock': 'same_generation_producer_render_start',
                'physical_exposure_sync_verified': False}

    def pack(self, samples, *, decision_time, native_scalars=None, native_receipt=None):
        identities = self.source_identities.copy()
        stripped = {}
        for name, sample in samples.items():
            if sample is None:
                stripped[name] = None
                continue
            if name not in VIEW_ORDER or set(sample) != self.sample_fields | {self.source_field}:
                raise ValueError('Producer time and original causal camera fields required')
            source = sample[self.source_field]
            if (not isinstance(source, (float, int, np.floating))
                    or not np.isfinite(source) or source < 0):
                raise ValueError('Finite nonnegative producer render clock required')
            capture, generation = sample['capture_id'], sample['generation']
            old = identities.get(name)
            if old is not None and (generation != old[0] or capture < old[1]
                    or capture == old[1] and source != old[2]
                    or capture > old[1] and source <= old[2]):
                raise ValueError('Producer frame identity or source time changed')
            identities[name] = (generation, capture, float(source))
            stripped[name] = {key: value for key, value in sample.items()
                              if key != self.source_field}
        result = super().pack(stripped, decision_time=decision_time,
                              native_scalars=native_scalars, native_receipt=native_receipt)
        history = self.source_history.copy()
        contrast = np.zeros_like(result['view_contrast'])
        valid = np.zeros_like(result['contrast_valid'])
        for index, name in enumerate(VIEW_ORDER):
            sample = samples.get(name)
            if sample is None or not result['view_valid'][index].all():
                history.pop(name, None)
                continue
            capture, source = sample['capture_id'], float(sample[self.source_field])
            value = result['view_rgb'][index]
            prior = history.get(name)
            if prior is not None:
                old_id, old_time, old_value, old_rate, old_valid = prior
                if capture == old_id:
                    contrast[index], valid[index] = old_rate, old_valid
                elif source - old_time <= self.max_age:
                    contrast[index] = np.tanh((value-old_value)/(source-old_time)/self.body.scale)
                    valid[index] = 1
            history[name] = (capture, source, value.copy(), contrast[index].copy(),
                             valid[index].copy())
        self.source_history, self.source_identities = history, identities
        result['view_contrast'], result['contrast_valid'] = contrast, valid
        return result
