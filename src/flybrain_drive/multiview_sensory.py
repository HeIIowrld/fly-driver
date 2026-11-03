"""Independent source-local retinal samples; no panoramic stitching or cross-clock subtraction."""

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

from .compound_eye import CompoundEyeProjector, PerspectiveView
from .sensor_contract import ActorObservation, VisualViewSample, readonly_array


@dataclass(frozen=True)
class ViewMap:
    source_id: str
    directions_body: np.ndarray
    acceptance_sigma_degrees: np.ndarray
    perspective: PerspectiveView
    provenance: str

    def __post_init__(self):
        # Reuse the public sample's source-name validator without constructing a measurement.
        import re

        directions = readonly_array(self.directions_body)
        sigma = readonly_array(self.acceptance_sigma_degrees)
        if (not isinstance(self.source_id, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,47}", self.source_id) is None
                or not isinstance(self.perspective, PerspectiveView)
                or not isinstance(self.provenance, str) or not self.provenance.strip()
                or directions.ndim != 2 or directions.shape[1] != 3 or not len(directions)
                or not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-7)
                or sigma.shape != (len(directions),) or np.any((sigma <= 0) | (sigma > 15))):
            raise ValueError("Source, unit rays, acceptance and perspective calibration required")
        object.__setattr__(self, "directions_body", directions)
        object.__setattr__(self, "acceptance_sigma_degrees", sigma)

    @property
    def origin_body_m(self):
        return self.perspective.origin_body_m

    def manifest(self):
        return {"source_id": self.source_id, "directions_body": self.directions_body.tolist(),
                "acceptance_sigma_degrees": self.acceptance_sigma_degrees.tolist(),
                "origin_body_m": self.origin_body_m.tolist(),
                "camera_to_body": self.perspective.camera_to_body.tolist(),
                "fovy_degrees": self.perspective.fovy_degrees,
                "projection_model": "direct_perspective_not_uncalibrated_reflection",
                "provenance": self.provenance}

    @cached_property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.manifest(), sort_keys=True,
                                         separators=(",", ":")).encode("utf8")).hexdigest()

    def projector(self):
        return CompoundEyeProjector(self, self.perspective)

    @classmethod
    def from_manifest(cls, value):
        if value.get("projection_model") != "direct_perspective_not_uncalibrated_reflection":
            raise ValueError("This view-map loader requires calibrated direct perspective sources")
        perspective = PerspectiveView(value["camera_to_body"], value["origin_body_m"],
                                      value["fovy_degrees"])
        return cls(value["source_id"], value["directions_body"], value["acceptance_sigma_degrees"],
                   perspective, value["provenance"])

    def project(self, image, *, frame_id, time, provenance, pixel_valid=None):
        response, valid, coverage = self.projector().sample_pixels(image, pixel_valid=pixel_valid)
        return VisualViewSample(self.source_id, frame_id, response, valid, time,
                                self.sha256, provenance), coverage


class MultiViewPacker:
    """Per-view RGB/temporal contrast and age; body state remains independent.

    Temporal contrast is a normalized RGB derivative, not calibrated optic flow.
    Reused exposure IDs hold their previous derivative. Missing/stale gaps clear
    derivative history, while exposure identity tombstones prevent ID rollback.
    """

    fields = {"view_rgb", "view_valid", "view_contrast", "contrast_valid", "view_age",
              "organs", "organ_valid"}

    def __init__(self, views, organs, *, max_view_age=.15, contrast_scale=10):
        self.views = tuple(views)
        if (not self.views or len({v.source_id for v in self.views}) != len(self.views)
                or not np.isfinite([max_view_age, contrast_scale]).all()
                or min(max_view_age, contrast_scale) <= 0):
            raise ValueError("Unique view maps and positive age/contrast calibration required")
        self.organs, self.max_view_age, self.contrast_scale = organs, max_view_age, contrast_scale
        self.reset()

    @property
    def calibration(self):
        organ_hash = self.organs.calibration_sha256
        if organ_hash is None:
            configuration = {"positions": [asdict(s) for s in self.organs.positions],
                             "rotation_scale": self.organs.rotation_scale,
                             "max_sample_age": self.organs.max_sample_age}
            organ_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True,
                                                   separators=(",", ":")).encode()).hexdigest()
        return {"view_order": [v.source_id for v in self.views],
                "view_hashes": {v.source_id: v.sha256 for v in self.views},
                "organ_size": self.organs.feature_count,
                "organ_calibration_sha256": organ_hash,
                "max_view_age": self.max_view_age, "contrast_scale": self.contrast_scale}

    def reset(self):
        self.episode = None
        self.identities, self.history = {}, {}
        self.organs.reset()

    def pack(self, observation):
        if not isinstance(observation, ActorObservation) or observation.eyes:
            raise ValueError("Multi-view packer requires source-specific ActorObservation")
        new_episode = observation.episode_id != self.episode
        identities = {} if new_episode else self.identities.copy()
        history = {} if new_episode else self.history
        samples = {v.source_id: v for v in observation.views}
        if set(samples) - {v.source_id for v in self.views}:
            raise ValueError("Unknown/unconfigured view source")
        # Validate every source before changing either visual or body history.
        for view in self.views:
            sample = samples.get(view.source_id)
            if sample is None:
                continue
            if (sample.calibration_sha256 != view.sha256
                    or sample.response.shape != (len(view.directions_body), 3)):
                raise ValueError("View response or calibration differs from declared source")
            previous = identities.get(view.source_id)
            if previous is not None:
                same = previous.frame_id == sample.frame_id
                if sample.frame_id < previous.frame_id:
                    raise ValueError("View exposure ID moved backwards, including after gaps")
                if same and (sample.time.simulation_seconds != previous.time.simulation_seconds
                             or not np.array_equal(sample.response, previous.response)
                             or not np.array_equal(sample.valid, previous.valid)):
                    raise ValueError("Reused view exposure changed timing/values/validity")
                if not same and sample.time.simulation_seconds <= previous.time.simulation_seconds:
                    raise ValueError("New view exposure requires increasing simulation timestamp")
            identities[view.source_id] = sample
        shape = (len(self.views), max(len(v.directions_body) for v in self.views))
        rgb, contrast = np.zeros((*shape, 3)), np.zeros((*shape, 3))
        valid, rate_valid = np.zeros(shape, bool), np.zeros(shape, bool)
        ages, next_history = np.full(len(self.views), self.max_view_age * 10), {}
        for i, view in enumerate(self.views):
            sample = samples.get(view.source_id)
            if sample is None:
                continue
            age = observation.simulation_seconds - sample.time.simulation_seconds
            ages[i] = min(age, self.max_view_age * 10)
            if age > self.max_view_age:
                continue
            n = len(sample.response)
            rgb[i, :n], valid[i, :n] = sample.response * sample.valid[:, None], sample.valid
            previous = history.get(view.source_id)
            derivative, mask = np.zeros((n, 3)), np.zeros(n, bool)
            if previous is not None:
                old, old_derivative, old_mask = previous
                if old.frame_id == sample.frame_id:
                    derivative, mask = old_derivative, old_mask
                else:
                    dt = sample.time.simulation_seconds - old.time.simulation_seconds
                    with np.errstate(over="ignore"):
                        derivative = np.tanh((sample.response - old.response)
                                             / dt / self.contrast_scale)
                    mask = sample.valid & old.valid
            contrast[i, :n] = derivative * mask[:, None]
            rate_valid[i, :n] = mask
            next_history[view.source_id] = (sample, derivative.copy(), mask.copy())
        encoded = self.organs.encode(observation)
        self.identities, self.history = identities, next_history
        self.episode = observation.episode_id
        return {key: np.asarray(value, dtype=np.float32) for key, value in {
            "view_rgb": rgb, "view_valid": valid, "view_contrast": contrast,
            "contrast_valid": rate_valid, "view_age": ages,
            "organs": encoded.values, "organ_valid": encoded.valid}.items()}
