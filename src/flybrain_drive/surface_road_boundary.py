"""Road-surface transition detector for four calibrated simulator views."""

import cv2
import numpy as np

from .pixel_road_boundary import PixelRoadBoundary


class SurfaceContrastRoadBoundary(PixelRoadBoundary):
    """Compare same-view strips either side of a candidate surface transition."""

    def __init__(self, mount, *, front_far=False):
        super().__init__(mount)
        self.front_far = front_far
        self.stations = np.array([4., 8., 10., 12., 15., 20.])
        self.lookup = self._sampling(mount, self.stations)
        self.extended_stations = np.r_[self.stations, [24., 28., 32.]]
        self.extended_lookup = self._sampling(mount, self.extended_stations)

    def detect(self, images, *, extended=False):
        baseline = super().detect(images)
        stations, lookup = self.stations, self.lookup
        indices = (0, 2, 3, 4, 5, 6)
        if extended:
            baseline += [None] * 3
            stations, lookup = self.extended_stations, self.extended_lookup
            indices += (7, 8, 9)
        center = 0.
        projected = []
        for name, (u, v, valid) in lookup.items():
            image = cv2.cvtColor(images[name], cv2.COLOR_BGR2LAB)
            height, width = image.shape[:2]
            samples = image[np.rint(np.clip(v, 0, 1)*(height-1)).astype(int),
                            np.rint(np.clip(u, 0, 1)*(width-1)).astype(int)].astype(float)
            projected.append((name, samples, valid))
        candidates = np.arange(9, len(self.lateral)-9)
        for station, (distance, section_index) in enumerate(zip(stations, indices, strict=True)):
            offset = np.arange(1, 5) if distance == 4. else np.arange(2, 9)
            boundaries = []
            for side in (-1, 1):
                inside = candidates[:, None]-side*offset
                outside = candidates[:, None]+side*offset
                supported = np.zeros(len(candidates), dtype=bool)
                for name, samples, valid in projected:
                    if self.front_far and distance >= 12. and name != "front":
                        continue
                    row = samples[station]
                    inner, outer = row[inside], row[outside]
                    a, b = np.median(inner, axis=1), np.median(outer, axis=1)
                    difference = b-a
                    chroma = np.linalg.norm(difference[:, 1:], axis=1)
                    inner_chroma = np.linalg.norm(a[:, 1:]-128., axis=1)
                    outer_chroma = np.linalg.norm(b[:, 1:]-128., axis=1)
                    support = valid[station, inside].all(1) & valid[station, outside].all(1)
                    eligible = support & (inner_chroma <= 13) & (outer_chroma-inner_chroma >= 6)
                    eligible &= (chroma >= 9) | ((chroma >= 6) & (abs(difference[:, 0]) >= 16))
                    eligible &= (self.lateral[candidates]-center)*side >= 1.3
                    supported |= eligible
                # Select the nearest transition supported at three lateral samples.
                contiguous = np.convolve(supported.astype(int), np.ones(3), mode="same") == 3
                indices = candidates[contiguous]
                edge = float(self.lateral[indices[0 if side > 0 else -1]]) if len(indices) else None
                boundaries.append(edge)
            right, left = boundaries
            if distance == 4.:
                original = baseline[0]
                right = original["right_y_m"] if original["right_y_m"] is not None else right
                left = original["left_y_m"] if original["left_y_m"] is not None else left
            valid = right is not None and left is not None and 3. < left-right < 20.
            if valid and distance > 4.:
                center = .5*(left+right)
            baseline[section_index] = dict(forward_x_m=float(distance), right_y_m=right, left_y_m=left, valid=valid)
        return baseline

    def line_sections(self, images):
        """Return the stations used by memory and the directly observed fallback."""
        detected = self.detect(images)
        observed = [section for section in detected if section["forward_x_m"] not in (6., 10.)]
        if observed[0]["valid"] or not detected[1]["valid"] or not observed[1]["valid"]:
            return observed, observed
        grass, visible = self._classify(images, self.near_lut)
        near = self._sections(self.near, grass, visible, near=True)[1]
        fallback = ([near] if near["valid"] else []) + [detected[1]] + observed[1:]
        return observed, fallback
