"""Trace two visible boundary curves in the front image before metric sampling."""
import cv2
import numpy as np

MIN_CONTRAST = 9.
MAX_STEP = 16
MAX_GAP = 2


def image_evidence(image, side):
    """Contrast along horizontal scans, with inward/outward Lab context."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    smoothed = cv2.blur(lab, (3, 1))
    inside = np.roll(smoothed, 3 * side, axis=1)
    outside = np.roll(smoothed, -3 * side, axis=1)
    difference = outside - inside
    contrast = np.linalg.norm(difference[:, :, 1:], axis=2) + .35 * np.abs(difference[:, :, 0])
    inside_chroma = np.linalg.norm(inside[:, :, 1:] - 128., axis=2)
    outside_chroma = np.linalg.norm(outside[:, :, 1:] - 128., axis=2)
    contrast += .25 * np.clip(outside_chroma - inside_chroma, -8., 16.)
    contrast[:, :5] = 0
    contrast[:, -5:] = 0
    return contrast


def trace(evidence, anchor, last_row, *, motion_prior=True):
    """Find a connected path with bounded displacement, returning supported rows only."""
    anchor_x, anchor_y = anchor
    start = int(round(anchor_y))
    width = evidence.shape[1]
    if not (0 <= last_row <= start < len(evidence) and 5 <= anchor_x < width - 5):
        return np.full(len(evidence), np.nan)
    columns = np.arange(width)
    offsets = np.arange(-MAX_STEP, MAX_STEP + 1)
    predecessors = columns[None, :] - offsets[:, None]
    valid = (predecessors >= 5) & (predecessors < width - 5)
    indices = predecessors.clip(0, width - 1)
    scores = np.where(abs(columns - anchor_x) <= 8,
                      np.minimum(evidence[start], 35.) - 1.2 * abs(columns - anchor_x), -np.inf)
    back = []
    for row in range(start - 1, last_row - 1, -1):
        expected = (.5 * (width - 1) - columns) / max(8, row + 1 - .5 * (len(evidence) - 1))
        prior = expected if motion_prior else 0.
        costs = .60 * np.abs(offsets[:, None] - prior) + .10 * abs(offsets[:, None])
        transitions = np.where(valid, scores[indices] - costs, -np.inf)
        best = np.argmax(transitions, axis=0)
        scores = transitions[best, columns] + np.clip(evidence[row] - MIN_CONTRAST, -7., 25.)
        back.append(indices[best, columns])
    output = np.full(len(evidence), np.nan)
    column = int(np.argmax(scores))
    pixels = [column]
    for choice in reversed(back):
        column = int(choice[column])
        pixels.append(column)
    gap = 0
    for row, column in reversed(list(zip(range(last_row, start + 1), pixels, strict=True))):
        if evidence[row, column] < MIN_CONTRAST:
            gap += 1
            if gap > MAX_GAP:
                break
            continue
        gap = 0
        left, center, right = evidence[row, column - 1:column + 2]
        denominator = left - 2. * center + right
        offset = np.clip(.5 * (left - right) / denominator, -.5, .5) if denominator < -1e-5 else 0.
        output[row] = column + offset
    return output


def at_rows(curve, rows):
    """Interpolate supported adjacent rows, without extrapolating through gaps."""
    rows = np.asarray(rows, dtype=float)
    finite = np.isfinite(rows)
    safe = np.where(finite, rows, 0.)
    below = np.floor(safe).astype(int)
    above = np.ceil(safe).astype(int)
    valid = finite & (below >= 0) & (above < len(curve))
    a = curve[below.clip(0, len(curve) - 1)]
    b = curve[above.clip(0, len(curve) - 1)]
    return np.where(valid, a + (safe - below) * (b - a), np.nan)


class FrontTrace:
    """Project a tracked visible edge through the fixed front-camera plane."""
    def __init__(self, front, stations, shape):
        if not (np.allclose(front["forward_body"], [1, 0, 0])
                and np.allclose(front["right_body"], [0, -1, 0])
                and np.allclose(front["up_body"], [0, 0, 1])):
            raise ValueError("Front row tracing requires the verified level front mount")
        self.height, self.width = shape
        self.origin = np.asarray(front["origin_body_m"], dtype=float)
        depth = stations - self.origin[0]
        self.scale = .5 * front["projection_x"] * (self.width - 1) / np.where(depth > 0, depth, np.nan)
        self.rows = (.5 + .5 * front["projection_y"] * self.origin[2]
                     / np.where(depth > 0, depth, np.nan)) * (self.height - 1)

    def curves(self, image, boundaries):
        if image.shape[:2] != (self.height, self.width):
            raise ValueError("Unexpected front camera shape")
        result = np.full((self.height, 2), np.nan)
        for index, side in enumerate((1, -1)):  # image right, image left
            lateral = boundaries[:, index]
            x = .5 * (self.width - 1) - self.scale * (lateral - self.origin[1])
            anchors = np.flatnonzero(np.isfinite(x[:7]) & (x[:7] >= 9.) & (x[:7] <= self.width - 10.)
                                     & (self.rows[:7] < self.height - 4))
            if len(anchors):
                i = int(anchors[0])
                result[:, index] = trace(image_evidence(image, side), (x[i], self.rows[i]),
                                         int(np.floor(self.rows[-1])) - 3)
        return result

    def project(self, curves, rows=None):
        sample_rows = np.repeat(self.rows[:, None], 2, axis=1) if rows is None else rows
        pixels = np.stack([at_rows(curve, row) for curve, row in zip(curves.T, sample_rows.T, strict=True)], axis=1)
        return self.origin[1] + (.5 * (self.width - 1) - pixels) / self.scale[:, None]
