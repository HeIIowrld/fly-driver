"""Connect image-supported road-edge points using the projected boundary tangent."""

import cv2
import numpy as np


def front_normals(image, lookup):
    """Sample Lab gradient directions at the same projected grid as the edge evidence."""
    smooth = cv2.GaussianBlur(image.astype(np.float32), (3, 3), .7)
    h, w = image.shape[:2]
    u, v, visible = lookup
    x, y = (u * (w - 1)).astype(np.float32), (v * (h - 1)).astype(np.float32)
    dx, dy = (cv2.remap(cv2.Sobel(smooth, cv2.CV_32F, a, b, ksize=3, scale=.125),
                        x, y, cv2.INTER_LINEAR) for a, b in ((1, 0), (0, 1)))
    channel = (dx * dx + dy * dy).argmax(axis=-1)
    i, j = np.indices(channel.shape)
    gradient = np.stack((dx[i, j, channel], dy[i, j, channel]), axis=-1)
    magnitude = np.linalg.norm(gradient, axis=-1)
    normal = gradient / np.maximum(magnitude[..., None], 1e-6)
    normal[~visible | (magnitude < 3.)] = 0.
    return np.stack((x, y), axis=-1), normal


def track_edge(raw, evidence, grid, stations, xy, normals, side, tangent_weight):
    """Fit observed segments only, with existing observations as fixed anchors."""
    result = raw[:, side].copy()
    sign = 2 * side - 1
    chain = []

    def commit():
        if not chain:
            return
        idx = int(np.argmax(chain[-1][2]))
        for station, cells, _, previous in reversed(chain):
            result[station] = grid[cells[idx]]
            if previous is None:
                break
            idx = previous[idx]
        chain.clear()

    for i, row in enumerate(raw):
        other = row[1 - side]
        if np.isfinite(row[side]):
            cells = np.array([int(np.argmin(abs(grid - row[side])))])
            emissions = np.zeros(1)
        elif np.isfinite(other):
            width = (grid - other) * sign
            supported = np.isfinite(evidence[i])
            supported &= np.r_[False, supported[:-1]] | np.r_[supported[1:], False]
            cells = np.flatnonzero(supported & (width > 3.) & (width < 20.))
            if not len(cells):
                continue
            emissions = np.maximum(-3., (evidence[i, cells] - evidence[i, cells].max()) / 12.)
        else:
            continue
        prev = None
        if chain:
            j, old, scores, _ = chain[-1]
            gap = stations[i] - stations[j]
            offset = abs(grid[cells][None, :] - grid[old][:, None])
            vector = xy[i, cells][None, :, :] - xy[j, old][:, None, :]
            length = np.linalg.norm(vector, axis=-1)
            unit = vector / np.maximum(length[..., None], 1e-6)
            disagreement = .5 * (
                abs(np.sum(unit * normals[j, old][:, None, :], axis=-1))
                + abs(np.sum(unit * normals[i, cells][None, :, :], axis=-1)))
            cost = .45 * offset + tangent_weight * disagreement
            old_other = raw[j, 1 - side]
            if np.isfinite(old_other) and np.isfinite(other):
                width_step = (grid[cells][None, :] - other) - (grid[old][:, None] - old_other)
                cost += .35 * abs(width_step)
            transitions = np.where(offset <= 1.3 * gap, scores[:, None] - cost, -np.inf)
            if gap > 4. or not np.isfinite(transitions).any():
                commit()
            else:
                prev = transitions.argmax(axis=0)
                emissions = emissions + transitions[prev, np.arange(len(cells))]
        chain.append((i, cells, emissions, prev))
    commit()
    # Preserve the original float and missingness semantics of measured cells.
    known = np.isfinite(raw[:, side])
    result[known] = raw[known, side]
    return result


def complete(raw, evidence, grid, stations, xy, normals, tangent_weight):
    edges = np.column_stack([track_edge(raw, evidence, grid, stations, xy, normals, side, tangent_weight)
                             for side in range(2)])
    width = edges[:, 1] - edges[:, 0]
    return np.where(((width > 3.) & (width < 20.))[:, None], edges, np.nan)
