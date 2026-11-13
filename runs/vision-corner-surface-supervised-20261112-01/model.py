"""Compact front-video boundary model with metric and image-space supervision."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CROP = (72, 152)
SIZE = (160, 40)
STATIONS = np.arange(10., 35., 2.)
CENTER_Y = 112.
ROW_SCALE = 20.
LATERAL_SCALE = 12.
SUPPORT_CUT = .5


def block(in_channels, out_channels, stride):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.SiLU())


class BoundaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = nn.Sequential(block(3, 24, 2), block(24, 32, 1), block(32, 32, 2))
        self.high = nn.Sequential(block(32, 48, 2), block(48, 48, 1))
        self.metric = nn.Sequential(nn.Flatten(), nn.Linear(48 * 5 * 20, 128), nn.SiLU(),
                                    nn.Dropout(.20), nn.Linear(128, len(STATIONS)*2*3))
        self.image = nn.Sequential(block(32, 24, 1), nn.Conv2d(24, 2, 1))

    def forward(self, image):
        early = self.low(image)
        metric = self.metric(self.high(early)).reshape(-1, len(STATIONS), 2, 3)
        image = F.interpolate(self.image(early), size=(SIZE[1],SIZE[0]),
                              mode="bilinear", align_corners=False)
        return metric, image


def camera_mask(pixels, witness):
    valid = witness & np.isfinite(pixels).all(axis=-1)
    return valid & (pixels[..., 1] >= CROP[0]+1) & (pixels[..., 1] < CROP[1]-1)


def pixel_targets(pixels, valid):
    marks = np.zeros((len(pixels), 2, SIZE[1], SIZE[0]), dtype=np.uint8)
    weights = np.zeros_like(marks)
    for i, (frame, okay) in enumerate(zip(pixels, valid, strict=True)):
        for j in range(2):
            trace = marks[i, j]
            positions = np.flatnonzero(okay[:, j])
            if not len(positions):
                continue
            xy = frame[positions, j].copy()
            xy[:, 1] -= CROP[0]
            xy = np.rint((xy + .5)/2. - .5).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, SIZE[0]-1)
            xy[:, 1] = np.clip(xy[:, 1], 0, SIZE[1]-1)
            for x, y in xy:
                trace[y, x] = 1
            for k in range(1, len(positions)):
                if positions[k] == positions[k-1]+1 and np.linalg.norm(xy[k]-xy[k-1]) <= 22.:
                    cv2.line(trace, tuple(xy[k-1]), tuple(xy[k]), 1, 1)
            distance = cv2.distanceTransform(1 - trace, cv2.DIST_L2, 3)
            trace[:] = np.rint(255. * np.exp(-.5 * (distance/1.1)**2)).astype(np.uint8)
            weights[i, j] = (distance <= 11.).astype(np.uint8)
    return marks, weights


def input_image(image, device):
    return image.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.


def predict(metric):
    lateral = metric[..., 0] * LATERAL_SCALE
    rows = metric[..., 1] * ROW_SCALE + CENTER_Y
    support = torch.sigmoid(metric[..., 2])
    return lateral, rows, support


def loss(output, image, metric_target, valid, pixel_target, pixel_weight):
    support = F.binary_cross_entropy_with_logits(output[..., 2], valid.float())
    delta = F.smooth_l1_loss(output[..., :2], metric_target, reduction="none", beta=.1)
    metric = (delta[..., 0] * valid).sum() / valid.sum().clamp_min(1.)
    vertical = (delta[..., 1] * valid).sum() / valid.sum().clamp_min(1.)
    error = F.binary_cross_entropy_with_logits(image, pixel_target, reduction="none")
    weight = pixel_weight * (1. + 3.*pixel_target)
    dense = (weight * error).sum() / weight.sum().clamp_min(1.)
    return metric + .45*vertical + .15*support + .20*dense
