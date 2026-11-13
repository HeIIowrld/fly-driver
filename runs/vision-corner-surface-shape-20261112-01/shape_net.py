"""Learn continuous boundary shape and a physical onset from the same front image."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "vision-corner-surface-supervised-20261112-01"
sys.path.insert(0, str(PREVIOUS))
from model import BoundaryNet, STATIONS, loss as point_loss


class ShapeNet(BoundaryNet):
    def __init__(self):
        super().__init__()
        self.onset = nn.Sequential(nn.Flatten(), nn.Linear(48*5*20, 64),
                                   nn.SiLU(), nn.Dropout(.20), nn.Linear(64, 4))

    def forward(self, image):
        early = self.low(image)
        features = self.high(early)
        sections = self.metric(features).reshape(-1, len(STATIONS), 2, 3)
        heat = F.interpolate(self.image(early), size=(40, 160), mode="bilinear", align_corners=False)
        return sections, heat, self.onset(features)


def masked_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1.)


def shape_loss(sections, target, valid):
    predicted, actual = sections[..., 0], target[..., 0]
    slope_valid = valid[:, 1:] & valid[:, :-1]
    bend_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    slope = F.smooth_l1_loss(predicted.diff(dim=1), actual.diff(dim=1), beta=.05, reduction="none")
    bend = F.smooth_l1_loss(predicted.diff(n=2, dim=1), actual.diff(n=2, dim=1), beta=.025, reduction="none")
    return masked_mean(slope, slope_valid), masked_mean(bend, bend_valid)


def objective(sections, heat, onset, metric, valid, pixels, weights, direction, distance, class_weights):
    basic = point_loss(sections, heat, metric, valid, pixels, weights)
    slope, bend = shape_loss(sections, metric, valid)
    classified = direction >= 0
    classification = F.cross_entropy(onset[:, :3], direction.clamp_min(0),
                                     weight=class_weights, reduction="none")
    distance_error = F.smooth_l1_loss(torch.sigmoid(onset[:, 3]), distance, beta=.15, reduction="none")
    return basic + .5*slope + bend + .25*masked_mean(classification, classified) + .15*masked_mean(distance_error, direction > 0)
