"""TRAIN-only calibration of readout values; original modeled neuron states stay unchanged."""

import numpy as np


class ChannelMoments:
    def __init__(self, channels=14):
        self.channels = channels
        self.count = 0
        self.total = np.zeros(channels, float)
        self.square = np.zeros(channels, float)

    def add(self, activity, *, role):
        if role != 'train':
            raise ValueError('Only preassigned TRAIN may calibrate the readout')
        if (activity.ndim != 4 or activity.shape[1:] != (4, self.channels, 721)
                or not np.isfinite(activity).all()):
            raise ValueError('Complete finite view/channel/retinal activity required')
        for start in range(0, len(activity), 64):
            x = activity[start:start+64].astype(float)
            self.total += x.sum(axis=(0, 1, 3))
            self.square += (x*x).sum(axis=(0, 1, 3))
            self.count += x.shape[0]*4*721

    def finish(self, floor):
        if not self.count or not np.isfinite(floor) or floor <= 0:
            raise ValueError('TRAIN observations and positive finite readout floor required')
        mean = self.total/self.count
        scale = np.sqrt(np.maximum(self.square/self.count-mean**2, 0)).clip(floor)
        return mean, scale
