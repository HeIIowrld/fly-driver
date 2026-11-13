"""Expose a joint-hit probability from a categorical confirmation model."""
import numpy as np


class JointProbability:
    def __init__(self, estimator):
        self.estimator = estimator
        self.index = estimator.classes_.tolist().index("joint_hit")

    def predict_proba(self, features):
        joint = self.estimator.predict_proba(features)[:, self.index]
        return np.column_stack((1.-joint,joint))
