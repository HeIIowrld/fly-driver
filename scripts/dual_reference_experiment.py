"""Same current actor/Adam with an additional TRAIN-only parent loss reference."""
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from scripts.direct_scene_gain_experiment import DirectSceneGainExperiment

class DualReferenceExperiment(DirectSceneGainExperiment):
 def __init__(self,plan):
  super().__init__(plan,learn_direct=False)
  path=Path(plan['direct_parent_evaluation'])/'predictions.npz'
  if sha256(path)!=plan['parent_reference_predictions_sha256']:raise ValueError('Exact immediate-parent reference required')
  with np.load(path) as z:self.parent_anchors={ep:z[f'prediction_{ep}'].copy() for ep in [0,1]}
  if any(self.parent_anchors[ep].shape!=self.anchors[ep].shape for ep in [0,1]):raise ValueError('Aligned TRAIN parent reference required')
