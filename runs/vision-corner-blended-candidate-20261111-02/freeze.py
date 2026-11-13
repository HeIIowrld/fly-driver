"""Freeze the blended T4/T5 direction head and replay normal inference."""
from pathlib import Path

import numpy as np

from infer_frozen_next_corner import infer, load_readout
from train_vision_teacher_line import digest, read, write

out=Path(__file__).parent
root=out.parent
base=root/"vision-frozen-next-corner-20261110-01"
study=root/"vision-corner-cross-track-direction-20261111-03"
pins=read(base/"pins.json")
weights=study/"ridge-100-gain-0.5.npz"
pins["models"]["direction"]=dict(path=str(weights),sha256=digest(weights))
pins["development_selection"]=dict(source=str(study/"result.json"),sha256=digest(study/"result.json"),
    ridge=100.,learned_fraction=.5,independent_evaluation_used=False,
    tradeoff="Magione outside alerts increase 26 to 27 in teacher-label development scoring",
    roles=read(study/"result.json")["roles"])
write(out/"pins.json",pins)
archive=base/"archive-parity"
infer(out/"pins.json",archive/"index.jsonl",archive/"original/features.npz",
      archive/"reflected/features.npz",out/"ordinal")
infer(out/"pins.json",archive/"index.jsonl",archive/"original/features.npz",
      archive/"reflected/features.npz",out/"joint-distance",
      joint_distance_pins=root/"vision-magione-joint-distance-fusion-20261111-01/joint-pins.json")
# Check that the model loader preserves the fitted head's linear arithmetic.
predict,_=load_readout(pins)
with np.load(archive/"original/features.npz") as a,np.load(archive/"reflected/features.npz") as b:
    direct,mirror=a["features"],b["features"]
    score,_,_=predict(direct,mirror,return_scores=True)
assert np.isfinite(score).all()
write(out/"result.json",dict(status="frozen_development_blended_candidate",seed=None,
      revision=pins["revision"],pins_sha256=digest(out/"pins.json"),
      inference_sha256=digest(Path("scripts/infer_frozen_next_corner.py")),
      freezing_script_sha256=digest(Path(__file__)),
      ordinal_result_sha256=digest(out/"ordinal/result.json"),
      joint_result_sha256=digest(out/"joint-distance/result.json"),
      termination="normal inference replay saved"))
