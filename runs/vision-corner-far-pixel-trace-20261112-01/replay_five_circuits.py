"""Transfer the already evaluated perturbation rule across the five circuit folds."""

from pathlib import Path

import cv2

from perturbations import OUT, MOUNT, context, run
from train_vision_teacher_line import digest, read, write


def main():
    cv2.setNumThreads(1)
    entry=context()
    replay=entry.import_replay()
    control=OUT/"perturbation-result.json"
    reviewed=read(control)
    assert reviewed["shifts_px"]==[-2.,2.] and reviewed["distance_tolerance_m"]==3.
    for name,sha in reviewed["source_sha256"].items():
        assert digest(Path(name))==sha
    hashes={Path(__file__),control,MOUNT,Path(run.__code__.co_filename)}
    results={}
    for track in replay.profile.CALIBRATION_TRACKS:
        results[track]=run(track,entry,replay,hashes)
        print(track,{k:v for k,v in results[track]["metrics"].items() if k.startswith("t4t5")},flush=True)
    write(OUT/"transfer-result.json",dict(status="evaluated",seed=None,posthoc=True,tracks=results,
        shifts_px=reviewed["shifts_px"],distance_tolerance_m=reviewed["distance_tolerance_m"],
        source_sha256={str(p):digest(p) for p in sorted(hashes)},termination="five_held_out_timelines_scored"))


if __name__=="__main__":
    main()
