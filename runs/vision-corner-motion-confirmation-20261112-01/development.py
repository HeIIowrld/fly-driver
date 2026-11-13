"""Replay the image-normal detector on the development circuits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

import image_normal
from evaluate_fused_corner import load_frozen
from profile_replay import CALIBRATION_TRACKS
from train_vision_teacher_line import digest,read,write

OUT=Path(__file__).parent


def main():
    cv2.setNumThreads(1)
    image_normal.checks()
    image_normal.OUT=OUT
    mount_path=Path(load_frozen(image_normal.PROTOCOL.with_name("candidate.json"))["camera_config"])
    mount=read(mount_path)

    def replay(track):
        hashes={Path(__file__),Path(image_normal.__file__),mount_path,
                Path(image_normal.baseline_data.__code__.co_filename),
                Path(image_normal.SurfaceContrast.detect.__code__.co_filename)}
        report=image_normal.run_track(track,mount,hashes)
        write(OUT/f"{track}-development.json",dict(status="evaluated",seed=None,track=track,
                    source_sha256={str(p):digest(p) for p in sorted(hashes)},**report))
        print(track,{name:value["score"]["metrics"]["candidate"]
                     for name,value in report["methods"].items()},flush=True)
        return track,digest(OUT/f"{track}-development.json")

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts=dict(executor.map(replay,CALIBRATION_TRACKS))
    write(OUT/"development-receipts.json",dict(status="evaluated",tracks=receipts,seed=None))


if __name__=="__main__":
    main()
