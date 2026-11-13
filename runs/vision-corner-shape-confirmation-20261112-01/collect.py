"""Generate the frozen shape candidate on circuits excluded from its training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
SHAPE = OUT.parent / "vision-corner-surface-shape-20261112-01"
sys.path.insert(0, str(SHAPE))
from evaluate_shape import combined, inference, old
from prepare import preprocess
from replay_learned import baseline_data, frame_key

DEVELOP = ("imola", "ks_laguna_seca", "ks_zandvoort", "mugello")
# The shape checkpoint was fixed before these archives were processed.


def crop_track(track, group, mapping, bundles):
    keys = [frame_key(row) for row in group["rows"]]
    expected = {key:i for i,key in enumerate(keys)}
    images = np.zeros((len(keys),40,160,3),np.uint8)
    available = np.zeros(len(keys),bool)
    seen = set()
    for tag,folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[tag]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key = tag,item["capture_id"]
                if key not in expected:
                    continue
                assert key not in seen
                seen.add(key)
                i = expected[key]
                meta = metadata(archive,item,track)
                views = admitted(item,meta)
                if views:
                    images[i] = preprocess(image(archive,views["front"]))
                    available[i] = True
    assert seen == set(keys)
    path = OUT/f"{track}-front.npz"
    np.savez_compressed(path,images=images,available=available,
                        run=np.array([k[0] for k in keys]),frame=np.array([k[1] for k in keys]))
    return path


def metadata(archive,item,track):
    name = Path(item["file"]).name
    assert archive.by_name[name]["sha256"] == item["file_sha256"]
    meta = archive.json(name)
    assert (meta["track_id"],meta["index"],meta["generation"]) == (track,item["capture_id"],item["generation"])
    return meta


def admitted(item,meta):
    views = {v["view_id"]:v for v in item["views"]}
    if (set(views) != {"front","rear","left","right"} or len(item["views"]) != 4
        or any(r.get(flag) for r in (item,meta) for flag in ("ideal_line_shown","aid_probe_only"))):
        return None
    return views


def image(archive,view):
    pixels = cv2.imdecode(np.frombuffer(archive.read(Path(view["file"]).name),np.uint8),cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError("Unreadable recorded image")
    return pixels


def export(track):
    freeze, checkpoint = SHAPE/"training.json", SHAPE/"shape.pt"
    frozen = old.read(freeze)
    assert track not in frozen["training_circuits"]
    assert old.digest(checkpoint)==frozen["checkpoint_sha256"]
    entry = old.context()
    sources = {Path(__file__),freeze,checkpoint,old.MOUNT,Path(entry.__file__),
               Path(old.selections.__code__.co_filename),Path(combined.__code__.co_filename),
               Path(preprocess.__code__.co_filename),Path(baseline_data.__code__.co_filename)}
    group, _, mapping, bundles = baseline_data(track,sources)
    cache = crop_track(track,group,mapping,bundles)
    sources.add(cache)
    mount = old.read(old.MOUNT)
    keys, available, estimates, onset = inference(cache,checkpoint,mount["views"]["front"])
    expected = {key:i for i,key in enumerate(keys)}
    near,far = (old.SurfaceContrast(mount,near=value) for value in (True,False))
    far.stations = old.STATIONS[3:]
    far.lookup = far._sampling(mount,far.stations)
    raw = np.full((len(keys),2),np.nan,dtype=float)
    bounds = np.full((len(keys),len(old.STATIONS),2),np.nan,dtype=float)
    seen = set()
    for tag,folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[tag]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key = tag,item["capture_id"]
                if key not in expected:
                    continue
                assert key not in seen
                seen.add(key)
                i = expected[key]
                meta = metadata(archive,item,track)
                views = admitted(item,meta)
                assert available[i] == bool(views)
                motion = old.read_motion(meta) if views else None
                if motion is None or motion[1] < old.MIN_SPEED_MPS:
                    continue
                images = {name:image(archive,view) for name,view in views.items()}
                labs = {name:cv2.cvtColor(pixels,cv2.COLOR_BGR2LAB) for name,pixels in images.items()}
                base = old.selections(near,far,images,labs,near.detect(images),old.edge_scores(near,labs))
                shape = combined(base,estimates[i])["shape_geometry"]
                bounds[i] = shape
                offer = entry.local(old.STATIONS,shape,10.)
                if offer:
                    raw[i] = offer
    assert seen == set(keys)
    path = OUT/f"{track}-candidates.npz"
    np.savez_compressed(path,run=np.array([k[0] for k in keys]),frame=np.array([k[1] for k in keys]),
                        candidates=raw,sections=bounds,onset=onset)
    old.write(path.with_suffix(".json"),dict(track=track,frames=len(keys),candidates=int(np.isfinite(raw).all(1).sum()),
               shape_training=frozen["training_circuits"],seed=frozen["seed"],source_revision=old.revision(),
               arrays_sha256=old.digest(path),camera_bundle_sha256_by_run=bundles,
               source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="archive_end"))
    print(track,len(keys),int(np.isfinite(raw).all(1).sum()),flush=True)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    tracks = tuple(sys.argv[1:]) or DEVELOP
    if len(set(tracks)) != len(tracks) or not set(tracks) <= set(DEVELOP):
        raise ValueError("Only designated confirmation development circuits are admitted")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(export,tracks))
