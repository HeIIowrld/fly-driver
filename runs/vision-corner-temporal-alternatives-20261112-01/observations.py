"""Collect only causal body speed/yaw and existing frozen boundary arrays."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
GATE = OUT.parent / "vision-corner-shape-confirmation-20261112-01"
sys.path.insert(0,str(GATE))
from collect import SHAPE, DEVELOP, admitted, inference, metadata, old
from features import cached, SILVER
from evaluate_vallelunga import CAPTURE, TRACK as VAL, collect_offers
from replay_learned import baseline_data

TRACKS = (*DEVELOP,SILVER,VAL)


def raw_inputs(track, sources):
    if track != VAL:
        keys, candidates, sections, base = cached(track,sources)
        group, _, mapping, bundles = baseline_data(track,sources)
        assert keys==[(r["run"],r["frame"]) for r in group["rows"]]
        if track==SILVER:
            path = SHAPE/"cross-sections.npz"
        else:
            path = GATE/f"{track}-candidates.npz"
        with np.load(path,allow_pickle=False) as data:
            heads = data["onset"]
        return keys,candidates,sections,base,heads,mapping,bundles

    proof = GATE/"vallelunga-result.json"
    previous = old.read(proof)
    saved = GATE/"vallelunga-frames.jsonl"
    assert old.digest(saved)==previous["rows_sha256"]
    rows = [json.loads(line) for line in saved.read_text(encoding="utf8").splitlines()]
    mapping, capture = CAPTURE/"archive-map.json",CAPTURE/"result.json"
    bundle = old.read(capture)["archive"]["bundle_sha256"]
    bundles = {VAL:bundle}
    cache = GATE/f"{VAL}-front.npz"
    checkpoint = SHAPE/"shape.pt"
    shape_training = SHAPE/"training.json"
    assert old.digest(checkpoint)==old.read(shape_training)["checkpoint_sha256"]
    assert old.digest(cache)==previous["source_sha256"][str(cache.resolve())]
    mount = old.read(old.MOUNT)
    keys,available,estimates,heads = inference(cache,checkpoint,mount["views"]["front"])
    candidates,sections = collect_offers(keys,available,estimates,mapping,bundles)
    assert [(r["run"],r["frame"]) for r in rows]==keys
    assert [list(o) if o else None for o in candidates]==[r["candidate"] for r in rows]
    sources.update((proof,saved,mapping,capture,cache,checkpoint,shape_training,old.MOUNT,
                    Path(collect_offers.__code__.co_filename),Path(inference.__code__.co_filename)))
    return keys,candidates,sections,[r["common_edge_warning"] for r in rows],heads,mapping,bundles


def collect(track):
    sources = {Path(__file__),Path(raw_inputs.__code__.co_filename),Path(cached.__code__.co_filename),
               Path(old.read_motion.__code__.co_filename),Path(metadata.__code__.co_filename)}
    keys,candidates,sections,base,heads,mapping,bundles = raw_inputs(track,sources)
    expected = {key:i for i,key in enumerate(keys)}
    motion = np.full((len(keys),3),np.nan,dtype=float)
    seen = set()
    for tag,folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            assert archive.bundle_sha256==bundles[tag]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key = tag,item["capture_id"]
                if key not in expected:
                    continue
                assert key not in seen
                seen.add(key)
                meta = metadata(archive,item,track)
                if admitted(item,meta):
                    value = old.read_motion(meta)
                    if value is not None:
                        motion[expected[key]] = value
    assert seen==set(keys)
    # Stored columns are source clock, local speed and local yaw rate only.
    path = OUT/f"{track}-observations.npz"
    def encode(choices):
        return np.asarray([x if x is not None else [np.nan,np.nan] for x in choices],dtype=float)

    np.savez_compressed(path,run=np.array([k[0] for k in keys]),frame=np.array([k[1] for k in keys]),
                        motion=motion,sections=sections,primary=encode(candidates),edge_base=encode(base),heads=heads)
    old.write(path.with_suffix(".json"),dict(track=track,frames=len(keys),seed=None,source_revision=old.revision(),
              arrays_sha256=old.digest(path),motion_frames=int(np.isfinite(motion).all(1).sum()),
              camera_bundle_sha256_by_run=bundles,
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="archive_end"))
    print(track,len(keys),int(np.isfinite(motion).all(1).sum()),flush=True)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    tracks = tuple(sys.argv[1:]) or TRACKS
    if not set(tracks)<=set(TRACKS) or len(set(tracks))!=len(tracks):
        raise ValueError("Unexpected circuit selection")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(collect,tracks))
