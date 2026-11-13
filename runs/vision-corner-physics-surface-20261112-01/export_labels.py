"""Export per-view labels and body cross-sections from installed surface geometry."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

from surface_replay import OUT, TRACK, projection, read, reference
from cross_sections import Surfaces
from kn5_surfaces import AC, digest, load_model, surface_config, surface_key, track_models
from replay_learned import baseline_data, frame_key

VIEWS = ("front", "left", "right", "rear")


def verified_surfaces(track):
    root = AC / "content" / "tracks" / track
    path = OUT / f"{track}-mesh-inventory.json"
    inventory = read(path)
    assert inventory["parser_sha256"] == digest(OUT / "kn5_surfaces.py")
    configuration, _ = surface_config(root)
    assert configuration == inventory["surfaces"]
    files, _ = track_models(root)
    for name, sha in inventory["config_sha256"].items():
        assert digest(Path(name)) == sha
    meshes = []
    for file in files:
        proof = inventory["models"][file.name]
        if not proof["nodes"]:
            assert digest(file) == proof["sha256"]
            continue
        actual, data = load_model(file)
        assert actual["sha256"] == proof["sha256"]
        for node, triangles in data:
            key = surface_key(node["name"], configuration)
            if key is not None:
                meshes.append((key, triangles))
            elif "WALL" not in node["name"].upper():
                raise ValueError(f"Unregistered physical mesh: {node['name']}")
    return Surfaces(meshes, configuration), path


def add_labels(surface, cameras, lateral, witness, image_px, body):
    points = surface.sections(body, np.r_[0., reference.STATIONS])[1:]
    valid = []
    world = []
    for i, section in enumerate(points):
        if section is None:
            continue
        for j, side in enumerate(section):
            lateral[i, j] = side["lateral_m"]
            witness[i, j] = side["surface_transition_confirmed"]
            valid.append((i, j))
            world.append(side["world"])
    for k, camera in enumerate(cameras):
        pixels = projection(world, camera)
        for (i, j), point in zip(valid, pixels, strict=True):
            if point is not None:
                image_px[k, i, j] = point


def export(track):
    surfaces, inventory = verified_surfaces(track)
    sources = {Path(__file__), Path(add_labels.__code__.co_filename), inventory,
               Path(Surfaces.sections.__code__.co_filename), Path(projection.__code__.co_filename),
               Path(reference.project_reference_points.__code__.co_filename),
               Path(baseline_data.__code__.co_filename)}
    group, _, mapping, bundles = baseline_data(track, sources)
    expected = {frame_key(row): i for i, row in enumerate(group["rows"])}
    length = len(expected)
    lateral = np.full((length, len(reference.STATIONS), 2), np.nan, dtype=np.float32)
    witness = np.zeros(lateral.shape, dtype=bool)
    image_px = np.full((length, len(VIEWS), len(reference.STATIONS), 2, 2), np.nan, dtype=np.float32)
    seen = set()
    for run, folder in read(mapping).items():
        with reference.CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[run]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                key = run, item["capture_id"]
                if key not in expected:
                    continue
                index = expected[key]
                assert key not in seen
                seen.add(key)
                name = Path(item["file"]).name
                assert archive.by_name[name]["sha256"] == item["file_sha256"]
                meta = archive.json(name)
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, key[1], item["generation"])
                by_view = {v["view_id"]: v for v in meta["views"]}
                if (len(meta["views"]) != 4 or set(by_view) != set(VIEWS)
                    or any(r.get(k) for r in (meta,item) for k in ("ideal_line_shown", "aid_probe_only"))):
                    continue
                body = by_view["front"].get("source_body_after_render")
                if not body or not body.get("physics_available"):
                    continue
                add_labels(surfaces, [by_view[v] for v in VIEWS], lateral[index], witness[index], image_px[index], body)
    assert set(expected) == seen
    path = OUT / f"{track}-train-labels.npz"
    np.savez_compressed(path, run=np.array([key[0] for key in expected]),
                        frame=np.array([key[1] for key in expected]), stations=reference.STATIONS,
                        views=np.array(VIEWS), lateral_m=lateral, adjacent_invalid=witness, image_px=image_px)
    supported = witness[:, None] & np.isfinite(image_px).all(axis=-1)
    receipt = dict(track=track, frames=length, label_role="offline_supervision_or_evaluation_only",
                   image_semantics="projection inside the viewport; occlusion has not been adjudicated",
                   raw_assets_copied=0, source_sha256={str(p): digest(p) for p in sorted(sources)},
                   camera_bundle_sha256_by_run=bundles, labels_sha256=digest(path), training_applied=False,
                   source_revision=reference.revision(), seed=None,
                   adjacent_invalid_counts_by_view_and_side=dict(zip(VIEWS, supported.sum(axis=(0, 2)).tolist())),
                   adjacent_invalid_pairs=int(witness.all(axis=-1).sum()), termination="archive_end")
    reference.write(path.with_suffix(".json"), receipt)
    print(track, length, receipt["adjacent_invalid_counts_by_view_and_side"], flush=True)


if __name__ == "__main__":
    names = tuple(sys.argv[1:]) or (TRACK, "spa", "magione")
    if len(set(names)) != len(names) or not set(names) <= {TRACK, "spa", "magione"}:
        raise ValueError("Unknown or duplicate circuit")
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(export, names))
