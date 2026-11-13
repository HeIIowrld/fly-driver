"""Create image-aligned physical-surface labels, never passed to the actor."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from cross_sections import Surfaces, intersect
from kn5_surfaces import AC, OUT, digest, load_model, surface_config, surface_key, track_models

RUNS = OUT.parent
sys.path.insert(0, str(RUNS / "vision-corner-image-row-trace-20261112-01"))
import replay as reference
from trace import CAP

TRACK = "ks_silverstone1967"
DIRECT = RUNS / "vision-corner-visible-line-labels-20261112-01"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def source_triangles():
    root = AC / "content" / "tracks" / TRACK
    receipt_path = OUT / f"{TRACK}-mesh-inventory.json"
    receipt = read(receipt_path)
    config, _ = surface_config(root)
    paths, _ = track_models(root)
    assert config == receipt["surfaces"]
    for filename, sha in receipt["config_sha256"].items():
        assert digest(Path(filename)) == sha
    for path in paths:
        expected = receipt["models"][path.name]
        if not expected["nodes"]:
            assert digest(path) == expected["sha256"]
            continue
        actual, meshes = load_model(path)
        assert actual["sha256"] == expected["sha256"] and len(actual["nodes"]) == len(expected["nodes"])
        for metadata, data in meshes:
            key = surface_key(metadata["name"], config)
            if key is None and "WALL" not in metadata["name"].upper():
                raise ValueError(f"Unregistered physical surface: {metadata['name']}")
            if key is not None:
                yield key, data


def projection(world, camera):
    points = np.asarray(world, dtype=float)
    if not len(points):
        return []
    uv, _, visible = reference.project_reference_points(points, camera)
    size = np.array([camera["width"]-1, camera["height"]-1])
    return [(p*size).tolist() if seen else None for p, seen in zip(uv, visible, strict=True)]


def audit_checks():
    tri = np.array([[[0., -2., 0.], [4., 2., 1.], [0., 2., 1.]]])
    out, valid = intersect(tri, 2.)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(out, [[[0., .5], [2., 1.]]])
    assert not intersect(tri, 4.)[1].any()


def main():
    audit_checks()
    track = AC / "content" / "tracks" / TRACK
    config, _ = surface_config(track)
    surfaces = Surfaces(source_triangles(), config)
    print("Parsed ground triangles:", len(surfaces.triangles), flush=True)
    entry = reference.context()
    long_result = reference.OLD / "result.json"
    long_rows = reference.OLD / f"{TRACK}-frames.jsonl"
    assert digest(long_rows) == read(long_result)["tracks"][TRACK]["rows_sha256"]
    expected = {row["frame"]: row for row in map(json.loads, long_rows.read_text(encoding="utf8").splitlines())}
    manual_path = DIRECT / "labels.json"
    manual = read(manual_path)["frames"]
    mapping, archive_proof = CAP / "archive-map.json", CAP / "result.json"
    pins = read(reference.LOCK)["hashes"]
    ai_path = next(Path(p) for p in pins if Path(p).name == "fast_lane.ai")
    assert digest(ai_path) == pins[str(ai_path.resolve())]
    ai, _ = reference.load_edges(ai_path)
    checked = set()
    count = Counter()
    diagnostic = {}
    predictions = []
    label_frames = []
    stations = np.r_[0., reference.STATIONS]
    with reference.CameraArchive(read(mapping)[TRACK]) as archive:
        assert archive.bundle_sha256 == read(archive_proof)["archive"]["bundle_sha256"]
        for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
            frame = item["capture_id"]
            if frame not in expected:
                continue
            name = Path(item["file"]).name
            assert archive.by_name[name]["sha256"] == item["file_sha256"]
            meta = archive.json(name)
            assert (meta["track_id"], meta["index"], meta["generation"]) == (TRACK, frame, item["generation"])
            if frame in checked:
                raise ValueError("Duplicate frame identity")
            checked.add(frame)
            info = expected[frame]
            camera = next(v for v in meta["views"] if v["view_id"] == "front")
            body = camera["source_body_after_render"]
            if not body or not body.get("physics_available"):
                count["missing_body"] += 1
                continue
            found = surfaces.sections(body, stations)
            if found[0]:
                count["body_inside_valid_surface"] += 1
            physical = found[1:]
            bounds = np.full((len(physical), 2), np.nan)
            witnessed = bounds.copy()
            label = []
            for i, row in enumerate(physical):
                if row is None:
                    continue
                bounds[i] = [a["lateral_m"] for a in row]
                pixels = projection([a["world"] for a in row], camera)
                for j, (point, pixel) in enumerate(zip(row, pixels, strict=True)):
                    known = point["surface_transition_confirmed"]
                    if known:
                        witnessed[i, j] = bounds[i, j]
                    label.append(dict(station_m=float(reference.STATIONS[i]), side=reference.SIDES[j],
                                      lateral_m=point["lateral_m"], pixel=pixel, inside=point["inside"],
                                      outside=point["outside"], witnessed=known))
            count["sections"] += int(np.isfinite(bounds).all(axis=1).sum())
            count["witnessed_pairs"] += int(np.isfinite(witnessed).all(axis=1).sum())
            raw = entry.local(reference.STATIONS, bounds, 10.)
            confirmed = entry.local(reference.STATIONS, witnessed, 10.)
            predictions.append(dict(frame=frame, truth=info["truth"], label=info["label"],
                                    truth_endpoints_m=info["truth_endpoints_m"],
                                    mesh=raw, adjacent_invalid_only=confirmed,
                                    image=info["offers"].get("far_wide")))
            if str(frame) in manual:
                exact = reference.section_points(ai, body, reference.STATIONS)[:, ::-1]
                pose = np.asarray(body["renderer_only_position"])
                axis = np.asarray(body["renderer_only_look"])[[2, 0]]
                lateral = (exact-pose)[:, :, [2, 0]] @ (np.array([-axis[1], axis[0]]) / np.linalg.norm(axis))
                label_frames.append(frame)
                for rec in label:
                    i, j = int(rec["station_m"] / 2) - 1, reference.SIDES.index(rec["side"])
                    rec["ai_lateral_m"] = float(lateral[i, j]) if np.isfinite(lateral[i, j]) else None
                    rec["ai_pixel"] = projection([exact[i, j]], camera)[0] if np.isfinite(exact[i, j]).all() else None
                diagnostic[frame] = dict(labels=label, truth_m=info["truth_endpoints_m"],
                                         mesh=raw, adjacent_invalid_only=confirmed, image=info["offers"].get("far_wide"),
                                         approach_label=info["label"])
            if len(checked) % 300 == 0:
                print("Frames:", len(checked), dict(count), flush=True)
    assert checked == set(expected) and set(label_frames) == {int(f) for f in manual}
    pred_path = OUT / "mesh-oracle.jsonl"
    reference.jsonl(pred_path, predictions)
    paths = (Path(__file__), OUT / "kn5_surfaces.py", OUT / "cross_sections.py", manual_path,
             OUT / f"{TRACK}-mesh-inventory.json", long_result, long_rows, mapping, archive_proof, ai_path,
             reference.LOCK, Path(reference.project_reference_points.__code__.co_filename), Path(entry.__file__))
    reference.write(OUT / "surface-labels.json", dict(
        status="offline_diagnostic", source_revision=reference.revision(), seed=None,
        meshes_read_from_install=True, source_asset_copies=0, frames=len(checked),
        scored_mesh_frames=len(predictions), counts=dict(count),
        actor_input=False, labels=diagnostic, mesh_oracle_sha256=digest(pred_path),
        source_sha256={str(p): digest(p) for p in paths}, termination="archive_end"))


if __name__ == "__main__":
    main()
