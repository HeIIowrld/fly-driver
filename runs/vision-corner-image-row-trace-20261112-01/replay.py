"""Evaluate image-connected edges and their projection on the same saved frames."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
for name in ("vision-corner-section-hard-negatives-20261112-01",
             "vision-corner-far-pixel-trace-20261112-01"):
    sys.path.insert(1, str(RUNS / name))

from replay_hard_negatives import cv2, np, revision
from audit_road_edge_projection import section_points
from corner_fusion_boundaries import read_motion
from evaluate_live_corner_geometry import load_edges
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.diagnostic_path_projection import project_reference_points
from image_normal import edge_scores
from image_trace import FrontTrace
from projection_counterfactual import selections
from relative_corner_boundaries import SurfaceContrast
from replay_learned import publish, signals
from score_corner_captures import MIN_SPEED_MPS
from summarize_paired import category
from trace import context, LOCK, MOUNT, OLD, STATIONS
from train_vision_teacher_line import digest, jsonl, read, write

MODES = ("base", "far_wide", "trace_fill", "trace_replace")
START = 16.
CONFIRM = .25
SIDES = ("right", "left")


def combine(base, traced, *, fill):
    result = base.copy()
    eligible = (STATIONS[:, None] >= START) & np.isfinite(traced)
    if fill:
        eligible &= ~np.isfinite(base)
    result[eligible] = traced[eligible]
    width = result[:, 1] - result[:, 0]
    invalid = np.isfinite(width) & ((width <= 3.) | (width >= 20.))
    result[invalid] = base[invalid]
    return result


def front_truth(meta, track_edges):
    camera = next(v for v in meta["views"] if v["view_id"] == "front")
    body = camera["source_body_after_render"]
    xyz = section_points(track_edges, body, STATIONS)[:, ::-1]
    pose = np.asarray(body["renderer_only_position"])
    heading = np.asarray(body["renderer_only_look"])[[2, 0]]
    normal = np.array([-heading[1], heading[0]]) / np.linalg.norm(heading)
    lateral = (xyz - pose)[:, :, [2, 0]] @ normal
    uv, _, visible = project_reference_points(xyz.reshape(-1, 3), camera)
    pixels = np.where(visible[:, None], uv * [319, 179], np.nan).reshape(len(STATIONS), 2, 2)
    return lateral, pixels


def detail_row(tracer, curves, base, projected, reference, pixels):
    corrected = tracer.project(curves, pixels[:, :, 1])
    valid = (STATIONS[:, None] >= START) & np.isfinite(reference)
    keep = valid & np.isfinite(base) & np.isfinite(projected) & np.isfinite(corrected)
    comparisons = dict(base=base, traced_flat=projected, traced_map_rows=corrected)
    return dict(
        errors={name: [np.abs(values - reference)[keep[:, s], s].tolist() for s in range(2)]
                for name, values in comparisons.items()},
        points={name: (valid & np.isfinite(values)).sum(axis=0).tolist()
                for name, values in comparisons.items()},
        trace=[[[int(y), float(curve[y])] for y in np.flatnonzero(np.isfinite(curve))]
               for curve in curves.T],
        map_row_candidate=combine(base, corrected, fill=False),
    )


def boundary_arrays(archive, item, near, far, tracer):
    images = {v["view_id"]: cv2.imdecode(np.frombuffer(archive.read(Path(v["file"]).name), np.uint8),
                                       cv2.IMREAD_COLOR) for v in item["views"]}
    assert all(im is not None for im in images.values())
    labs = {name: cv2.cvtColor(image, cv2.COLOR_BGR2LAB) for name, image in images.items()}
    base = selections(near, far, images, labs, near.detect(images), edge_scores(near, labs))
    curves = tracer.curves(images["front"], base)
    trace_projected = tracer.project(curves)
    return base, trace_projected, curves


def replay_track(track, entry, replay, mount):
    sources = set()
    rows, old, mapping, bundles, profiles, training = replay.setup(track, sources)
    prior, receipt = OLD / f"{track}-frames.jsonl", OLD / "result.json"
    assert digest(prior) == read(receipt)["tracks"][track]["rows_sha256"]
    sources.update((prior, receipt))
    observed = {replay.key(r): r for r in map(json.loads, prior.read_text(encoding="utf8").splitlines())}
    expected, old = ({replay.key(r): r for r in value} for value in (rows, old))
    near, far = (SurfaceContrast(mount, near=value) for value in (True, False))
    far.stations = STATIONS[3:]
    far.lookup = far._sampling(mount, far.stations)
    tracer = FrontTrace(mount["views"]["front"], STATIONS, (180, 320))
    if track == "ks_silverstone1967":
        pins = read(LOCK)["hashes"]
        source = next(Path(p) for p in pins if Path(p).name == "fast_lane.ai")
        assert digest(source) == pins[str(source.resolve())]
        sources.update((source, LOCK))
        track_edges, _ = load_edges(source)
    else:
        track_edges = None
    timelines = {f"{m}_{g}": defaultdict(list) for m in replay.MODELS for g in MODES}
    records, diagnostic, paired = [], {}, Counter()
    for tag, folder in read(mapping).items():
        with CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[tag]
            for item in map(json.loads, archive.read("stream.jsonl").splitlines()):
                key = tag, item["capture_id"]
                if key not in expected:
                    continue
                t, reference = expected[key], old[key]
                path = Path(item["file"]).name
                assert archive.by_name[path]["sha256"] == item["file_sha256"]
                meta = archive.json(path)
                assert (meta["track_id"], meta["index"], meta["generation"]) == (track, key[1], item["generation"])
                usable = ({v["view_id"] for v in item["views"]} == {"front", "rear", "left", "right"}
                          and len(item["views"]) == 4 and not any(
                              row.get(flag) for row in (meta, item) for flag in ("ideal_line_shown", "aid_probe_only")))
                motion = read_motion(meta) if usable else None
                offers, pairs = {}, {}
                if motion is not None and motion[1] >= MIN_SPEED_MPS:
                    base, projected, curves = boundary_arrays(archive, item, near, far, tracer)
                    for mode, bounds in (("far_wide", base), ("trace_fill", combine(base, projected, fill=True)),
                                         ("trace_replace", combine(base, projected, fill=False))):
                        offers[mode] = entry.local(STATIONS, bounds, 10.)
                        pairs[mode] = int(np.isfinite(bounds[STATIONS >= START]).all(axis=1).sum())
                    if track_edges is not None and t["truth"] == "approach" and t["label"] == "2":
                        truth, pixels = front_truth(meta, track_edges)
                        report = detail_row(tracer, curves, base, projected, truth, pixels)
                        corrected = report.pop("map_row_candidate")
                        diagnostic[key[1]] = dict(report, truth_m=t["truth_endpoints_m"],
                                                  map_row_candidate=entry.local(STATIONS, corrected, 10.),
                                                  offers=offers)
                control = observed[key]["offers"].get("far_wide")
                assert (list(offers["far_wide"]) if offers.get("far_wide") else None) == control
                base = {m: reference["proposals"][spec[1]] for m, spec in replay.MODELS.items()}
                decisions = {}
                for m in replay.MODELS:
                    for mode in MODES:
                        offer = offers.get(mode)
                        score = signals(profiles[m].get(key), *offer)["local"] if offer else None
                        decision = base[m] or (offer if score is not None and score >= CONFIRM else None)
                        name = f"{m}_{mode}"
                        decisions[name] = decision
                        timelines[name][tag].append(publish(t, decision,
                                                             reference["baseline_observable"] or decision is not None))
                        if mode == "far_wide":
                            assert (list(decision) if decision else None) == observed[key]["final"][name]
                if not any(base.values()):
                    for mode in MODES[1:]:
                        a, b = (decisions[f"{m}_{mode}"] for m in replay.MODELS)
                        if a != b:
                            paired[mode, "t4t5" if b else "edge", category(t, a or b)] += 1
                records.append(dict(run=tag, frame=key[1], truth=t["truth"], label=t["label"],
                                    truth_endpoints_m=t["truth_endpoints_m"], physical_corner=t["physical_corner"],
                                    pairs=pairs, offers=offers, decisions=decisions))
    assert [replay.key(r) for r in records] == list(expected)
    path = OUT / f"{track}-frames.jsonl"
    jsonl(path, records)
    return dict(training=training, metrics={k: replay.totals(v) for k, v in timelines.items()},
                paired=[dict(mode=k[0], accepted_by=k[1], category=k[2], frames=v)
                        for k, v in sorted(paired.items())],
                diagnostic=diagnostic, rows_sha256=digest(path),
                source_sha256={str(p): digest(p) for p in sorted(sources)})


def main():
    cv2.setNumThreads(1)
    entry = context()
    replay = entry.import_replay()
    track_list = tuple(sys.argv[1:]) or replay.CHECK
    if len(set(track_list)) != len(track_list) or any(t not in (*replay.CHECK, *replay.profile.CALIBRATION_TRACKS)
                                                     for t in track_list):
        raise ValueError("Unknown or repeated track")
    mount = read(MOUNT)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = {t: executor.submit(replay_track, t, entry, replay, mount) for t in track_list}
        tracks = {}
        for t, job in jobs.items():
            tracks[t] = job.result()
            print(t, tracks[t]["metrics"], flush=True)
    sources = {Path(__file__), MOUNT, Path(FrontTrace.curves.__code__.co_filename),
               Path(entry.__file__), Path(replay.__file__), Path(selections.__code__.co_filename),
               Path(section_points.__code__.co_filename), Path(project_reference_points.__code__.co_filename)}
    filename = "result.json" if track_list == replay.CHECK else "development-result.json"
    write(OUT / filename, dict(seed=None, source_revision=revision(), confirmation_cut=CONFIRM,
                              replaced_from_m=START, tracks=tracks,
                              source_sha256={str(p): digest(p) for p in sorted(sources)},
                              termination="all_saved_frames_scored_with_control_parity"))


if __name__ == "__main__":
    main()
