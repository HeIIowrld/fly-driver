"""Propagate aligned image-boundary uncertainty into next-corner decisions."""
from collections import deque
import importlib.util
from pathlib import Path

import numpy as np

from train_vision_teacher_line import digest, jsonl, read, write

out = Path(__file__).parent
runs = out.parent
previous = runs/"vision-corner-temporal-boundaries-20261111-01"
spec = importlib.util.spec_from_file_location("temporal_replay", previous/"replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
prior = read(previous/"result.json")
assert digest(previous/"replay.py") == prior["source_sha256"]
assert digest(previous/"extraction.json") == prior["extraction_sha256"]
geometry = replay.read_rows(previous/"geometry.jsonl", prior["geometry_sha256"])
observations = replay.read_rows(previous/"observations.jsonl", replay.extraction["observations_sha256"])
stations = replay.stations
floors = (0., .05, .10, .20)
modes = ("bounded", "quadrature")
setting_names = [f"{mode}_{floor}" for mode in modes for floor in floors]
coherent_names = ("whole_shape_min", "whole_shape_quartile", "whole_shape_median")


def recovered(rows, proofs):
    history = deque()
    prior_motion = None
    for row, proof in zip(rows, proofs, strict=True):
        assert replay.key(row) == replay.key(proof)
        motion = row["motion"]
        if not proof["motion_contiguous"]:
            history.clear()
        else:
            rotation, move = replay.transform(motion, prior_motion)
            history = deque((t, replay.advance(points, rotation, move))
                            for t, points in history if motion[0]-t <= .24)
        if motion is not None and row["sections"] is not None:
            sections = np.array(row["sections"], dtype=float).reshape(-1, 3)
            points = np.stack((np.repeat(sections[:, :1], 2, axis=1), sections[:, 1:]), axis=-1)
            history.append((motion[0], points))
            values = np.stack([replay.section_values(points) for _, points in history])
            stamps = np.array([t for t, _ in history])
            center = replay.fuse(values, stamps, motion[0], .24)
            bounds = np.full_like(center, np.nan)
            support = np.isfinite(values).all(axis=-1)
            for i in np.flatnonzero(support.sum(0) >= 2):
                sample = values[support[:, i], i]
                bounds[i] = np.max(np.abs(sample-center[i]), axis=0)
            yield proof, center, bounds, values
        else:
            yield proof, None, None, None
        prior_motion = motion


def curvature_bounds(sections, scatter):
    result = {name: [] for name in setting_names}
    raw = []
    usable = np.isfinite(sections).all(axis=1)
    width = sections[:, 1]-sections[:, 0]
    usable &= (width > 3.) & (width < 20.)
    for center in np.arange(4., 22.):
        offset = stations-center
        mask = usable & (abs(offset) <= replay.radius)
        x = offset[mask]
        if len(x) < 4 or x[0] >= 0 or x[-1] <= 0 or max(np.diff(x)) > 4.:
            continue
        design = np.column_stack((np.ones(len(x)), x, x*x))
        weights = np.linalg.pinv(design)
        coefficients = weights@sections[mask]
        denom = (1.+coefficients[1]**2)**1.5
        k = 2.*coefficients[2]/denom
        sign = np.sign(k[0]) if k[0]*k[1] > 0 else 0.
        raw.append((float(center), float(sign*min(abs(k)))))
        for floor in floors:
            changes = 2.*np.abs(weights[2, :, None])*np.maximum(scatter[mask], floor)
            reductions = {"bounded": changes.sum(axis=0)/denom,
                          "quadrature": np.sqrt((changes**2).sum(axis=0))/denom}
            for mode, reduction in reductions.items():
                lower = max(0., float(min(np.abs(k)-reduction)))
                result[f"{mode}_{floor}"].append((float(center), float(sign*lower)))
    return raw, result


def whole_shape_bounds(raw, aligned):
    """Preserve within-frame correlation by evaluating complete observed shapes."""
    samples = [dict(replay.curvature(shape)) for shape in aligned]
    result = {name: [] for name in coherent_names}
    for station, signed in raw:
        direction = np.sign(signed)
        observed = [direction*sample[station] for sample in samples if station in sample]
        for name, quantile in zip(coherent_names, (0., .25, .5), strict=True):
            bound = float(np.quantile(observed, quantile)) if len(observed) >= 2 else 0.
            result[name].append((station, float(direction*min(abs(signed), max(bound, 0.)))))
    return result

def run():
    forecasts = []
    for proof, center, scatter, values in recovered(observations, geometry):
        original = proof["curvature"].get("history_0.24", [])
        if center is None:
            assert not original
            result = {}
        else:
            raw, result = curvature_bounds(center, scatter)
            result.update(whole_shape_bounds(raw, values))
            np.testing.assert_allclose(np.asarray(raw), np.asarray(original), rtol=0., atol=1e-10)
        forecasts.append(dict(run=proof["run"], frame=proof["frame"], travel_m=proof["travel_m"], curvature=result))
    files = {
        "magione": runs/"vision-corner-independent-runner-rehearsal-20261111-01"/"replay"/"candidate"/"score"/"rows.jsonl",
        "imola": runs/"vision-corner-imola-full-blended-20261111-01"/"rows.jsonl",
    }
    reports = {}
    for track, path in files.items():
        reference = replay.read_rows(path, prior["scoring_inputs"][str(path)])
        baseline = replay.reports(reference)
        summaries = {"existing_blend": replay.compact(baseline)}
        for mode in ("unmodified", *setting_names, *coherent_names):
            source = geometry if mode == "unmodified" else [
                dict(r, curvature={"history_0.24": r["curvature"].get(mode, [])}) for r in forecasts]
            for cut in (.002, .004):
                confirmed = replay.confirmed_predictions(source, cut, 3, 1.)
                updated = [replay.rescore(row, confirmed[replay.key(row)]) if confirmed[replay.key(row)]
                           and row["state"] not in ("left", "right") else row for row in reference]
                summary = replay.reports(updated)
                summaries[f"{mode}_{cut}"] = dict(replay.compact(summary),
                    paired_with_blend=replay.matched_changes(baseline, summary))
        # Keep the unmodified onset and use its local margin only to decide
        # whether to emit an additional warning on the current frame.
        base_confirmed = replay.confirmed_predictions(geometry, .002, 3, 1.)
        score_index = {replay.key(r): r for r in forecasts}
        raw_index = {replay.key(r): dict(r["curvature"].get("history_0.24", [])) for r in geometry}
        for mode in ("bounded_0.0", "quadrature_0.0", "quadrature_0.05", *coherent_names):
            evidence = {}
            for key, choice in base_confirmed.items():
                if choice is None:
                    continue
                _, distance = choice
                scores = dict(score_index[key]["curvature"][mode])
                raw = raw_index[key]
                evidence[key] = min(abs(scores[x])/abs(raw[x]) for x in (distance, distance+1))
            for threshold in (.10, .25, .50, .75):
                changed = []
                for row in reference:
                    key = replay.key(row)
                    if row["state"] not in ("left", "right") and evidence.get(key, 0.) >= threshold:
                        row = replay.rescore(row, base_confirmed[key])
                    changed.append(row)
                report = replay.reports(changed)
                summaries[f"local_margin_{mode}_{threshold}"] = dict(replay.compact(report),
                    paired_with_blend=replay.matched_changes(baseline, report))

        reports[track] = summaries
    jsonl(out/"bounds.jsonl", forecasts)
    write(out/"result.json", dict(summary=reports, candidate_promoted=False, seed=None, termination="archive_end",
          source_sha256=digest(Path(__file__)), bounds_sha256=digest(out/"bounds.jsonl"),
          revision=prior["revision"], inputs={str(p):digest(p) for p in (
              previous/"result.json", previous/"geometry.jsonl", previous/"extraction.json",
              previous/"replay.py", previous/"observations.jsonl", *files.values())}))
    for mode, row in reports["magione"].items():
        print(mode, *(row[k] for k in ("left", "right", "joint", "outside")),
              reports["imola"][mode]["outside"])


if __name__ == "__main__":
    run()
