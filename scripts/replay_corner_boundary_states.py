"""Replay the frozen corner observer using only recorded visual and local body inputs."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTING = ROOT/"runs/vision-corner-recovery-routing-20261112-01"
sys.path.insert(0, str(ROUTING))
from routing import ALIGNED, LISTWISE, MAX_ALONG, old, joblib
from aligned_sections import sequence, valid_pairs
from directional_features import PHASE, relative_features, causal_prefix
from phase_features import frame_features
from train_phase import phase_scores
from rank import encode
import alternatives as alt

METHODS = ("temporal", "t4t5_confirm")
POLICY = "empty_or_consistent"


def load_observer():
    paths = {
        "rank": LISTWISE/"rankers.joblib",
        "phase": PHASE/"phase.joblib",
        "straight": ROUTING/"straight.joblib",
    }
    rank_training = old.read(LISTWISE/"training.json")
    phase_training = old.read(PHASE/"training.json")
    straight_training = old.read(ROUTING/"straight-training.json")
    routing = old.read(ROUTING/"training.json")
    for name, receipt in (("rank", rank_training), ("phase", phase_training), ("straight", straight_training)):
        if old.digest(paths[name]) != receipt["checkpoint_sha256"]:
            raise ValueError(f"Frozen {name} checkpoint differs from its receipt")
    if routing["shared_phase_veto"] != phase_training["veto_corner_probability_max"]:
        raise ValueError("Routing and phase confirmation settings disagree")
    models = {name: joblib.load(path) for name, path in paths.items()}
    settings = dict(cuts=routing["cuts"][POLICY], phase_cut=routing["shared_phase_veto"],
                    straight_cut=straight_training["straight_cut"],
                    motion_features=models["rank"]["t4t5"]["confirmer"]["motion"].n_features_in_,
                    max_agreement_m=routing["max_agreement_distance_m"])
    assert settings["max_agreement_m"] == MAX_ALONG
    inputs = {*paths.values(), LISTWISE/"training.json", PHASE/"training.json",
              ROUTING/"straight-training.json", ROUTING/"training.json",
              Path(sequence.__code__.co_filename), Path(frame_features.__code__.co_filename),
              Path(causal_prefix.__code__.co_filename), Path(alt.sequence.__code__.co_filename),
              Path(alt.local_features.__code__.co_filename), Path(alt.transform.__code__.co_filename),
              Path(encode.__code__.co_filename), Path(phase_scores.__code__.co_filename), Path(__file__)}
    return models, settings, inputs


def sensor_rows(data, settings):
    expected = {"run", "frame", "generation", "received_s", "stations", "sections",
                "motion", "heads", "t4t5"}
    if set(data.files) != expected:
        raise ValueError("Sensor bundle must use the declared fields only")
    if not np.array_equal(data["stations"], old.STATIONS):
        raise ValueError("Boundary station grid differs from the model")
    n = len(data["frame"])
    if n == 0 or len(set(zip(data["run"], data["frame"], strict=True))) != n:
        raise ValueError("Timeline is empty or contains duplicate frame identities")
    for name, shape in (("run", (n,)), ("frame", (n,)), ("generation", (n,)), ("received_s", (n,)),
                        ("sections", (n, len(old.STATIONS), 2)), ("motion", (n, 3)),
                        ("t4t5", (n, 2, settings["motion_features"]))):
        if data[name].shape != shape:
            raise ValueError(f"Unexpected {name} shape")
    if data["heads"].shape != (n, 4):
        raise ValueError("Unexpected onset head shape")
    return [dict(run=str(run), frame=int(frame), generation=str(generation), received_s=float(received))
            for run, frame, generation, received in zip(data["run"], data["frame"], data["generation"],
                                                        data["received_s"], strict=True)]


def proposals(rows, section, motion, heads):
    primary = np.full((len(rows), 2), np.nan)
    for i, boundary in enumerate(section):
        if not np.isfinite(motion[i]).all() or motion[i, 1] < old.MIN_SPEED_MPS:
            continue
        wide = alt.segments(boundary, 10.)
        if wide:
            primary[i] = wide[0][:2]
    return alt.sequence(rows, dict(sections=section, primary=primary, motion=motion, heads=heads))


def rank_candidates(rows, sections, original, motion, visual, phase, direct, recovered, models):
    geometry, history, appearance, offers = [], [], [], []
    for i, ((before, before_lags), (after, after_lags)) in enumerate(zip(zip(*direct), zip(*recovered), strict=True)):
        current = {c["offer"] for c in before}
        origins = [(c, h, original[i], False) for c, h in zip(before, before_lags, strict=True)]
        origins.extend((c, h, sections[i], True) for c, h in zip(after, after_lags, strict=True)
                       if c["offer"] not in current)
        if not origins:
            continue
        local = causal_prefix(rows, motion, i)
        for c, h, bounds, restored in origins:
            offer = c["offer"]
            geom = alt.local_features(old.STATIONS, bounds, offer, None, 0.)[:-4]
            directional = relative_features(old.STATIONS, bounds, motion[i], local, offer)
            geometry.append(np.r_[geom, c["source"], phase[i], directional])
            history.append(h)
            appearance.append(visual[i, offer[0]-1])
            offers.append((i, offer, restored))
    if not offers:
        return {}, {}
    features = dict(edge=np.asarray(geometry), history=np.asarray(history), motion=np.asarray(appearance))
    temporal = models["temporal"]
    values = encode(features, temporal["confirmer"], "temporal")
    ranking = values @ temporal["selector"]["weights"]
    edge = temporal["confirmer"]["model"].predict_proba(values)[:, 1]
    motion_valid = np.isfinite(visual).all(axis=(1, 2))
    valid = np.array([motion_valid[i] for i, _, _ in offers])
    motion_score = np.full(len(offers), np.nan)
    if valid.any():
        combined = models["t4t5"]["confirmer"]
        values = encode({k: v[valid] for k, v in features.items()}, combined, "t4t5")
        motion_score[valid] = combined["model"].predict_proba(values)[:, 1]
    best, best_current = {}, {}
    for i, (frame, offer, restored) in enumerate(offers):
        if frame not in best or ranking[i] > ranking[best[frame]]:
            best[frame] = i
        if not restored and (frame not in best_current or ranking[i] > ranking[best_current[frame]]):
            best_current[frame] = i
    choices, scores = {}, {}
    for frame, j in best.items():
        current = best_current.get(frame)
        if current is not None:
            a, b = offers[current][1], offers[j][1]
            if a[0] != b[0] or abs(a[1]-b[1]) > MAX_ALONG:
                j = current
        choices[frame] = offers[j][1]
        scores[frame] = (edge[j], motion_score[j])
    return choices, scores


def observe(data, models, settings):
    rows = sensor_rows(data, settings)
    motion, raw, heads, visual = (data[n] for n in ("motion", "sections", "heads", "t4t5"))
    continuity = [False, *(alt.adjacent(a, b) for a, b in zip(rows, rows[1:]))]
    aligned = sequence(old.STATIONS, continuity, raw, motion)
    changed = valid_pairs(aligned) & ~valid_pairs(raw)
    phase = frame_features(rows, raw, motion)
    phase_available = np.isfinite(phase).all(axis=1)
    q = np.full(len(rows), np.nan)
    if phase_available.any():
        q = phase_scores(dict(phase_x=phase), models["phase"])[:, 2]
    direct = proposals(rows, raw, motion, heads)
    recovered = proposals(rows, aligned, motion, heads)
    choices, scores = rank_candidates(rows, aligned, raw, motion, visual, phase, direct, recovered, models["rank"])
    can_straight = (np.isfinite(raw).all(axis=2).sum(axis=1) >= 6) & phase_available
    straight = np.full(len(rows), np.nan)
    if can_straight.any():
        straight[can_straight] = models["straight"].predict_proba(phase[can_straight])[:, 1]
    available_motion = np.isfinite(visual).all(axis=(1, 2))
    results = []
    for i, row in enumerate(rows):
        offer = choices.get(i)
        forecasts = {}
        for j, method in enumerate(METHODS):
            score = scores[i][j] if offer is not None else np.nan
            allowed = method == "temporal" or available_motion[i]
            turn = allowed and np.isfinite(score) and score >= settings["cuts"][method] and q[i] <= settings["phase_cut"]
            if turn:
                state, distance = {1: "left", 2: "right"}[offer[0]], offer[1]
            else:
                state, distance = ("straight" if allowed and straight[i] >= settings["straight_cut"] else "abstain"), None
            forecasts[method] = dict(state=state, distance_m=distance)
        results.append(dict(run=row["run"], frame=row["frame"], motion_vision_available=bool(available_motion[i]),
                            body_motion_available=bool(np.isfinite(motion[i]).all() and motion[i, 1] >= 0.),
                            candidate=offer, forecasts=forecasts))
    return results, aligned, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any((args.output/name).resolve() == args.input.resolve()
           for name in ("boundaries.npz", "forecasts.jsonl", "result.json")):
        parser.error("The output artifacts must not overwrite the sensor bundle")
    args.output.mkdir(parents=True, exist_ok=True)
    models, settings, sources = load_observer()
    with np.load(args.input, allow_pickle=False) as data:
        records, aligned, changed = observe(data, models, settings)
        boundary_path = args.output/"boundaries.npz"
        np.savez_compressed(boundary_path, run=data["run"], frame=data["frame"], stations=data["stations"],
                            observed=data["sections"], estimated=aligned, restored=changed)
    forecast_path = args.output/"forecasts.jsonl"
    old.jsonl(forecast_path, records)
    old.write(args.output/"result.json", dict(status="observed_archive", source_revision=old.revision(),
              frames=len(records), settings=settings, input_sha256=old.digest(args.input),
              source_sha256={str(p): old.digest(p) for p in sorted(sources)},
              forecasts_sha256=old.digest(forecast_path), boundaries_sha256=old.digest(boundary_path),
              termination="sensor_stream_exhausted"))


if __name__ == "__main__":
    main()
