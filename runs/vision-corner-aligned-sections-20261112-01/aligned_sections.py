"""Recover 34 m road sections from contiguous raw observations in local body coordinates."""
from collections import deque
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
LISTWISE = OUT.parent/"vision-corner-listwise-20261112-01"
sys.path.insert(0, str(LISTWISE))
from listwise import enrich, old
from directional_features import PARENT, relative_features, causal_prefix
import alternatives as alt

MAX_AGE_S = .45
MAX_HISTORY = 4
MAX_SAMPLE_GAP_M = 4.1
ANCHOR_RADIUS_M = 6.
ANCHOR_RESIDUAL_M = 1.25
PAST_AGREEMENT_M = 1.


def project_sections(stations, points):
    output = np.full((len(stations), 2), np.nan)
    for side, curve in enumerate(points.transpose(1, 0, 2)):
        if not len(curve):
            continue
        x, y = curve.T
        # Folded curves have multiple possible road edges at the same station.
        if np.any(np.diff(x) <= 0.):
            continue
        j = np.searchsorted(x, stations)
        for i, k in enumerate(j):
            if k < len(x) and abs(x[k]-stations[i]) < 1.e-8:
                output[i, side] = y[k]
            elif 0 < k < len(x) and x[k]-x[k-1] <= MAX_SAMPLE_GAP_M:
                output[i, side] = np.interp(stations[i], x[k-1:k+1], y[k-1:k+1])
    return output


def valid_pairs(values):
    width = values[..., 1] - values[..., 0]
    return np.isfinite(values).all(axis=-1) & (width > 3.) & (width < 20.)


def fill(stations, current, histories):
    filled = current.copy()
    if not histories:
        return filled
    known = valid_pairs(current)
    for i in np.flatnonzero(~np.isfinite(current).all(axis=1)):
        local = known & (abs(stations-stations[i]) <= ANCHOR_RADIUS_M)
        if not local.any():
            continue
        checked = []
        bracketed = []
        for past in histories:
            paired = valid_pairs(past)
            overlap = paired & local
            if not paired[i] or not overlap.any():
                continue
            if np.max(abs(past[overlap]-current[overlap])) > ANCHOR_RESIDUAL_M:
                continue
            observed = np.isfinite(current[i])
            if np.any(abs(past[i, observed]-current[i, observed]) > ANCHOR_RESIDUAL_M):
                continue
            checked.append(past[i])
            if np.any(overlap[:i]) and np.any(overlap[i+1:]):
                bracketed.append(past[i])
        matching = [a for j, a in enumerate(checked) if any(
                    k != j and np.max(abs(a-b)) <= PAST_AGREEMENT_M for k, b in enumerate(checked))]
        if matching and np.max(np.ptp(np.stack(matching), axis=0)) <= PAST_AGREEMENT_M:
            proposal = np.median(matching, axis=0)
        elif len(checked) == len(bracketed) == 1:
            proposal = bracketed[0]
        else:
            continue
        patched = np.where(np.isfinite(current[i]), current[i], proposal)
        if valid_pairs(patched):
            filled[i] = patched
    return filled


def sequence(stations, continuity, sections, motion):
    """Transport only raw observations, with no interpolation across resets or missing capture IDs."""
    previous = deque(maxlen=MAX_HISTORY)
    output = sections.copy()
    moving = np.isfinite(motion).all(axis=1) & (motion[:, 1] >= old.MIN_SPEED_MPS)
    for i, raw in enumerate(sections):
        step = alt.transform(motion[i], motion[i-1]) if i and continuity[i] and moving[i-1] and moving[i] else None
        if step is None:
            previous.clear()
        else:
            previous = deque(((stamp, alt.advance(points, *step)) for stamp, points in previous
                              if motion[i, 0]-stamp <= MAX_AGE_S), maxlen=MAX_HISTORY)
        history = [project_sections(stations, points) for _, points in previous]
        output[i] = fill(stations, raw, history)
        if moving[i]:
            pair = valid_pairs(raw)
            points = np.stack((np.repeat(stations[pair, None], 2, axis=1), raw[pair]), axis=-1)
            if len(points):
                previous.append((motion[i, 0], points))
    return output


def expand(track, sources):
    group = enrich(track, sources)
    path = PARENT/f"{track}-observations.npz"
    assert old.digest(path) == old.read(path.with_suffix(".json"))["arrays_sha256"]
    with np.load(path, allow_pickle=False) as data:
        raw = {key: data[key] for key in ("sections", "motion", "heads", "primary")}
    rows, stations = group["rows"], old.STATIONS
    contiguous = [False, *(alt.adjacent(a, b) for a, b in zip(rows, rows[1:]))]
    patched = sequence(stations, contiguous, raw["sections"], raw["motion"])
    unchanged = np.isfinite(raw["sections"])
    assert np.array_equal(patched[unchanged], raw["sections"][unchanged])
    new_sections = valid_pairs(patched) & ~valid_pairs(raw["sections"])
    primary = raw["primary"].copy()
    for i in np.flatnonzero(new_sections.any(axis=1)):
        options = alt.segments(patched[i], 10.)
        primary[i] = options[0][:2] if options else [np.nan, np.nan]
    pools, history = alt.sequence(rows, dict(raw, sections=patched, primary=primary))
    direct = [set(map(tuple, offers)) for offers in group["pools"]]
    changed = {i: [(c, h) for c, h in zip(offers, lags, strict=True) if c["offer"] not in direct[i]]
               for i, (offers, lags) in enumerate(zip(pools, history, strict=True)) if new_sections[i].any()}
    columns = ("index", "offers", "edge", "history", "motion", "errors", "origin", "target")
    add = {name: [] for name in columns}
    final_pools = [list(offers) for offers in group["pools"]]
    if any(changed.values()):
        visual = alt.load_features(track, sources)
        by_key = {(r["run"], r["frame"]): x for r, x in zip(visual["selected"], visual["x"], strict=True)}
        for i, options in changed.items():
            features = by_key.get(group["keys"][i])
            if features is None or not np.isfinite(features).all():
                continue
            for candidate, h in options:
                offer, source = candidate["offer"], candidate["source"]
                geom = alt.local_features(stations, patched[i], offer, None, 0.)[:-4]
                direction = relative_features(stations, patched[i], raw["motion"][i],
                                              causal_prefix(rows, raw["motion"], i), offer)
                edge = np.r_[geom, source, group["phase_x"][i], direction]
                assert np.isfinite(edge).all()
                error = old.category(rows[i], offer)
                values = (i, offer, edge, h, features[offer[0]-1], error, source, error == "joint_hit")
                for name, value in zip(columns, values, strict=True):
                    add[name].append(value)
                final_pools[i].append(offer)
    if add["index"]:
        order = np.argsort(np.r_[group["index"], add["index"]], kind="stable")
        updated = {name: np.concatenate((group[name], add[name]), axis=0)[order]
                   for name in columns if name != "offers"}
        offers = list(group["offers"]) + add["offers"]
        updated["offers"] = [offers[i] for i in order]
        assert all(int(o[0]) == o[0] and str(o[0]) in ("1", "2") for o in updated["offers"])
    else:
        updated = {}
    sources.update((Path(__file__), path, path.with_suffix(".json"), Path(alt.advance.__code__.co_filename)))
    return dict(group, **updated, sections=patched, pools=final_pools), group, new_sections
