"""Apply the frozen width and local-image recovery policy to an archived camera stream."""

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIDTH = ROOT / "runs/vision-corner-width-conditioned-boundary-20261112-01"
LOCAL = ROOT / "runs/vision-corner-local-pixel-recovery-20261112-01"


def _recovery_settings(old):
    width_receipt = WIDTH / "training.json"
    checkpoint = WIDTH / "selector.joblib"
    width = old.read(width_receipt)
    if old.digest(checkpoint) != width["checkpoint_sha256"]:
        raise ValueError("Boundary recovery checkpoint does not match its training receipt")

    experiment = LOCAL / "training.json"
    policy_path = LOCAL / "equal-cost/training.json"
    policy = old.read(policy_path)
    if old.digest(experiment) != policy["prior_training_sha256"]:
        raise ValueError("Local recovery policy does not match its development receipt")
    choices = [
        (mode, float(cut), results)
        for mode, thresholds in old.read(experiment)["totals"].items()
        for cut, results in thresholds.items()
    ]
    chosen = max(choices, key=lambda a: (2 * a[2]["correct"] - a[2]["accepted"], a[2]["correct"], a[1]))
    if chosen != (policy["mode"], policy["cut"], policy["development"]) or chosen[0] != "localized":
        raise ValueError("Local recovery settings differ from the validated policy")
    return checkpoint, width["threshold"], policy["cut"], {width_receipt, checkpoint, experiment, policy_path}


def _local_options(sensor, bank, folders, bundles, stream, image, mount, strip_features, offsets, shifts, signs):
    width_rows = np.flatnonzero(bank["source"] == 5)
    positions = bank["positions"][bank["cell"][width_rows]]
    frame_index, station, side = positions.T
    width = bank["value"][width_rows]
    result = dict(pos=positions, coordinate=width.copy(), score=np.full(len(width), -np.inf))
    keys = {(str(run), int(frame)): i for i, (run, frame) in enumerate(zip(sensor["run"], sensor["frame"], strict=True))}
    if len(keys) != len(sensor["frame"]):
        raise ValueError("Recovery requires unique frame identities")

    order = np.argsort(frame_index, kind="stable")
    bounds = np.searchsorted(frame_index[order], np.arange(len(keys) + 1))
    seen = set()
    for run, folder in folders.items():
        for archive, item, _, views in stream(folder, bundles[run], None):
            index = keys[run, item["capture_id"]]
            if index in seen:
                raise ValueError("Recovery archive repeats a frame")
            seen.add(index)
            active = order[bounds[index]:bounds[index + 1]]
            if views is None or not len(active):
                continue
            frame_lab = {}
            for side_index, outward in enumerate(("right", "left")):
                candidates = active[side[active] == side_index]
                if not len(candidates):
                    continue
                location = width[candidates, None] + shifts
                lateral = (location[..., None] + offsets) * signs[side_index]
                forward = np.broadcast_to(sensor["stations"][station[candidates], None, None], lateral.shape)
                points = np.stack((forward, lateral, np.zeros_like(lateral)), axis=-1).reshape(-1, len(offsets), 3)
                values = []
                for view in ("front", outward):
                    if view not in frame_lab:
                        frame_lab[view] = cv2.cvtColor(image(archive, views[view]), cv2.COLOR_BGR2LAB)
                    features = strip_features(frame_lab[view], mount[view], points)
                    brightness = np.maximum(-features[:, 0], 0.)
                    central = features[:, 6]
                    maximum = features[:, 7]
                    score = np.sqrt(brightness * central) * np.minimum(1., (central + 1.) / (maximum + 1.))
                    values.append(np.where(np.isfinite(score), score, -np.inf))
                scores = np.maximum(*values).reshape(len(candidates), len(shifts)) - 12. * abs(shifts)
                best = np.argmax(scores, axis=1)
                result["score"][candidates] = scores[np.arange(len(best)), best]
                result["coordinate"][candidates] = location[np.arange(len(best)), best]
    if seen != set(keys.values()):
        raise ValueError("Recovery archive did not cover the sensor timeline")
    return result


def recover_sections(sensor, folders, bundles, stream, image, old):
    # Reuse the geometry and image sampling that were exercised by the frozen experiments.
    sys.path.insert(0, str(LOCAL))
    from local_transition import SHIFTS, add
    from pixel_checked import OFFSET, strip_features
    from width_conditioned import SIGNS, SensorArrays, apply, bank, choose, joblib, load_observer

    checkpoint, width_cut, image_cut, sources = _recovery_settings(old)
    _, settings, _ = load_observer()
    raw = sensor["sections"]
    candidates = bank(SensorArrays(sensor), settings)
    width = apply(raw, candidates, choose(candidates, joblib.load(checkpoint)), width_cut)
    options = _local_options(sensor, candidates, folders, bundles, stream, image, old.read(old.MOUNT)["views"],
                             strip_features, OFFSET, SHIFTS, SIGNS)
    repaired = add(raw, width, options, image_cut)
    if not np.array_equal(repaired[np.isfinite(raw)], raw[np.isfinite(raw)]):
        raise ValueError("Recovery altered a directly observed boundary")
    sources.update((Path(__file__), old.MOUNT, Path(bank.__code__.co_filename),
                    Path(add.__code__.co_filename), Path(strip_features.__code__.co_filename)))
    settings = dict(mode="local-image", width_cut=width_cut, image_cut=image_cut,
                    width_added=int((~np.isfinite(raw) & np.isfinite(width)).sum()),
                    image_added=int((~np.isfinite(width) & np.isfinite(repaired)).sum()))
    return repaired, settings, sources
