"""Run a frozen next-corner readout on paired original and reflected FlyVis features."""
import argparse
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np

from train_vision_teacher_line import digest, jsonl, predict, read, write


KEYS = ("mean", "scale", "offset", "coefficients")
FEATURE_DIM = 3 * 4 * 8 * 16


def load_features(path, index_digest, count):
    proof = read(path.with_name("flyvis-result.json"))
    if proof["rows_sha256"] != index_digest or proof["features_sha256"] != digest(path):
        raise ValueError("Feature archive does not match the supplied index or receipt")
    with np.load(path, allow_pickle=False) as archive:
        features, valid = archive["features"], archive["available"]
    if features.shape != (count, FEATURE_DIM) or valid.shape != (count,) or valid.dtype != bool:
        raise ValueError("Expected aligned three-frame, four-view T4/T5 features")
    if not np.isfinite(features[valid]).all():
        raise ValueError("Available features must be finite")
    return features, valid, proof


def contrast(features):
    grid = features.astype(float).reshape(-1, 3, 4, 8, 16)
    return (grid - grid.mean(axis=-1, keepdims=True)).reshape(-1, FEATURE_DIM)


def load_joint_distances(spec):
    """Return a paired direction, distance and gate from pinned experimental weights."""
    path = Path(spec["path"])
    if digest(path) != spec["sha256"]:
        raise ValueError("Changed joint distance weights")
    with np.load(path, allow_pickle=False) as data:
        model = tuple(data[k] for k in KEYS)
        centers = data["centers"]
        threshold = float(data["threshold"])
    if centers.shape != (3,) or not np.isfinite(np.r_[centers, threshold]).all() or (
        np.any(centers < 4.) or np.any(centers > 20.) or np.any(np.diff(centers) <= 0.)
    ):
        raise ValueError("Invalid joint distance centers or threshold")

    def estimate(direct, mirror):
        score = .5 * (predict(model, contrast(direct)) + predict(model, contrast(mirror))[:, [0, 4, 5, 6, 1, 2, 3]])
        score = np.maximum(0., score)
        sides = score[:, 1:].reshape(-1, 2, 3)
        winner = sides.sum(axis=2).argmax(axis=1)
        values = sides[np.arange(len(score)), winner]
        mass = values.sum(axis=1)
        distance = np.divide(values @ centers, mass, out=np.zeros_like(mass), where=mass > 0.)
        return winner + 1, distance, (mass > 0.) & (mass - score[:, 0] >= threshold)

    return estimate, path


def load_readout(weights):
    paths = {}
    for name, spec in weights["models"].items():
        path = Path(spec["path"])
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Changed frozen weights: {name}")
        paths[name] = path
    with np.load(paths["direction"], allow_pickle=False) as model:
        classifier = tuple(model[f"cls_{key}"] for key in KEYS)
        bias = float(model["bias"])
    background_offset = weights.get("background_score_offset", 0.)
    if (isinstance(background_offset, bool) or not isinstance(background_offset, (int, float))
            or not np.isfinite(background_offset)):
        raise ValueError("Background score offset must be a finite number")
    bias += background_offset
    with np.load(paths["distance"], allow_pickle=False) as model:
        ordinal = tuple(model[key] for key in KEYS)
        centers = model["centers"]
    with np.load(paths["presence"], allow_pickle=False) as model:
        gate = tuple(model[key] for key in KEYS)
        threshold = float(model["threshold"])

    def predict_pair(direct, mirror, *, return_scores=False):
        if direct is None and mirror is None:
            raise ValueError("At least one feature branch required")
        count = len(direct if direct is not None else mirror)
        if direct is not None and mirror is not None and len(direct) != len(mirror):
            raise ValueError("Feature branches must have matching batch lengths")
        if count > 1:
            # Match the live one-decision arithmetic at pinned decision thresholds.
            rows = [predict_pair(direct[i:i+1] if direct is not None else None,
                                 mirror[i:i+1] if mirror is not None else None,
                                 return_scores=return_scores)
                    for i in range(count)]
            return tuple(np.concatenate([row[field] for row in rows]) for field in range(3))

        def distance(features):
            first, second = predict(ordinal, features).clip(0., 1.).T
            average = .5 * (first + second)
            ordered = first >= second
            first, second = np.where(ordered, first, average), np.where(ordered, second, average)
            return np.column_stack((1.-first, first-second, second)) @ centers

        if direct is None:
            y = contrast(mirror)
            logits, forecast, gate_scores = predict(classifier,y)[:,[0,2,1]], distance(y), predict(gate,y)[:,[0,2,1]]
        elif mirror is None:
            x = contrast(direct)
            logits, forecast, gate_scores = predict(classifier,x), distance(x), predict(gate,x)
        else:
            x, y = contrast(direct), contrast(mirror)
            logits = .5 * (predict(classifier,x) + predict(classifier,y)[:,[0,2,1]])
            forecast = .5 * (distance(x) + distance(y))
            gate_scores = .5 * (predict(gate,x) + predict(gate,y)[:,[0,2,1]])
        scores = logits + [bias, 0., 0.]
        direction = scores.argmax(axis=1)
        direction[(scores == scores.max(axis=1, keepdims=True)).sum(axis=1) > 1] = 0
        presence = gate_scores[:, 1:].max(axis=1)
        return (scores if return_scores else direction), forecast, presence
    return predict_pair, threshold


def forecast_values(prediction, threshold):
    if prediction is None:
        return dict(available=False, direction=0, distance_m=None, presence_score=None, alert=False)
    side, estimate, score = prediction
    return dict(available=True, direction=int(side), distance_m=float(estimate) if side else None,
                presence_score=float(score), alert=bool(side and estimate <= 20. and score >= threshold))


class FrozenCornerStream:
    """Apply the frozen readout to successive delivered camera groups."""

    def __init__(self, pins, visual, provenance):
        weights = read(pins)
        for field in ("author_checkpoint_sha256", "camera_config_sha256"):
            if provenance[field] != weights[field]:
                raise ValueError(f"Feature source differs from pinned {field}")
        self.predict, self.threshold = load_readout(weights)
        self.visual = visual
        self.history = deque(maxlen=3)
        self.generation = self.last_frame = None

    def push_candidate(self, recorder, candidate):
        """Consume the recorder's owned RGB cache without reading camera files."""
        if (candidate["generation"] == self.generation and self.last_frame is not None
                and candidate["capture_id"] <= self.last_frame):
            return forecast_values(None, self.threshold)
        samples = recorder.samples_for(candidate)
        images = {}
        for view in candidate["views"]:
            name = view["view_id"]
            sample = samples[name]
            if (sample["capture_id"] != candidate["capture_id"]
                    or sample["generation"] != candidate["generation"]
                    or sample["received"] != candidate["received"]
                    or sample["render_started_lower"] != view["render_started_lower"]):
                raise ValueError("Owned camera cache metadata differs")
            images[name] = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2BGR)
        return self.push(candidate, lambda name, _: images[name])

    def push(self, row, image_at):
        frame, generation = row["capture_id"], row["generation"]
        if generation != self.generation:
            self.generation, self.last_frame = generation, None
        elif self.last_frame is not None and frame <= self.last_frame:
            return forecast_values(None, self.threshold)
        if self.last_frame is None or frame != self.last_frame+1:
            self.history.clear()
        self.last_frame = frame
        features = self.visual.push(row, image_at)
        if features is None:
            self.history.clear()
            return forecast_values(None, self.threshold)
        self.history.append(features)
        if len(self.history) < self.history.maxlen:
            return forecast_values(None, self.threshold)
        direct, reflected = (np.concatenate(branch)[None] for branch in zip(*self.history, strict=True))
        if not np.isfinite(direct).all() or not np.isfinite(reflected).all():
            self.history.clear()
            return forecast_values(None, self.threshold)
        prediction = tuple(value[0] for value in self.predict(direct, reflected))
        return forecast_values(prediction, self.threshold)


def infer(pins, index, original, reflected, output, joint_distance_pins=None):
    weights = read(pins)
    records = [json.loads(line) for line in index.read_text(encoding="utf8").splitlines()]
    frame_keys = [(r["run"], r["frame"]) for r in records]
    if not frame_keys or len(set(frame_keys)) != len(frame_keys):
        raise ValueError("Expected a nonempty index with unique run/frame pairs")
    index_digest = digest(index)
    direct, a, direct_proof = load_features(original, index_digest, len(records))
    mirror, b, mirror_proof = load_features(reflected, index_digest, len(records))
    for field in ("author_checkpoint_sha256", "camera_config_sha256"):
        if direct_proof[field] != weights[field] or mirror_proof[field] != weights[field]:
            raise ValueError(f"Feature source differs from pinned {field}")
    if (direct_proof.get("read_at") != "last-render" or mirror_proof.get("read_at") != "last-render"
            or not mirror_proof.get("reflection") or direct_proof.get("reflection")
            or direct_proof.get("state_reset") or mirror_proof.get("state_reset")):
        raise ValueError("Expected last-render stateful original and retinal-reflection features")
    if direct_proof.get("verified_track_id") != mirror_proof.get("verified_track_id"):
        raise ValueError("Feature branches disagree on verified camera track")
    if direct_proof["inventory_sha256"] != mirror_proof["inventory_sha256"]:
        raise ValueError("Feature archives reference different capture inventories")
    bundles = lambda proof: {run: record["bundle_sha256"] for run, record in proof["reports"].items()}
    if bundles(direct_proof) != bundles(mirror_proof) or set(bundles(direct_proof)) != {run for run, _ in frame_keys}:
        raise ValueError("Feature archives do not use the same recorded camera bundles")
    predict_pair, threshold = load_readout(weights)
    valid = a & b
    direction, forecast, presence = predict_pair(direct[valid], mirror[valid])
    available = [forecast_values(values, threshold)
                 for values in zip(direction, forecast, presence, strict=True)]
    joint_inputs = ()
    if joint_distance_pins is not None:
        estimate, joint_weights = load_joint_distances(read(joint_distance_pins))
        sides, distances, accepted = estimate(direct[valid], mirror[valid])
        for prediction, side, distance, accept in zip(available, sides, distances, accepted, strict=True):
            if prediction["alert"] and accept and prediction["direction"] == side:
                prediction["distance_m"] = float(distance)
        joint_inputs = (joint_distance_pins, joint_weights)
    available_iter = iter(available)
    predictions = [dict(run=run, frame=frame,
                        **(next(available_iter) if observed else forecast_values(None, threshold)))
                   for (run, frame), observed in zip(frame_keys, valid, strict=True)]
    output.mkdir(parents=True, exist_ok=False)
    jsonl(output/"rows.jsonl", predictions)
    inputs = (pins, index, original, reflected, original.with_name("flyvis-result.json"),
              reflected.with_name("flyvis-result.json"), Path(__file__), *joint_inputs)
    write(output/"result.json", dict(revision=weights["revision"], seed=None,
          inputs_sha256={str(p): digest(p) for p in inputs},
          rows_sha256=digest(output/"rows.jsonl"), n=len(records), available=int(valid.sum()),
          **(dict(verified_track_id=direct_proof["verified_track_id"],
                  camera_bundle_sha256_by_run=bundles(direct_proof))
             if "verified_track_id" in direct_proof else {}),
          termination="frozen paired readout completed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pins", "index", "original", "reflected", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--joint-distance-pins", type=Path)
    infer(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
