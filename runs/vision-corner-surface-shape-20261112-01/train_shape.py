"""Fit and freeze shape-aware boundary and onset prediction outside Silverstone."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from shape_net import OUT, ShapeNet, objective, shape_loss
from model import CENTER_Y, LATERAL_SCALE, ROW_SCALE, SUPPORT_CUT, predict
from prepare import SILVER, TRAIN, labels, old
from train import BATCH, EPOCHS, SEED, load_track, mirror, to_device

MIN_RANGE = 4.
MAX_RANGE = 20.
ONSET_CUT = .5


def onset_targets(track, entry, sources):
    label, path = labels(track)
    sources.add(path)
    station = label["stations"]
    bounds = np.where(label["adjacent_invalid"], label["lateral_m"], np.nan)
    directions = np.full(len(bounds), -1, dtype=np.int64)
    distance = np.zeros(len(bounds), dtype=np.float32)
    for i, edges in enumerate(bounds):
        usable = np.isfinite(edges).all(axis=-1)
        usable &= (edges[:, 1]-edges[:, 0] > 3.) & (edges[:, 1]-edges[:, 0] < 20.)
        if usable.sum() < 8 or usable[station <= 14.].sum() < 3:
            continue
        proposal = entry.local(station, edges, 10.)
        directions[i] = 0
        if proposal is not None:
            directions[i], meters = proposal
            distance[i] = (meters-MIN_RANGE)/(MAX_RANGE-MIN_RANGE)
    return directions, distance


@torch.no_grad()
def validate(network, data, directions, distances, indices, device):
    network.eval()
    counts, meters, rows = (np.zeros(2) for _ in range(3))
    slope, bend = 0., 0.
    n = 0
    cm = np.zeros((3, 3), dtype=np.int64)
    distance_error = []
    for start in range(0, len(indices), BATCH):
        batch = indices[start:start+BATCH]
        image, metric, valid, _, _ = to_device(data, batch, device)
        raw, _, onset = network(image)
        lat, row, support = predict(raw)
        meters += ((lat - metric[..., 0]*LATERAL_SCALE).abs()*valid).sum((0, 1)).cpu().numpy()
        rows += ((row - (metric[..., 1]*ROW_SCALE + CENTER_Y)).abs()*valid).sum((0, 1)).cpu().numpy()
        counts += valid.sum((0, 1)).cpu().numpy()
        a, b = shape_loss(raw, metric, valid)
        slope += a.item()*len(batch)
        bend += b.item()*len(batch)
        n += len(batch)
        estimated = onset[:, :3].argmax(1).cpu().numpy()
        target = directions[batch].numpy()
        valid_onset = target >= 0
        np.add.at(cm, (target[valid_onset], estimated[valid_onset]), 1)
        positive = target > 0
        diff = abs(torch.sigmoid(onset[:,3]).cpu().numpy() - distances[batch].numpy())
        distance_error.extend((diff[positive]*(MAX_RANGE-MIN_RANGE)).tolist())
    if not counts.all() or not cm.sum(axis=1).all() or not distance_error:
        raise ValueError("Validation lacks one geometry class")
    return dict(lateral_mae_m=(meters/counts).tolist(), row_mae_px=(rows/counts).tolist(),
                slope_loss=slope/n, bend_loss=bend/n, onset_confusion=cm.tolist(),
                onset_balanced_accuracy=float(np.mean(cm.diagonal()/cm.sum(axis=1))),
                onset_distance_mae_m=float(np.mean(distance_error)))


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entry = old.context()
    sources = {Path(__file__), Path(ShapeNet.forward.__code__.co_filename),
               Path(load_track.__code__.co_filename), Path(old.digest.__code__.co_filename),
               Path(entry.__file__), Path(entry.decide.__code__.co_filename)}
    loaded = [load_track(track, sources) for track in TRAIN]
    targets = [onset_targets(track, entry, sources) for track in TRAIN]
    directions, distances = (torch.from_numpy(np.concatenate(v)) for v in zip(*targets, strict=True))
    data = tuple(torch.cat(values) for values in zip(*(r[0] for r in loaded), strict=True))
    training = np.flatnonzero(np.concatenate([r[1] for r in loaded]))
    validation = np.flatnonzero(np.concatenate([r[2] for r in loaded]))
    histogram = torch.bincount(directions[training][directions[training] >= 0], minlength=3)
    if len(histogram) != 3 or not histogram.all():
        raise ValueError("Training lacks one onset class")
    weights = histogram.float().rsqrt()
    weights = (weights/weights.mean()).to(device)
    network = ShapeNet().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.003)
    checkpoint = OUT / "shape.pt"
    history, best = [], np.inf
    print("Training frames:",len(training),"validation:",len(validation),"class counts:",histogram.tolist(),flush=True)
    for epoch in range(1, EPOCHS+1):
        network.train()
        costs = []
        rng.shuffle(training)
        for start in range(0,len(training),BATCH):
            indices = training[start:start+BATCH]
            batch = to_device(data, indices, device)
            direction = directions[indices].to(device)
            if rng.random() < .5:
                batch = mirror(*batch)
                direction = torch.where(direction > 0, 3-direction, direction)
            image, metric, valid, heat, pixel_weight = batch
            image = torch.clamp(image*float(rng.uniform(.85,1.15)) + float(rng.uniform(-.12,.12)), -1.,1.)
            raw, pixel, onset = network(image)
            cost = objective(raw, pixel, onset, metric, valid, heat, pixel_weight, direction,
                             distances[indices].to(device), weights)
            optimizer.zero_grad(set_to_none=True)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
            optimizer.step()
            costs.append(float(cost.detach().cpu()))
        v = validate(network, data, directions, distances, validation, device)
        score = float(np.mean(v["lateral_mae_m"]) + .15*np.mean(v["row_mae_px"])
                      + .4*(1-v["onset_balanced_accuracy"]) + .1*v["onset_distance_mae_m"])
        history.append(dict(epoch=epoch, loss=float(np.mean(costs)), score=score, validation=v))
        if score < best:
            best = score
            torch.save({k:v.detach().cpu() for k,v in network.state_dict().items()},checkpoint)
        print(epoch, round(score,3), v, flush=True)
    result = dict(status="frozen_for_replay", seed=SEED, source_revision=old.revision(), device=str(device),
                  excluded_circuits=[SILVER], training_circuits=TRAIN, training_frames=len(training),
                  validation_frames=len(validation), validation_partitions={t:r[3] for t,r in zip(TRAIN,loaded)},
                  training_labels=dict(Counter(directions[training].tolist())), onset_cut=ONSET_CUT,
                  support_cut=SUPPORT_CUT, checkpoint_sha256=old.digest(checkpoint), history=history,
                  best_epoch=min(history,key=lambda r:r["score"])["epoch"],
                  source_sha256={str(p):old.digest(p) for p in sorted(sources)},
                  termination="fixed_epoch_budget_complete")
    old.write(OUT/"training.json",result)


if __name__ == "__main__":
    main()
