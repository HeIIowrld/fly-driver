"""Train and freeze the physical-surface boundary model without reading Silverstone."""
from pathlib import Path

import numpy as np
import torch

from model import (BoundaryNet, CENTER_Y, LATERAL_SCALE, ROW_SCALE, STATIONS, SUPPORT_CUT,
                   camera_mask, input_image, loss, pixel_targets, predict)
from prepare import OUT, SILVER, TRAIN, labels, old

SEED = 20261112
EPOCHS = 14
BATCH = 48
GAP = 12


def load_track(track, sources):
    assert track in TRAIN and track != SILVER
    label, label_path = labels(track)
    path = OUT / f"{track}-front.npz"
    receipt = old.read(path.with_suffix(".json"))
    assert old.digest(path) == receipt["images_sha256"]
    sources.update((path, label_path, path.with_suffix(".json"), label_path.with_suffix(".json")))
    with np.load(path, allow_pickle=False) as pixels:
        np.testing.assert_array_equal(pixels["run"], label["run"])
        np.testing.assert_array_equal(pixels["frame"], label["frame"])
        image = pixels["images"]
        available = pixels["available"]
    selected = label["stations"] >= STATIONS[0]
    np.testing.assert_array_equal(label["stations"][selected], STATIONS)
    xy = label["image_px"][:, 0, selected]
    support = camera_mask(xy, label["adjacent_invalid"][:, selected])
    targets = np.stack((label["lateral_m"][:, selected]/LATERAL_SCALE,
                       (xy[..., 1] - CENTER_Y)/ROW_SCALE), axis=-1).astype(np.float32)
    targets[~support] = 0.
    heat, weights = pixel_targets(xy, support)
    arrays = tuple(torch.from_numpy(a) for a in (image, targets, support, heat, weights))
    training, validation = np.zeros(len(image), bool), np.zeros(len(image), bool)
    partition = []
    for run in dict.fromkeys(label["run"]):
        idx = np.flatnonzero(label["run"] == run)
        cut = int(len(idx)*.8)
        if cut <= GAP or len(idx) - cut < 4:
            raise ValueError(f"Insufficient consecutive samples in {run}")
        training[idx[:cut-GAP]] = True
        validation[idx[cut:]] = True
        partition.append(dict(run=run, training_until=int(label["frame"][idx[cut-GAP-1]]),
                              validation_from=int(label["frame"][idx[cut]])))
    return arrays, training & available, validation & available, partition


def to_device(data, indices, device):
    image, metric, valid, pixels, weights = (d[indices] for d in data)
    return (input_image(image, device), metric.to(device), valid.to(device),
            pixels.to(device=device, dtype=torch.float32) / 255.,
            weights.to(device=device, dtype=torch.float32))


def mirror(image, metric, valid, pixels, weights):
    metric = metric.flip((2,))
    metric[..., 0].neg_()
    return image.flip((-1,)), metric, valid.flip((2,)), pixels.flip((1, -1)), weights.flip((1, -1))


@torch.no_grad()
def validate(network, data, indices, device):
    network.eval()
    counts = np.zeros(2)
    meters, rows = np.zeros(2), np.zeros(2)
    retained, excess = np.zeros(2), np.zeros(2)
    for batch in np.array_split(indices, max(1, (len(indices)+BATCH-1)//BATCH)):
        image, metric, valid, _, _ = to_device(data, batch, device)
        raw, _ = network(image)
        lat, row, prob = predict(raw)
        meters += ((lat - metric[..., 0]*LATERAL_SCALE).abs()*valid).sum((0, 1)).cpu().numpy()
        rows += ((row - (metric[..., 1]*ROW_SCALE + CENTER_Y)).abs()*valid).sum((0, 1)).cpu().numpy()
        counts += valid.sum((0, 1)).cpu().numpy()
        retained += ((prob >= SUPPORT_CUT) & valid).sum((0, 1)).cpu().numpy()
        excess += ((prob >= SUPPORT_CUT) & ~valid).sum((0, 1)).cpu().numpy()
    if np.any(counts == 0):
        raise ValueError("Validation lacks one boundary side")
    return dict(lateral_mae_m=(meters/counts).tolist(), row_mae_px=(rows/counts).tolist(),
                labelled_points=counts.astype(int).tolist(), retained=retained.astype(int).tolist(),
                other_predictions=excess.astype(int).tolist())


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = {Path(__file__), Path(BoundaryNet.forward.__code__.co_filename)}
    loaded = [load_track(track, sources) for track in TRAIN]
    data = tuple(torch.cat(values) for values in zip(*(r[0] for r in loaded), strict=True))
    training = np.flatnonzero(np.concatenate([r[1] for r in loaded]))
    validation = np.flatnonzero(np.concatenate([r[2] for r in loaded]))
    assert not np.intersect1d(training, validation).size
    network = BoundaryNet().to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.003)
    best = np.inf
    history = []
    checkpoint = OUT / "boundary.pt"
    print("Training device:", device, "frames", len(training), len(validation), flush=True)
    for epoch in range(1, EPOCHS+1):
        network.train()
        costs = []
        rng.shuffle(training)
        for start in range(0, len(training), BATCH):
            batch = to_device(data, training[start:start+BATCH], device)
            if rng.random() < .5:
                batch = mirror(*batch)
            image, metric, valid, heat, weight = batch
            image = torch.clamp(image * float(rng.uniform(.85, 1.15)) +
                                float(rng.uniform(-.12, .12)), -1., 1.)
            out, img = network(image)
            cost = loss(out, img, metric, valid, heat, weight)
            optimizer.zero_grad(set_to_none=True)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
            optimizer.step()
            costs.append(float(cost.detach().cpu()))
        measures = validate(network, data, validation, device)
        score = float(np.mean(measures["lateral_mae_m"]) + .15*np.mean(measures["row_mae_px"]))
        history.append(dict(epoch=epoch, training_loss=float(np.mean(costs)), validation=measures,
                            selection_score=score))
        if score < best:
            best = score
            torch.save({k: v.detach().cpu() for k, v in network.state_dict().items()}, checkpoint)
        print(epoch, round(float(np.mean(costs)), 4), measures, flush=True)
    best_epoch = min(history, key=lambda x: x["selection_score"])["epoch"]
    result = dict(status="frozen_for_silverstone_replay", training_circuits=TRAIN, excluded_circuits=[SILVER],
                  validation="last 20 percent of each training archive, with a preceding 12-frame exclusion",
                  partitions={t: r[3] for t,r in zip(TRAIN, loaded, strict=True)},
                  training_frames=len(training), validation_frames=len(validation), best_epoch=best_epoch,
                  support_cut=SUPPORT_CUT, stations=STATIONS.tolist(), seed=SEED, device=str(device),
                  torch_version=torch.__version__, source_revision=old.revision(),
                  checkpoint_sha256=old.digest(checkpoint), history=history,
                  source_sha256={str(p):old.digest(p) for p in sorted(sources)},
                  termination="fixed_epoch_budget_complete")
    old.write(OUT / "training.json", result)


if __name__ == "__main__":
    main()
