"""Replay new geometry and onset readouts on the same frozen Silverstone timeline."""
from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np
import torch

from shape_net import OUT, PREVIOUS, ShapeNet
from model import STATIONS, SUPPORT_CUT, input_image, predict
from predict import readout, HEAT_CUT, REFINE_RADIUS_PX
from prepare import ROOT, SILVER, old
from evaluate import score_boundaries
from train_shape import MIN_RANGE, MAX_RANGE, ONSET_CUT

MODES = ("base", "far_wide", "shape_fill", "shape_geometry", "shape_heat", "head", "head_supported")
ARRAYS = ("far_wide", "shape_fill", "shape_geometry", "shape_heat")
CANDIDATE_AGREEMENT_M = 4.
CONFIRM_CUT = .25


@torch.no_grad()
def inference(cache, checkpoint, front):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    with np.load(cache, allow_pickle=False) as data:
        keys = list(zip(data["run"], data["frame"], strict=True))
        images = data["images"]
        available = data["available"]
    assert len(keys) == len(set(keys))
    output = np.full((len(keys),len(STATIONS),2,3), np.nan, dtype=np.float32)
    heads = np.full((len(keys),4), np.nan, dtype=np.float32)
    scale = .5*319.*front["projection_x"]/(STATIONS-front["origin_body_m"][0])
    for start in range(0,len(keys),48):
        indices = np.arange(start,min(start+48,len(keys)))
        indices = indices[available[indices]]
        if not len(indices):
            continue
        raw, heat, onset = model(input_image(torch.from_numpy(images[indices]),device))
        meters, rows, support = (v.cpu().numpy() for v in predict(raw))
        heat = torch.sigmoid(heat).cpu().numpy()
        heads[indices,:3] = torch.softmax(onset[:,:3],dim=1).cpu().numpy()
        heads[indices,3] = MIN_RANGE + (MAX_RANGE-MIN_RANGE)*torch.sigmoid(onset[:,3]).cpu().numpy()
        for i,index in enumerate(indices):
            for j,k in np.argwhere((support[i]>=SUPPORT_CUT)&(abs(meters[i])<=16.)):
                lateral = float(meters[i,j,k])
                row = float(rows[i,j,k])
                origin = front["origin_body_m"][1]
                anchor = 159.5-scale[j]*(lateral-origin)
                if not 0. <= anchor <= 319.:
                    continue
                output[index,j,k,:2] = lateral,row
                x = readout(heat[i,k],anchor,row)
                if x is not None:
                    output[index,j,k,2] = origin+(159.5-x)/scale[j]
    return keys, available, output, heads


def combined(base, prediction):
    arrays = {"far_wide":base}
    mask = old.STATIONS>=STATIONS[0]
    for mode, channel in (("shape_fill",0),("shape_geometry",0),("shape_heat",2)):
        points = np.full_like(base,np.nan)
        points[mask] = prediction[...,channel]
        arrays[mode] = old.combine(base,points,fill=mode=="shape_fill")
    return arrays


def candidate_from_head(probabilities):
    if not np.isfinite(probabilities).all():
        return None
    direction = int(probabilities[:3].argmax())
    return (direction,float(probabilities[3])) if direction and probabilities[direction] >= ONSET_CUT else None


def raw_stats(rows, mode):
    count = Counter()
    for row in rows:
        p = row["offers"].get(mode)
        category = old.category(row,p)
        count[category] += 1
        if row["truth"] == "approach" and row["label"] == "2":
            count[f"right_{category}"] += 1
    return dict(count)


def main():
    freeze, checkpoint = OUT/"training.json", OUT/"shape.pt"
    trained = old.read(freeze)
    assert trained["excluded_circuits"] == [SILVER] and SILVER not in trained["training_circuits"]
    assert old.digest(checkpoint) == trained["checkpoint_sha256"]
    for name, sha in trained["source_sha256"].items():
        assert SILVER not in name and old.digest(ROOT/name) == sha
    assert trained["onset_cut"] == ONSET_CUT and trained["support_cut"] == SUPPORT_CUT
    control = PREVIOUS/"result.json"
    point = old.read(control)
    cache = PREVIOUS/f"{SILVER}-front.npz"
    assert old.digest(cache) == old.read(cache.with_suffix(".json"))["images_sha256"]
    existing = PREVIOUS/"observed-cross-sections.npz"
    assert old.digest(existing) == point["sections_sha256"]
    mount = old.read(old.MOUNT)
    keys, available, estimates, onset = inference(cache,checkpoint,mount["views"]["front"])
    with np.load(existing,allow_pickle=False) as data:
        assert list(zip(data["run"],data["frame"],strict=True)) == keys
        old_boundaries = data["far_wide"]
    entry = old.context()
    runner = entry.import_replay()
    sources = {Path(__file__),freeze,checkpoint,control,cache,cache.with_suffix(".json"),existing,old.MOUNT,
               Path(entry.__file__),Path(runner.__file__),Path(readout.__code__.co_filename),
               Path(score_boundaries.__code__.co_filename),Path(old.signals.__code__.co_filename)}
    truth, priors, mapping, bundles, profiles, training = runner.setup(SILVER,sources)
    assert [runner.key(r) for r in truth] == keys
    expected = {key:i for i,key in enumerate(keys)}
    reference_path = PREVIOUS/"silverstone-frames.jsonl"
    assert old.digest(reference_path) == point["rows_sha256"]
    sources.add(reference_path)
    references = [json.loads(line) for line in reference_path.read_text(encoding="utf8").splitlines()]
    assert [runner.key(r) for r in references] == keys
    timelines = {f"{m}_{mode}":defaultdict(list) for m in runner.MODELS for mode in MODES}
    arrays = {m:np.full_like(old_boundaries,np.nan) for m in ARRAYS}
    paired, recorded = Counter(), []
    for tag,folder in old.read(mapping).items():
        with old.CameraArchive(folder) as archive:
            assert archive.bundle_sha256 == bundles[tag]
            for item in map(json.loads,archive.read("stream.jsonl").splitlines()):
                key = tag,item["capture_id"]
                if key not in expected:
                    continue
                i = expected[key]
                row, prior = truth[i], priors[i]
                filename = Path(item["file"]).name
                assert archive.by_name[filename]["sha256"] == item["file_sha256"]
                meta = archive.json(filename)
                assert (meta["track_id"],meta["index"],meta["generation"]) == (SILVER,key[1],item["generation"])
                motion = old.read_motion(meta) if available[i] else None
                offers = {}
                if motion is not None and motion[1] >= old.MIN_SPEED_MPS:
                    for mode,bounds in combined(old_boundaries[i],estimates[i]).items():
                        arrays[mode][i] = bounds
                        offers[mode] = entry.local(old.STATIONS,bounds,10.)
                    head = candidate_from_head(onset[i])
                    offers["head"] = head
                    shape = offers["shape_geometry"]
                    offers["head_supported"] = head if (head and shape and head[0]==shape[0]
                                                         and abs(head[1]-shape[1]) <= CANDIDATE_AGREEMENT_M) else None
                raw = offers.get("far_wide")
                assert (list(raw) if raw else None) == references[i]["offers"].get("far_wide"),key
                base = {m:prior["proposals"][spec[1]] for m,spec in runner.MODELS.items()}
                decisions = {}
                for mode in MODES:
                    offer = offers.get(mode)
                    for m in runner.MODELS:
                        score = old.signals(profiles[m].get(key),*offer)["local"] if offer else None
                        decision = base[m] or (offer if score is not None and score >= CONFIRM_CUT else None)
                        method = f"{m}_{mode}"
                        decisions[method] = decision
                        timelines[method][tag].append(old.publish(row,decision,
                                            prior["baseline_observable"] or decision is not None))
                        if mode == "far_wide":
                            assert (list(decision) if decision else None) == references[i]["decisions"][method],key
                    if mode != "base" and not any(base.values()):
                        a,b = (decisions[f"{m}_{mode}"] for m in runner.MODELS)
                        if a != b:
                            paired[mode,"t4t5" if b else "edge",old.category(row,a or b)] += 1
                recorded.append(dict(run=tag,frame=key[1],truth=row["truth"],label=row["label"],
                                     truth_endpoints_m=row["truth_endpoints_m"],offers=offers,decisions=decisions))
    assert [runner.key(r) for r in recorded] == keys
    geometry, label_file = score_boundaries(keys,arrays,truth)
    sources.add(label_file)
    row_file, section_file = OUT/"silverstone-frames.jsonl", OUT/"cross-sections.npz"
    old.jsonl(row_file,recorded)
    np.savez_compressed(section_file,run=np.array([k[0] for k in keys]),frame=np.array([k[1] for k in keys]),
                        stations=old.STATIONS,onset=onset,**arrays)
    metrics = {k:runner.totals(v) for k,v in timelines.items()}
    result = dict(status="silverstone_replay_complete",seed=trained["seed"],source_revision=old.revision(),
                  frames=len(keys),training_circuits=trained["training_circuits"],confirmation_training=training,
                  config=dict(onset_cut=ONSET_CUT,support_cut=SUPPORT_CUT,heat_cut=HEAT_CUT,
                              heat_radius_px=REFINE_RADIUS_PX,confirm_cut=CONFIRM_CUT,
                              agreement_m=CANDIDATE_AGREEMENT_M),
                  metrics=metrics,raw={m:raw_stats(recorded,m) for m in MODES[1:]},boundary_metrics=geometry,
                  paired=[dict(mode=k[0],accepted_by=k[1],category=k[2],frames=v) for k,v in sorted(paired.items())],
                  source_sha256={str(p):old.digest(p) for p in sorted(sources)},
                  rows_sha256=old.digest(row_file),sections_sha256=old.digest(section_file),
                  termination="archive_end_and_controls_reproduced")
    old.write(OUT/"result.json",result)
    print(json.dumps({k:v for k,v in metrics.items() if k.startswith("t4t5")},indent=2),flush=True)


if __name__ == "__main__":
    main()
