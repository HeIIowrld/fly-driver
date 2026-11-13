"""Choose among visible outward-camera transitions with development-only calibration."""
from collections import Counter
from itertools import product
from pathlib import Path
import sys

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
SIDE = OUT.parent/"vision-corner-outward-camera-20261112-01"
sys.path.insert(0, str(SIDE))
from side_model import old, image, stream, sampling, image_features, labels, baseline_data
from side_model import SIDES, STATIONS, DISTANCES, DEV, TEST, SEED, TOLERANCE_M
from evaluate_side import RAW_ARCHIVE, VAL, ROUTING, CAPTURE, MESH, metrics, mesh_score, guard, select
from image_normal import across_normal
from replay_corner_boundary_states import load_observer, observe
from verify_directional import frames
from aligned_sections import sequence
from alternatives import adjacent

TRIALS = tuple(product((2., 3.), (-4., -2., 0., 1., 2., 4., 8.), (0., .2)))
BASELINE = (2., 0., 0.)
CACHE = OUT.parent/"vision-corner-boundary-hinge-20261111-01"


def evidence(track, mapping, bundles, keys, observed, motion=None):
    lookup = sampling(old.read(old.MOUNT))
    index = {key: i for i, key in enumerate(keys)}
    seen = set()
    for name, folder in old.read(mapping).items():
        for archive, item, _, views in stream(folder, bundles[name], track):
            key = name, item["capture_id"]
            if key not in index:
                continue
            i = index[key]
            assert i not in seen
            seen.add(i)
            if not views or (motion is not None and (not np.isfinite(motion[i]).all() or motion[i,1]<old.MIN_SPEED_MPS)):
                continue
            for side, view in enumerate(SIDES):
                missing = np.flatnonzero(~np.isfinite(observed[i,:len(STATIONS),side]))
                if not len(missing):
                    continue
                bitmap = image(archive, views[view])
                features, valid = image_features(bitmap, lookup[view])
                u, v, mask = lookup[view]
                normal = across_normal(cv2.cvtColor(bitmap, cv2.COLOR_BGR2LAB), u/319., v/179., mask)
                grid = (-1., 1.)[side]*DISTANCES
                for s in missing:
                    support = valid[s] & guard(observed[i], old.STATIONS, s, side, grid, False)
                    finite = np.isfinite(normal[s])
                    support &= finite & (np.r_[False,finite[:-1]] | np.r_[finite[1:],False])
                    anchors = np.isfinite(observed[i,:,side]) & (abs(old.STATIONS-old.STATIONS[s])<=4.)
                    if not support.any() or not anchors.any():
                        continue
                    separation = np.min(abs(grid[:,None]-observed[i,anchors,side]), axis=1)
                    # The outside strip is above the inside strip in each outward view.
                    dark = np.clip(-features[s,:,6], 0., 40.)
                    yield i,s,side,grid,normal[s],dark,support,separation
    assert len(seen)==len(keys)


def read_development(track, hashes):
    group, _, mapping, bundles = baseline_data(track, hashes)
    keys = [(r["run"],r["frame"]) for r in group["rows"]]
    path = CACHE/f"{track}-sections.npz"
    receipt = path.with_suffix(".json")
    assert old.digest(path)==old.read(receipt)["sections_sha256"]
    with np.load(path, allow_pickle=False) as data:
        assert keys==list(zip(data["run"],data["frame"],strict=True))
        stations = data["stations"]
        raw = data["values"][:,0]
    assert np.array_equal(stations, old.STATIONS[:len(stations)])
    observed = np.full((len(keys), len(old.STATIONS),2), np.nan)
    observed[:,:len(stations)] = raw
    lab, label_path = labels(track)
    assert keys==list(zip(lab["run"],lab["frame"],strict=True))
    cam = np.stack((lab["image_px"][:,2,:,0], lab["image_px"][:,1,:,1]),axis=2)
    witness = lab["adjacent_invalid"] & np.isfinite(cam).all(axis=-1)
    target = lab["lateral_m"]
    points = [(entry, target[i,s,side]) for entry in evidence(track,mapping,bundles,keys,observed)
              for i,s,side in [entry[:3]] if witness[i,s,side]]
    hashes.update((path,receipt,label_path,label_path.with_suffix(".json"),mapping))
    return points


def proposal(entry, trial):
    _,_,_,grid,normal,dark,support,separation = entry
    max_anchor, outward, darkness = trial
    score = normal + outward*abs(grid) + darkness*dark
    return select(score,grid,support&(separation<=max_anchor),-np.inf)


def errors(points, trial):
    estimates = np.array([proposal(entry,trial) for entry,_ in points])
    truth = np.array([target for _,target in points])
    good = np.isfinite(estimates)
    signed = (abs(estimates[good])-abs(truth[good]))
    return dict(support=len(points),accepted=int(good.sum()),
                mae_m=float(abs(signed).mean()), inward=int((signed<0).sum()),
                signed_m=float(signed.mean()), within_1_25_m=int((abs(signed)<=TOLERANCE_M).sum()))


def calibrate(points):
    trials = {trial:{track:errors(group,trial) for track,group in points.items()} for trial in TRIALS}
    baseline = trials[BASELINE]
    usable = [trial for trial in trials if all(trials[trial][t]["accepted"]>=baseline[t]["accepted"] for t in points)]
    def rank(trial):
        ratios = [trials[trial][t]["mae_m"]/baseline[t]["mae_m"] for t in points]
        return max(ratios),sum(ratios),trial
    winner = min(usable,key=rank)
    return winner, trials


def apply(observed, entries, trial):
    patched = observed.copy()
    for entry in entries:
        i,s,side = entry[:3]
        assert not np.isfinite(observed[i,s,side])
        patched[i,s,side] = proposal(entry,trial)
    width = patched[:,:,1]-patched[:,:,0]
    invalid = np.isfinite(width)&((width<=3.)|(width>=20.))
    patched[invalid] = observed[invalid]
    np.testing.assert_array_equal(patched[np.isfinite(observed)],observed[np.isfinite(observed)])
    return patched


def main():
    cv2.setNumThreads(1)
    sources = {Path(__file__),old.MOUNT,Path(across_normal.__code__.co_filename),
               Path(image_features.__code__.co_filename),Path(guard.__code__.co_filename)}
    development = {track:read_development(track,sources) for track in DEV}
    winner, candidates = calibrate(development)
    trained = OUT/"training.json"
    assert not any(TEST in str(p) or VAL in str(p) for p in sources)
    old.write(trained,dict(status="frozen_before_evaluation",development=DEV,excluded=[TEST,VAL],seed=SEED,
              source_revision=old.revision(),policy=winner,
              parameters=["anchor_tolerance_m","outward_score_per_m","darkness_weight"],
              trials=[dict(parameters=t,metrics=v) for t,v in candidates.items()],
              source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="development_only_ranking"))
    print("Frozen",winner,candidates[BASELINE],candidates[winner],flush=True)
    models, settings, sources = load_observer()
    prior = old.read(ROUTING/"straight-verification.json")
    side_result = old.read(SIDE/"side-result.json")
    reports = {}
    for track,name,mapping in (
            (TEST,"silverstone",CAPTURE),
            (VAL,"vallelunga",OUT.parent/"vision-vallelunga-local-corner-capture-20261112-01"/"archive-map.json")):
        directory = RAW_ARCHIVE/name
        raw_path = directory/"sensors.npz"
        receipt_path = directory/"result.json"
        receipt = old.read(receipt_path)
        assert old.digest(raw_path)==receipt["inputs_sha256"]
        with np.load(raw_path,allow_pickle=False) as arrays:
            raw = {key:arrays[key] for key in arrays.files}
        keys = list(zip(raw["run"],raw["frame"],strict=True))
        entries = list(evidence(track,mapping,receipt["archive_bundles"],keys,raw["sections"],raw["motion"]))
        control = apply(raw["sections"],entries,BASELINE)
        control_path = SIDE/f"{track}-normal_anchor-sensors.npz"
        assert old.digest(control_path)==side_result["tracks"][track]["methods"]["normal_anchor"]["sensor_sha256"]
        with np.load(control_path,allow_pickle=False) as arrays:
            np.testing.assert_array_equal(control,arrays["sections"])
        selected = apply(raw["sections"],entries,winner)
        sensors = OUT/f"{track}-sensors.npz"
        np.savez_compressed(sensors,**dict(raw,sections=selected))
        with np.load(sensors,allow_pickle=False) as arrays:
            replay,aligned,restored = observe(arrays,models,settings)
        evidence_path = OUT/f"{track}-frames.jsonl"
        old.jsonl(evidence_path,replay)
        reference_path = ROUTING/f"{track}-state-distance.jsonl"
        reference = frames(reference_path,prior["tracks"][track]["rows_sha256"])
        report,hits = metrics(replay,reference)
        hashes = {*sources,raw_path,receipt_path,control_path,reference_path,mapping,trained}
        result = dict(metrics=report,right_hits=hits,frames_sha256=old.digest(evidence_path),
                      sensors_sha256=old.digest(sensors),base_reproduced=True)
        if track==TEST:
            observed_path = directory/"predictions/boundaries.npz"
            with np.load(observed_path,allow_pickle=False) as base:
                result["mesh"] = mesh_score(aligned,base["estimated"],reference,keys)
            hashes.update((MESH, MESH.with_suffix(".json"),observed_path))
        result["source_sha256"]={str(p):old.digest(p) for p in sorted(hashes)}
        reports[track]=result
        print(track,hits,{k:v for k,v in result.get("mesh",{}).items() if k!="points"},flush=True)
    old.write(OUT/"result.json",dict(status="compared",training_sha256=old.digest(trained),tracks=reports,
              termination="calibrated_outward_transition_and_same_candidate_replay"))


if __name__=="__main__":
    main()
