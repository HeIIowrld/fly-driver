"""Audit where high-speed residuals occur on original label and decision clocks."""
import argparse
from pathlib import Path
import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.fit_current_flight_launch_pedals import datasets


def phase_masks(item):
    d = item["data"]
    time, high = d["decision_time"], item["high"]
    speed = d["body"][:, 16]
    throttle, brake = d["labels"][:, 1:].T
    valid = np.flatnonzero(high)
    # Classify adjacent accepted labels only, and do not bridge missing sections.
    left, right = valid[:-1], valid[1:]
    paired = (time[right] - time[left] <= .10) & (time[right] > time[left])
    on = right[paired & (brake[left] <= .1) & (brake[right] > .1)]
    off = right[paired & (brake[left] > .1) & (brake[right] <= .1)]
    masks = dict(all_high=high, medium=high & (speed >= 25) & (speed < 40),
                 fast=high & (speed >= 40), braking=high & (brake > .1),
                 powered=high & (throttle > .8))
    for name, events in (("brake_onset", on), ("brake_release", off)):
        mask = np.zeros(len(time), dtype=bool)
        for j in events:
            mask |= (time >= time[j]) & (time <= time[j]+.25)
        masks[name] = high & mask
    return masks, on, off


def statistics(x, y):
    e = x-y
    return dict(rmse=np.sqrt(np.mean(e**2, axis=0)).tolist(), signed_mean=e.mean(axis=0).tolist(),
                squared_sum=np.sum(e**2, axis=0).tolist())


def evaluate(item, actual, original):
    d = item["data"]
    y, time = d["labels"][:, 1:], d["decision_time"]
    masks, on, off = phase_masks(item)
    row = dict(role=item["receipt"]["role"], onset_events=on.tolist(), release_events=off.tolist(),
               groups={}, onset_paired_lag=[])
    for name, mask in masks.items():
        count = int(mask.sum())
        row["groups"][name] = dict(labels=count) if not count else dict(
            labels=count, old=statistics(original[mask], y[mask]), phase=statistics(actual[mask], y[mask]))
    # Same sample set across the entire lag comparison; no invented label or interpolation.
    window = np.flatnonzero(masks["brake_onset"])
    lag_values = np.array([-.08, -.04, 0., .04, .08])
    if len(window):
        target = time[window,None]+lag_values
        right = np.clip(np.searchsorted(time,target),1,len(time)-1)
        near = np.where(abs(time[right]-target)<abs(time[right-1]-target),right,right-1)
        keep = np.max(abs(time[near]-target),axis=1)<=.018
        if keep.any():
            for lag, indices in zip(lag_values, near[keep].T):
                row["onset_paired_lag"].append(dict(lag_seconds=float(lag), labels=int(keep.sum()),
                    old_brake_rmse=statistics(original[indices,1],y[window[keep],1])["rmse"],
                    phase_brake_rmse=statistics(actual[indices,1],y[window[keep],1])["rmse"]))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", action="append", required=True)
    p.add_argument("--output", required=True)
    a=p.parse_args()
    out=begin_run(a.output,dict(args=vars(a),program_sha256=sha256(__file__),seed=None,vehicle_control_sent=False))
    result=dict(status="failed", fits={}, scope="Clock-aligned observational residuals; no intervention on the plant")
    try:
        for fit in map(Path,a.fit):
            cfg=read(fit/"manifest.json")["config"]
            digest=read(fit/"result.json")["predictions_sha256"]
            if sha256(fit/"predictions.npz")!=digest:
                raise ValueError("Original prediction hash differs")
            with np.load(fit/"predictions.npz") as z:
                result["fits"][str(fit)]=dict(predictions_sha256=digest, episodes=[
                    evaluate(item,z[f"prediction_{i}"][:,1:],item["original"][:,1:])
                    for i,item in enumerate(datasets(Path(cfg["evaluation"])))])
        result["status"]="completed"
    except Exception as e:
        result["reason"]=f"{type(e).__name__}: {e}"
        raise
    finally:
        write_json(out/"result.json",result)

if __name__=="__main__":
    main()
