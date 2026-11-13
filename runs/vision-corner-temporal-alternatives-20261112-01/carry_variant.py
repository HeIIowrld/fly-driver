"""Exploratory carry when two past candidates align and the current pool is empty."""
from pathlib import Path

import numpy as np

import alternatives as alt
import rank
import evaluate_rank as replay

ROOT_RUN = alt.OUT
OUT = ROOT_RUN/"carry"
original_sequence = alt.sequence


def steps_for(rows,motion):
    steps = [None]
    for i in range(1,len(rows)):
        valid = (alt.adjacent(rows[i-1],rows[i]) and np.isfinite(motion[i-1:i+1]).all()
                 and min(motion[i-1:i+1,1])>=alt.old.MIN_SPEED_MPS)
        steps.append(alt.transform(motion[i],motion[i-1]) if valid else None)
    return steps


def advance_pair(near, far, early, late, boundaries):
    if near["offer"][0]!=far["offer"][0] or near["point"] is None or far["point"] is None:
        return None
    a = alt.advance(near["point"],*late)
    b = alt.advance(alt.advance(far["point"],*early),*late)
    dx,dy = abs(a-b)
    if dx>alt.MAX_ALONG or dy>alt.MAX_LATERAL:
        return None
    distance = float(np.rint(a[0]+b[0])/2.)
    if not 4.<=distance<=20.:
        return None
    point = alt.onset_point(boundaries,distance)
    if point is None or abs(point[1]-(a[1]+b[1])/2.)>alt.MAX_LATERAL:
        return None
    error = dx/alt.MAX_ALONG + dy/alt.MAX_LATERAL
    return error,dict(offer=(near["offer"][0],distance),point=point,source=np.array([0.,0.,0.,1.]))


def with_carry(rows,raw):
    direct, history = original_sequence(rows,raw)
    for pool in direct:
        for item in pool:
            item["source"] = np.r_[item["source"],0.]
    final = [list(pool) for pool in direct]
    steps = steps_for(rows,raw["motion"])
    for i in range(2,len(rows)):
        if direct[i] or steps[i] is None or steps[i-1] is None:
            continue
        options = {}
        for near in direct[i-1]:
            for far in direct[i-2]:
                pair = advance_pair(near,far,steps[i-1],steps[i],raw["sections"][i])
                if pair is None:
                    continue
                error,candidate = pair
                side = candidate["offer"][0]
                if side not in options or error<options[side][0]:
                    options[side] = pair
        for _,candidate in options.values():
            final[i].append(candidate)
            history[i].append(alt.history_vector(candidate,[direct[i-1],direct[i-2],[]],
                                                  [steps[i:i+1],steps[i-1:i+1],None]))
    return final,history


def main():
    OUT.mkdir(exist_ok=True)
    alt.sequence = with_carry
    rank.OUT = OUT
    rank.main()
    manifest = OUT/"training.json"
    trained = alt.old.read(manifest)
    trained["source_sha256"][str(Path(__file__).resolve())] = alt.old.digest(Path(__file__))
    trained["carry"] = dict(required_prior_frames=2,current_pool_empty=True,recursive=False,
                            maximum_longitudinal_m=alt.MAX_ALONG,maximum_lateral_m=alt.MAX_LATERAL)
    alt.old.write(manifest,trained)
    replay.OUT = OUT
    replay.main()


if __name__ == "__main__":
    main()
