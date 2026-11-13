"""Measure whether the missed side-edge coordinates are in the image-supported pool."""
from pathlib import Path

import numpy as np

from rank_transitions import OUT, SIDE, TEST, VAL, RAW_ARCHIVE, CAPTURE, evidence, old, TOLERANCE_M


def main():
    report = old.read(OUT/"result.json")
    training_path = OUT/"training.json"
    trained = old.read(training_path)
    assert old.digest(training_path)==report["training_sha256"]
    for path,digest in trained["source_sha256"].items():
        assert TEST not in path and VAL not in path and old.digest(Path(path))==digest
    prior = old.read(SIDE/"side-result.json")
    for track,r in report["tracks"].items():
        for path,digest in r["source_sha256"].items():
            assert old.digest(Path(path))==digest
        assert old.digest(OUT/f"{track}-frames.jsonl")==r["frames_sha256"]
        if trained["policy"] == [2.,0.,0.]:
            assert r["frames_sha256"]==prior["tracks"][track]["methods"]["normal_anchor"]["frames_sha256"]
    audit_path = SIDE/"right-camera-audit.json"
    missing = {(r["frame"],r["station_m"]):r for r in old.read(audit_path)["points"]}
    source = RAW_ARCHIVE/"silverstone/sensors.npz"
    receipt = old.read(source.with_name("result.json"))
    assert old.digest(source)==receipt["inputs_sha256"]
    with np.load(source,allow_pickle=False) as data:
        keys = list(zip(data["run"],data["frame"],strict=True))
        stations, observed, motion = data["stations"],data["sections"],data["motion"]
    found = {}
    for entry in evidence(TEST,CAPTURE,receipt["archive_bundles"],keys,observed,motion):
        i,j,side,grid,normal,dark,support,separation = entry
        key = keys[i][1],stations[j]
        if side or key not in missing:
            continue
        distance = missing[key]["lateral_m"]
        row = {}
        for label,allowed in (("all",support),("anchor_3m",support&(separation<=3.)),("anchor_2m",support&(separation<=2.))):
            row[label] = dict(candidates=int(allowed.sum()),best_possible_error_m=float(abs(grid[allowed]-distance).min())
                              if allowed.any() else None)
        found[key]=dict(frame=int(key[0]),station_m=float(key[1]),candidates=row)
    assert found.keys()==missing.keys()
    summary = {name:sum(row["candidates"][name]["best_possible_error_m"] is not None and
                       row["candidates"][name]["best_possible_error_m"]<=TOLERANCE_M for row in found.values())
               for name in ("all","anchor_3m","anchor_2m")}
    old.write(OUT/"ceiling-audit.json",dict(status="diagnostic_only",within_tolerance=summary,rows=list(found.values()),
              source_sha256={str(p):old.digest(p) for p in (Path(__file__),source,audit_path,training_path)}))
    print(summary)


if __name__=="__main__":
    main()
