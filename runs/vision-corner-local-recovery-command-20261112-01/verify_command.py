"""Recheck the local-image CLI from its raw inputs and the prior scored replay."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
EQUAL=OUT.parent/"vision-corner-local-pixel-recovery-20261112-01/equal-cost"
sys.path.insert(0,str(ROOT/"scripts"))
from replay_corner_boundary_states import old
from corner_archive_recovery import recover_sections
from replay_corner_camera_archive import stream, image

sys.path.insert(0,str(OUT.parent/"vision-corner-paired-risk-20261112-01"))
from paired_risk import ARCHIVE,reference,frames

CASES=(("silverstone","ks_silverstone1967"),("vallelunga","ks_vallelunga"),("monza","monza"))
METHODS=("temporal","t4t5_confirm")


def run():
    prior=old.read(EQUAL/"result.json")
    reports={}
    for folder,track in CASES:
        root=OUT/("vallelunga-retry" if folder=="vallelunga" else folder)
        r=old.read(root/"result.json")
        assert r["boundary_recovery"]["mode"]=="local-image"
        assert old.digest(root/"sensors.npz")==r["inputs_sha256"]
        assert old.digest(root/"recovered-sensors.npz")==r["recovered_input_sha256"]
        for path,digest in r["source_sha256"].items():
            actual=Path(path)
            if actual.name=="corner_archive_recovery.py" and old.digest(actual)!=digest:
                actual=OUT/"recovery-before-empty-guard.py"
            assert old.digest(actual)==digest,path
        previous=prior["tracks"][track]
        coordinates=EQUAL/f"{track}-boundaries.npz"
        assert old.digest(coordinates)==previous["boundaries_sha256"]
        with np.load(root/"sensors.npz",allow_pickle=False) as raw, \
             np.load(root/"recovered-sensors.npz",allow_pickle=False) as restored, \
             np.load(coordinates,allow_pickle=False) as reference_edges, \
             np.load(ARCHIVE/folder/"sensors.npz",allow_pickle=False) as original:
            for key in raw.files:
                if key!="sections":
                    np.testing.assert_array_equal(raw[key],restored[key])
                if key=="t4t5":
                    np.testing.assert_allclose(raw[key],original[key],atol=1.e-6,rtol=0.)
                else:
                    np.testing.assert_array_equal(raw[key],original[key])
            for key in ("run","frame","stations","sections"):
                np.testing.assert_array_equal(restored[key],reference_edges[key])
            observed=np.isfinite(raw["sections"])
            np.testing.assert_array_equal(restored["sections"][observed],raw["sections"][observed])
            added=int((~observed&np.isfinite(restored["sections"])).sum())
            if folder=="silverstone":
                mapping=OUT.parent/"vision-silverstone1967-independent-capture-20261112-05/archive-map.json"
                folders={k:Path(v) for k,v in old.read(mapping).items()}
                repaired,settings,_=recover_sections({key:raw[key] for key in raw.files},folders,r["archive_bundles"],stream,image,old)
                np.testing.assert_array_equal(repaired,restored["sections"])
                assert settings==r["boundary_recovery"]
                short={key:raw[key] if key=="stations" else raw[key][:3] for key in raw.files}
                short["sections"]=np.full_like(short["sections"],np.nan)
                def no_stream(*args):
                    raise AssertionError("No camera decode is needed without boundary candidates")
                empty,empty_receipt,_=recover_sections(short,folders,r["archive_bundles"],no_stream,image,old)
                assert np.isnan(empty).all() and empty_receipt["image_added"]==empty_receipt["width_added"]==0
        predictions=root/"predictions"
        receipt=old.read(predictions/"result.json")
        assert old.digest(predictions/"result.json")==r["prediction_receipt_sha256"]
        assert receipt["input_sha256"]==r["recovered_input_sha256"]
        assert old.digest(predictions/"boundaries.npz")==receipt["boundaries_sha256"]
        forecasts=frames(predictions/"forecasts.jsonl",receipt["forecasts_sha256"])
        old_forecasts=frames(EQUAL/f"{track}-frames.jsonl",previous["trace_sha256"])
        table={(a["run"],a["frame"]):a for a in old_forecasts}
        truth,_=reference(track)
        by_key={(a["run"],a["frame"]):a for a in truth}
        assert set(table)==set(by_key)
        counts={m:Counter() for m in METHODS}
        changes=Counter()
        checked=0
        for record in forecasts:
            key=record["run"],record["frame"]
            if key not in table:
                continue
            expected=table[key]
            assert record==expected,key
            checked+=1
            ref=by_key[key]
            pair=[]
            for method,x in record["forecasts"].items():
                state=x["state"]
                side={"left":1,"right":2}.get(state)
                if side:
                    assert record["candidate"]==[side,x["distance_m"]] and 4<=x["distance_m"]<=20
                else:
                    assert state in ("straight","abstain") and x["distance_m"] is None
                category=old.category(ref,(side,x["distance_m"]) if side else None)
                counts[method][category]+=1
                if category=="joint_hit":
                    counts[method][f"joint_{ref['label']}"]+=1
                pair.append(category)
            if pair[0]!=pair[1]:
                changes[f"{pair[0]} -> {pair[1]}"]+=1
        assert counts==previous["metrics"] and checked==len(table)
        reports[track]=dict(frames=len(forecasts),labeled=checked,boundary_points_added=added,
                            local_image_points=r["boundary_recovery"]["image_added"],metrics=counts,paired=changes,
                            cli_receipt_sha256=old.digest(root/"result.json"))
    assert prior["tracks"]["ks_silverstone1967"]["right_approach"]["right"]["new"]==1
    old.write(OUT/"verification.json",dict(status="verified",tracks=reports))
    print("Verified CLI coordinates and forecasts against all three scored replays.",flush=True)


if __name__=="__main__":
    run()
