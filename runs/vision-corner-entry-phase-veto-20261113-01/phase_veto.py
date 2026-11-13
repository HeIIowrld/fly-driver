"""Calibrate an entry-phase veto for supplemental alerts using only development archives."""
from pathlib import Path
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/"vision-corner-left-risk-reallocation-20261113-01"))
from left_policy import (
    CLI, DEVELOP, EVAL, RISK, METHOD, PER_RECORDING, SELECTED, SensorArrays, apply_policy, archive_mapping,
    dependencies, describe, digests, image, keys, label, load_sensor, old, read_development,
    recover_sections, reference, route, score, stream, cumulative, ERRORS,
)

# Reserve internal-alert headroom for transfer. Phase cut uses the existing
# three-state probabilities on the original selected left candidate.
VETO_CUTS=(.2,.35,.5,.65,.8,.9)
PREVIOUS=OUT.parent/"vision-corner-left-risk-reallocation-20261113-01/frozen.json"
HIGHLANDS=OUT.parent/"vision-highlands-short-fixed-corner-evaluation-20261113-01"


def suppress_inside(base, offers, scores, cut):
    return [offer for offer in offers if base[offer[0]]["forecasts"][METHOD]["state"]=="left"
            or offer[-1][4] <= cut]


def choose_point(rows):
    keep=np.unique([0.,1.01,*(p for original,_,_ in rows for p,_ in original.values())])
    add=np.unique([0.,1.01,*(p for _,alternate,_ in rows for p,_ in alternate.values())])
    removed=np.stack([cumulative(orig,truth,keep,True) for orig,_,truth in rows])
    added=np.stack([cumulative(alt,truth,add) for _,alt,truth in rows])
    risk=np.array(list(RISK.values()))
    per_track=np.array([PER_RECORDING["outside"],1,PER_RECORDING["wrong_direction"],PER_RECORDING["distance_error"]])
    best=None
    for i, keep_cut in enumerate(keep):
        each=removed[:,i,None,:]+added
        overall=each.sum(axis=0)
        eligible=((each[:,:,4]>=0).all(axis=0)&(each[:,:,:4]<=per_track).all(axis=(0,2))
                  &(overall[:,:4]<=risk).all(axis=1)&(overall[:,1]<=0))
        for j in np.flatnonzero(eligible):
            rank=(int(overall[j,4]),int((each[:,j,4]>0).sum()),-int(overall[j,1]),
                  -int(overall[j,:4].sum()),float(add[j]),-float(keep_cut))
            if best is None or rank>best[0]:
                best=(rank,dict(keep=float(keep_cut),add=float(add[j]),development_deltas=dict(zip(
                    DEVELOP,(dict(zip(ERRORS, vals,strict=True)) for vals in each[:,j,:].tolist()),strict=True))))
    return best


def calibrate():
    observer,settings,structured,paths=dependencies()
    mapping,map_code=archive_mapping()
    paths.update((Path(__file__),map_code,PREVIOUS))
    dev={}
    for track in DEVELOP:
        sensor,truth,proof=read_development(track,paths,settings)
        raw=SensorArrays(sensor)
        source=mapping[track]
        folders={name:Path(p) for name,p in old.read(source).items()}
        repaired,_,provenance=recover_sections(raw,folders,proof["camera_bundle_sha256_by_run"],stream,image,old)
        paths.update((*provenance,source))
        data=SensorArrays(dict(raw,sections=repaired))
        base,offers=describe(raw,data,observer,settings,structured)
        if keys(base)!=keys(truth):
            raise ValueError("Development sensor and truth timeline differs")
        dev[track]=(base,offers,truth)
        print(track,len(offers),flush=True)
    options={}
    for veto in VETO_CUTS:
        routed=[]
        for base,offers,truth in dev.values():
            available=suppress_inside(base,offers,None,veto)
            routed.append((*route(base,available,np.array([x[3] for x in available]),"raw"),truth))
        options[veto]=choose_point(routed)
    veto=max(options,key=lambda k:options[k][0])
    rank,point=options[veto]
    chosen=dict(method="structured",source="raw",veto_inside_probability=veto,**point)
    receipt=OUT/"frozen.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    old.write(receipt,dict(chosen=chosen,development=list(DEVELOP),
         max_additional_inside_per_dev_recording=1,max_additional_inside_pooled_dev=0,
         deployment_risk_caps=PER_RECORDING,comparison_tracks=[t for t,_ in EVAL]+["ks_highlands"],
         source_revision=old.revision(),sources=digests(paths),settings=settings,
         development_options={str(k):dict(rank=v[0],**v[1]) for k,v in options.items()},
         seed=None,termination="frozen_before_replay"))
    print("Frozen",chosen,rank,flush=True)


def compare():
    receipt=OUT/"frozen.json"
    policy=old.read(receipt)
    for path,sha in policy["sources"].items():
        if old.digest(Path(path))!=sha:
            raise ValueError(f"Frozen input changed: {path}")
    observer,settings,structured,_=dependencies()
    if settings!=policy["settings"]:
        raise ValueError("Observer settings changed")
    reports={}
    for track,name in (*EVAL,("ks_highlands","highlands")):
        directory=(HIGHLANDS/"archive-replay" if name=="highlands" else CLI/
                   ("vallelunga-retry" if name=="vallelunga" else name))
        proof=old.read(directory/"result.json")
        raw=directory/"sensors.npz"
        recovered=directory/"recovered-sensors.npz"
        base,offers=describe(load_sensor(raw,proof["inputs_sha256"]),
                            load_sensor(recovered,proof["recovered_input_sha256"]),observer,settings,structured)
        available=suppress_inside(base,offers,None,policy["chosen"]["veto_inside_probability"])
        revised,changes=apply_policy(base,available,policy,None)
        if name=="highlands":
            path=HIGHLANDS/"labels.jsonl"
            import json
            truth=[json.loads(line) for line in path.read_bytes().splitlines()]
        else:
            truth,path=reference(track)
        if any(a!=b for a,b in zip(base,revised,strict=True) if a["forecasts"][METHOD]["state"]=="right"):
            raise ValueError("Right turn forecast changed")
        before,after=score(base,truth),score(revised,truth)
        delta={k:after["counts"].get(k,0)-before["counts"].get(k,0) for k in RISK}
        trace=OUT/f"{track}-forecasts.jsonl"
        old.jsonl(trace,revised)
        reports[track]=dict(before=before,after=after,net_risk=delta,changes=len(changes),
            caps_pass=all(delta[k]<=cap for k,cap in PER_RECORDING.items()),
            forecasts_sha256=old.digest(trace),sources=digests((path,raw,recovered,directory/"result.json")))
        print(track,after["counts"],delta,flush=True)
    old.write(OUT/"comparison.json",dict(tracks=reports,freeze_sha256=old.digest(receipt),
        highlands_role="previously evaluated transfer archive",termination="archive_end"))


if __name__=="__main__":
    calibrate()
    compare()
