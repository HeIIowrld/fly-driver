"""Measure per-view motion feature changes in the frozen approach readout."""
from collections import Counter
from pathlib import Path

import numpy as np

from pair_audit import CALIBRATION_TRACKS, OUT, TRACKS, sources
from evaluate_fused_corner import frame_key, load_frozen, load_score, read_rows
from flybrain_drive.flyvis_camera import VIEW_ORDER
from infer_frozen_next_corner import load_features, load_readout
from train_vision_teacher_line import digest, write, read
from transfer import compare_tracks, score

MODES = ("none", *VIEW_ORDER, "oldest", "latest")

def run(track):
    final,base,protocol_path=sources(track)
    protocol=load_frozen(protocol_path)
    rules_path=Path(protocol["base_protocol"])
    rules=read(rules_path)["comparison"]
    final_receipt,base_receipt=final/"result.json",base/"result.json"
    proof,base_proof=read(final_receipt),read(base_receipt)
    assert digest(protocol_path)==proof["protocol_sha256"]
    assert digest(base_receipt)==proof["base_receipt_sha256"]
    current=load_score(final/"score",proof["score_receipt_sha256"])
    parent=load_score(base/"parent/score",base_proof["score_receipt_sha256"]["parent"])
    feature=base/"features"
    index=feature/"index.jsonl"
    direct=feature/"original/features.npz"
    mirrored=feature/"reflected/features.npz"
    keys=[frame_key(r) for r in read_rows(index)]
    x,xok,_=load_features(direct,digest(index),len(keys))
    y,yok,_=load_features(mirrored,digest(index),len(keys))
    assert [frame_key(r) for r in current]==keys
    available=xok & yok
    x=x[available].reshape(-1,3,4,8,16)
    y=y[available].reshape(-1,3,4,8,16)
    pins=Path(rules["candidate_pins"])
    readout,threshold=load_readout(read(pins))
    raw_score,_,_=readout(x.reshape(-1,1536),y.reshape(-1,1536),return_scores=True)
    margins=raw_score[:,1]-raw_score[:,0]
    summary={}
    for mode in MODES:
        a,b=x.copy(),y.copy()
        if mode in VIEW_ORDER:
            camera=VIEW_ORDER.index(mode)
            a[:,:,camera]=0
            b[:,:,[0,2,1,3][camera]]=0
        if mode in ("oldest","latest"):
            t=0 if mode=="oldest" else 2
            a[:,t]=0
            b[:,t]=0
        logits,dist,pres=readout(a.reshape(-1,1536),b.reshape(-1,1536),return_scores=True)
        current_margin=logits[:,1]-logits[:,0]
        diff=iter(zip(logits,dist,pres,current_margin-margins,strict=True))
        out=[]
        breakdown=Counter()
        improvements=[]
        for row,ok in zip(current,available,strict=True):
            if ok:
                scores,d,p,margin_change=next(diff)
                side=int(scores.argmax())
                may_add=(row["state"]=="straight" or row["state"]=="abstain" and row["abstain_reason"]=="presence_rejected")
                if row["truth"]=="approach" and row["label"]=="1" and not row["direction_hit"]:
                    breakdown["left_missed"]+=1
                    breakdown["raw_left_top"]+=side==1
                    breakdown["left_with_presence"]+=side==1 and p>=threshold and d<=20
                    improvements.append(float(margin_change))
                if mode!="none" and may_add and side and p>=threshold and 0<=d<=20 and (scores==scores[side]).sum()==1:
                    row=score(row,(side,float(d)))
                    breakdown["added"]+=1
            out.append(row)
        result=compare_tracks(parent,out,rules)
        summary[mode]=dict(comparison=result, counts={k:int(v) for k,v in breakdown.items()},
            mean_left_margin_delta=float(np.mean(improvements)) if improvements else None)
    return dict(modes=summary,revision=proof["revision"],source_sha256={str(p):digest(p) for p in
            (protocol_path,rules_path,final_receipt,base_receipt,index,direct,mirrored,pins)})

def main():
    by_track={track:run(track) for track in TRACKS}
    for name,group in by_track.items():
        print(name,{k:(v["counts"].get("left_with_presence"),v["comparison"]["metrics"]["candidate"])
                    for k,v in group["modes"].items()},flush=True)
    write(OUT/"spa-camera-sensitivity.json",dict(tracks=by_track,program_sha256=digest(Path(__file__)),
          seed=None,termination="archived_feature_sensitivity_completed"))

if __name__=="__main__":
    main()
