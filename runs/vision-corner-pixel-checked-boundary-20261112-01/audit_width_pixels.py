"""Read the saved cameras at the two accurate but withheld width hypotheses."""
from pixel_checked import (ARCHIVE,MAP,N,OUT,SIGNS,SILVER,VIEWS,SensorArrays,bank,joblib,load_labels,load_observer,
                           np,old,pixel_bank)


def run():
    root=ARCHIVE/"silverstone"
    receipt=root/"result.json"
    source=root/"sensors.npz"
    proof=old.read(receipt)
    assert old.digest(source)==proof["inputs_sha256"]
    _,settings,paths=load_observer()
    paths.update((source,receipt))
    with np.load(source,allow_pickle=False) as d:
        sensor=SensorArrays({k:d[k] for k in d.files})
    original=bank(sensor,settings)
    pos=original["positions"]
    index=pos[original["cell"]]
    frames=sensor["frame"][index[:,0]]
    station=sensor["stations"][index[:,1]]
    focus=np.isin(frames,[620,1049])&(station==10.)&(index[:,2]==0)
    b=dict(original)
    for key in ("cell","source","value","x","counts"):
        b[key]=original[key][focus]
    samples,channels=pixel_bank(SILVER,sensor,b,proof["archive_bundles"],paths)
    label_file=OUT.parent/"vision-corner-physics-surface-20261112-01"/f"{SILVER}-train-labels.npz"
    edge,valid=load_labels(label_file,sensor,paths)
    model=joblib.load(OUT/"selector.joblib")
    score=model.predict_proba(samples["x"])[:,1]
    raw_features=("luminance_delta","luminance_abs","chroma_delta","inner_chroma","outer_chroma",
                  "outer_minus_inner_chroma","center_gradient","max_gradient","strip_span_px")
    rows=[]
    for row,(i,j,k) in enumerate(index[focus]):
        assert valid[i,j,k]
        result={}
        for m,name in enumerate(VIEWS):
            group=channels[row,m*N:(m+1)*N]
            result[name]=dict(zip(raw_features,group,strict=True)) if np.isfinite(group).all() else None
        rows.append(dict(frame=int(sensor["frame"][i]),source=int(b["source"][row]),score=float(score[row]),
                         error_m=float(abs(b["value"][row]*SIGNS[k]-edge[i,j,k])),views=result))
    assert len([r for r in rows if r["source"]==5])==2
    paths.add(OUT/"selector.joblib")
    old.write(OUT/"silverstone-width-pixels.json",dict(rows=rows,source_sha256={str(p):old.digest(p) for p in sorted(paths)}))
    print([r for r in rows if r["source"]==5],flush=True)


if __name__=="__main__":
    run()
