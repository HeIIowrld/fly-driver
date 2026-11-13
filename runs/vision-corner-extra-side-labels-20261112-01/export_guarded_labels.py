"""Export offline boundary labels while withholding sections near unregistered collision meshes."""
from pathlib import Path
from collections import Counter
import sys

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PHYSICS=OUT.parent/"vision-corner-physics-surface-20261112-01"
sys.path.insert(0,str(ROOT/"scripts"))
from replay_corner_camera_archive import stream, old
sys.path.insert(0,str(PHYSICS))
from export_labels import add_labels, baseline_data, VIEWS
from cross_sections import Surfaces, coordinates, intersect, CELL
from kn5_surfaces import AC, load_model, surface_config, surface_key, track_models

TRACKS=("imola","ks_laguna_seca")
GROUND_HEIGHT_M=.5
EDGE_MARGIN_M=1.


def raw_meshes(track,sources):
    directory=AC/"content"/"tracks"/track
    inventory=PHYSICS/f"{track}-mesh-inventory.json"
    proof=old.read(inventory)
    assert proof["parser_sha256"]==old.digest(PHYSICS/"kn5_surfaces.py")
    config,paths=surface_config(directory)
    files,model_config=track_models(directory)
    assert config==proof["surfaces"]
    for path,sha in proof["config_sha256"].items():
        assert old.digest(Path(path))==sha
    recognized=[]
    unknown=[]
    unknown_names=Counter()
    for path in files:
        expected=proof["models"][path.name]
        if not expected["nodes"]:
            assert old.digest(path)==expected["sha256"]
            continue
        report,meshes=load_model(path)
        assert report["sha256"]==expected["sha256"]
        for node,triangles in meshes:
            key=surface_key(node["name"],config)
            if key is not None:
                recognized.append((key,triangles))
            elif "WALL" not in node["name"].upper():
                unknown.append(("unknown",triangles))
                unknown_names[node["name"]]+=len(triangles)
    sources.update((inventory,*paths,Path(load_model.__code__.co_filename),PHYSICS/"cross_sections.py"))
    if model_config:
        sources.add(model_config)
    uncertain=Surfaces(unknown,{"unknown":{"valid":False}}) if unknown else None
    return Surfaces(recognized,config),uncertain,unknown_names


def nearby_unknown(surface,body,stations):
    origin,forward,left=coordinates(body)
    footprint=np.array([origin[[2,0]]+forward*s+left*y
                        for s in (min(stations),max(stations)) for y in (-25.,25.)])[:,::-1]
    low=np.floor(footprint.min(axis=0)/CELL).astype(int)
    high=np.floor(footprint.max(axis=0)/CELL).astype(int)
    ids=sorted({j for x in range(low[0],high[0]+1) for z in range(low[1],high[1]+1)
                for j in surface.cells.get((x,z),())})
    points=surface.triangles[ids]-origin
    xy=points[:,:,[2,0]]
    return np.stack((xy@forward,xy@left,points[:,:,1]),axis=-1)


def overlaps_ground(segments, edges, origin):
    a,b=(edge["lateral_m"] for edge in edges)
    za,zb=(edge["world"][1]-origin[1] for edge in edges)
    if not len(segments):
        return False
    y0,y1=segments[:,:,0].T
    lo,hi=np.maximum(y0,a-EDGE_MARGIN_M),np.minimum(y1,b+EDGE_MARGIN_M)
    positive=(hi>=lo)&(y1>y0)
    if not positive.any():
        return False
    y0,y1,lo,hi=(x[positive] for x in (y0,y1,lo,hi))
    z0,z1=segments[positive,:,1].T
    slope=(z1-z0)/(y1-y0)
    known_slope=(zb-za)/(b-a)
    def difference(y):
        return z0+slope*(y-y0)-(za+known_slope*(y-a))
    d1,d2=difference(lo),difference(hi)
    return bool(np.any((d1*d2<=0.)|(np.minimum(abs(d1),abs(d2))<=GROUND_HEIGHT_M)))


class ConfidentSections:
    def __init__(self,known,uncertain):
        self.known,self.uncertain=known,uncertain
        self.rejected=0

    def sections(self,body,stations):
        known=self.known.sections(body,stations)
        if self.uncertain is None:
            return known
        tri=nearby_unknown(self.uncertain,body,stations)
        if not len(tri):
            return known
        origin,_,_=coordinates(body)
        minimum,maximum=tri[:,:,0].min(axis=1),tri[:,:,0].max(axis=1)
        for station,edges in zip(stations,known,strict=True):
            if edges is None or not any(e["surface_transition_confirmed"] for e in edges):
                continue
            segments,_=intersect(tri[(minimum<=station)&(maximum>station)],station)
            if overlaps_ground(segments,edges,origin):
                for edge in edges:
                    self.rejected+=int(edge["surface_transition_confirmed"])
                    edge["surface_transition_confirmed"]=False
        return known


def export(track):
    sources={Path(__file__),Path(add_labels.__code__.co_filename),Path(baseline_data.__code__.co_filename),
             Path(stream.__code__.co_filename)}
    known,unknown,names=raw_meshes(track,sources)
    surface=ConfidentSections(known,unknown)
    group,_,mapping,bundles=baseline_data(track,sources)
    sources.add(mapping)
    keys=[(r["run"],r["frame"]) for r in group["rows"]]
    indices={key:i for i,key in enumerate(keys)}
    n=len(keys)
    stations=old.STATIONS
    lateral=np.full((n,len(stations),2),np.nan,dtype=np.float32)
    witness=np.zeros(lateral.shape,bool)
    pixels=np.full((n,4,len(stations),2,2),np.nan,dtype=np.float32)
    seen=set()
    for name,folder in old.read(mapping).items():
        for _,item,meta,views in stream(folder,bundles[name],track):
            key=name,item["capture_id"]
            if key not in indices:
                continue
            assert key not in seen
            seen.add(key)
            if not views:
                continue
            cameras={v["view_id"]:v for v in meta["views"]}
            body=cameras["front"].get("source_body_after_render")
            if not body or not body.get("physics_available"):
                continue
            i=indices[key]
            add_labels(surface,[cameras[v] for v in VIEWS],lateral[i],witness[i],pixels[i],body)
    assert seen==set(indices)
    path=OUT/f"{track}-train-labels.npz"
    np.savez_compressed(path,run=[k[0] for k in keys],frame=[k[1] for k in keys],stations=stations,
                        views=VIEWS,lateral_m=lateral,adjacent_invalid=witness,image_px=pixels)
    mask=witness[:,None]&np.isfinite(pixels).all(-1)
    result=dict(status="offline_only",track=track,frames=n,unregistered_names=dict(names),
                rejected_surface_witnesses=surface.rejected,counts_by_view=dict(zip(VIEWS,mask.sum(axis=(0,2)).tolist(),strict=True)),
                camera_bundle_sha256_by_run=bundles,labels_sha256=old.digest(path),source_revision=old.revision(),
                source_sha256={str(p):old.digest(p) for p in sorted(sources)},termination="verified_known_surface_export")
    old.write(path.with_suffix(".json"),result)
    print(track,n,surface.rejected,result["counts_by_view"],flush=True)


if __name__=="__main__":
    for track in sys.argv[1:] or TRACKS:
        if track not in TRACKS:
            raise ValueError("Unknown offline development circuit")
        export(track)
