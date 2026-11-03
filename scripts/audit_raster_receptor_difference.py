"""Audit raster/receptor differences at the observed synthetic divergence."""
import hashlib, json
from pathlib import Path
import numpy as np
from flybrain_drive.synthetic_planar_road_visual import SyntheticPlanarRoadVisual
from flybrain_drive.flyvis_camera import FlyvisCameraProjector

B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=B/'20261102-paired-direction-loop-1';b=B/'20261102-straight-balanced-varied-1';out=B/'20261102-single-frame-raster-audit-1';ar,br=read(a/'result.json'),read(b/'result.json')
 if sha(a/'trajectory.npz')!=ar['output_sha256'] or sha(b/'trajectory.npz')!=br['output_sha256']:raise ValueError('Original trajectory hashes required')
 camera_path=Path('configs/ac-csp-exterior-front-v3-retina.json');camera=read(camera_path);road=SyntheticPlanarRoadVisual(camera,curvature=.012,bend_start_m=12.);coords=[(q,r) for q in range(-15,16) for r in range(max(-15,-15-q),min(15,15-q)+1)];u,v=np.asarray(coords).T;projector=FlyvisCameraProjector(camera,u,v)
 with np.load(a/'trajectory.npz',allow_pickle=False) as z,np.load(b/'trajectory.npz',allow_pickle=False) as q:
  current=z['original_882617_012_0'];old=q['candidate_882617_012_0'];rows=[]
  for i in range(60,72):
   def render(r):return road.render(bend=0,forward_m=float(r[1]),lateral_m=float(r[2]),yaw_rad=float(r[3]),texture_seed=882617)
   x,y=render(current[i]),render(old[i]);difference=np.abs(x.astype(np.int16)-y.astype(np.int16));rx=np.stack([projector.sample(frame) for frame in x]);ry=np.stack([projector.sample(frame) for frame in y]);dr=np.abs(rx-ry)
   rows.append(dict(index=i,pose_max_abs=float(np.max(abs(current[i,1:4]-old[i,1:4]))),changed_pixels=int(np.count_nonzero(np.any(x!=y,axis=-1))),max_channel_difference=int(difference.max()),changed_receptors=int(np.count_nonzero(dr)),max_retina_difference=float(dr.max()),intent_difference=float(current[i,6]-old[i,6])))
 report=dict(status='completed_synthetic_raster_comparison',rows=rows,source_a_sha256=ar['output_sha256'],source_b_sha256=br['output_sha256'],camera_sha256=sha(camera_path),program_sha256=sha(__file__),interpretation='Exact renderer and retinal sampling reconstruction at original synthetic poses; no real exposure or physiological claim')
 (out/'raster-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(rows[4],indent=2))
if __name__=='__main__':main()
